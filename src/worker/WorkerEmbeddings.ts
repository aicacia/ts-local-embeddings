import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { LoadEmbeddingRuntimeOptions } from "../runtime/embeddingRuntime.js";
import { isDebugLoggingEnabled } from "../debug.js";
import type {
	WorkerRequestMap,
	WorkerResponseMap,
	SerializedError,
	WorkerSuccessResponse,
} from "./embeddingWorkerProtocol.js";
import type { WorkerPort } from "./workerPort.js";
import {
	isSerializedError,
	isWorkerResponse,
} from "./embeddingWorkerProtocol.js";
import { WorkerChannel } from "./workerChannel.js";
import {
	getWorkerMessageData,
	isMessageEvent,
	mapDocumentsEmbeddedResponse,
	mapQueryEmbeddedResponse,
} from "./workerEmbeddingsUtils.js";
import {
	embedDocuments,
	embedQuery,
	packEmbeddings,
} from "../utils/embeddingUtils.js";

export type WorkerEmbeddingsOptions = {
	runtime?: LoadEmbeddingRuntimeOptions;
	worker?: WorkerPort;
	requestTimeoutMs?: number;
	onProgress?: (progress: WorkerResponseMap["progress"]) => void;
	/**
	 * Request raw transferred embedding buffers instead of nested arrays.
	 * Use `embedDocumentsRaw` to retrieve the raw buffer when available.
	 */
	returnRawBuffer?: boolean;
};

type WorkerResponsePayloadByType = {
	[T in WorkerSuccessResponse["type"]]: Extract<
		WorkerSuccessResponse,
		{ type: T }
	>["payload"];
};

type WorkerResultByRequestType = {
	init: WorkerResponsePayloadByType["ready"]["runtime"];
	embedDocuments: Array<number[] | Float32Array>;
	embedQuery: number[] | Float32Array;
};

export class WorkerEmbeddings
	implements EmbeddingsInterface<number[] | Float32Array>
{
	/**
	 * WorkerEmbeddings provides embedding generation using a Web Worker pipeline.
	 * Implements EmbeddingsInterface for compatibility with LangChain.
	 */
	readonly #worker: WorkerPort;
	readonly #channel: WorkerChannel;
	#initialization: Promise<void> | null;
	#documentRequestQueue: Promise<void> = Promise.resolve();
	readonly #runtimeOptions: LoadEmbeddingRuntimeOptions | undefined;
	readonly #onProgress: WorkerEmbeddingsOptions["onProgress"];
	#terminated = false;

	/**
	 * Create a WorkerEmbeddings instance.
	 * @param options - Worker and runtime options.
	 */
	constructor(options: WorkerEmbeddingsOptions = {}) {
		const debugLogging = isDebugLoggingEnabled();
		this.#runtimeOptions = options.runtime
			? {
					...options.runtime,
					debugLogging: options.runtime.debugLogging ?? debugLogging,
				}
			: debugLogging
				? { debugLogging: true }
				: undefined;

		if (options.worker) {
			this.#worker = options.worker;
		} else {
			if (typeof Worker === "undefined") {
				throw new Error("Web Workers are not available in this environment.");
			}

			this.#worker = new Worker(
				new URL("./embedding.worker.js", import.meta.url),
				{
					type: "module",
				},
			);
		}

		this.#channel = new WorkerChannel(this.#worker, {
			requestTimeoutMs: options.requestTimeoutMs,
		});
		this.#onProgress = options.onProgress;

		this.#worker.onmessage = (event: MessageEvent<unknown>): void => {
			if (this.#terminated) {
				return;
			}

			const message = getWorkerMessageData(event);
			if (!isWorkerResponse(message)) {
				if (isMessageEvent(event)) {
					this.#handleWorkerFailure(
						(event.data as { payload: SerializedError }).payload.message,
					);
				}
				return;
			}

			if (message.type === "progress") {
				this.#onProgress?.(message.payload);
				return;
			}

			this.#channel.handleResponse(message);
		};

		this.#worker.onerror = (event): void => {
			if (this.#terminated) {
				return;
			}

			this.#handleWorkerFailure(event.message || "Embedding worker crashed.");
		};

		this.#worker.onmessageerror = (): void => {
			if (this.#terminated) {
				return;
			}

			this.#handleWorkerFailure(
				"Embedding worker message deserialization failed.",
			);
		};

		this.#initialization = this.#startInitialization();
	}

	terminate(): void {
		if (this.#terminated) {
			return;
		}

		this.#terminated = true;
		this.#detachWorkerHandlers();
		this.#channel.handleFailure("Embedding worker terminated.");
		this.#worker.terminate();
	}

	/**
	 * Embed an array of documents.
	 * @param documents - Array of strings to embed.
	 * @returns Array of embeddings (number[] or Float32Array).
	 */
	async embedDocuments(
		documents: string[],
	): Promise<Array<number[] | Float32Array>> {
		return this.#enqueueDocumentRequest(async () => {
			await this.#ensureInitialized();
			return this.#request("embedDocuments", {
				documents,
			});
		});
	}

	/**
	 * Quick path: return transferred Float32Array buffer (rows/dims) when the
	 * worker provides it. This avoids materializing large nested `number[][]`
	 * allocations on the main thread.
	 */
	/**
	 * Return a packed ArrayBuffer with embeddings laid out row-major (rows * dims).
	 * Uses transferred buffer from worker if available, otherwise packs nested arrays.
	 * @param documents - Array of strings to embed.
	 * @returns Object with buffer, rows, and dims.
	 */
	async embedDocumentsRaw(documents: string[]): Promise<{
		buffer: ArrayBuffer;
		rows: number;
		dims: number;
	}> {
		return this.#enqueueDocumentRequest(async () => {
			await this.#ensureInitialized();
			// Use channel directly to avoid the default mapping that converts
			// transferred buffers into nested arrays.
			const response = await this.#channel.sendRequest("embedDocuments", {
				documents,
			});

			if (response.type !== "documentsEmbedded") {
				throw new Error(
					`Embedding worker returned ${response.type} for embedDocuments request.`,
				);
			}

			const payload =
				response.payload as WorkerResponseMap["documentsEmbedded"];

			if (payload.embeddingsBuffer?.buffer) {
				const { buffer, rows, dims } = payload.embeddingsBuffer;
				return { buffer: buffer as ArrayBuffer, rows, dims };
			}

			// Fallback: convert legacy nested arrays into a packed Float32Array
			if (Array.isArray(payload.embeddings)) {
				const packed = packEmbeddings(
					payload.embeddings as ArrayLike<number>[],
				);
				return { buffer: packed.buffer, rows: packed.rows, dims: packed.dims };
			}

			return { buffer: new ArrayBuffer(0), rows: 0, dims: 0 };
		});
	}

	/**
	 * Embed a single query/document.
	 * @param document - String to embed.
	 * @returns Embedding (number[] or Float32Array).
	 */
	async embedQuery(document: string): Promise<number[] | Float32Array> {
		await this.#ensureInitialized();
		return this.#request("embedQuery", { document });
	}

	async getEmbeddingProvenance(): Promise<string> {
		await this.#ensureInitialized();
		const runtime = await this.#request("init", {
			options: this.#runtimeOptions,
		});
		return `${runtime.modelId}:${runtime.variant}`;
	}

	#startInitialization(): Promise<void> {
		return this.#request("init", {
			options: this.#runtimeOptions,
		}).then(() => undefined);
	}

	async #ensureInitialized(): Promise<void> {
		if (this.#terminated) {
			throw new Error("Embedding worker already terminated.");
		}

		if (this.#initialization === null) {
			this.#initialization = this.#startInitialization();
		}

		const currentInitialization = this.#initialization;
		try {
			await currentInitialization;
		} catch (error) {
			if (this.#initialization === currentInitialization) {
				this.#initialization = null;
			}
			throw error;
		}
	}

	#request<T extends keyof WorkerRequestMap>(
		type: T,
		payload: WorkerRequestMap[T],
	): Promise<WorkerResultByRequestType[T]> {
		if (this.#terminated) {
			return Promise.reject(new Error("Embedding worker already terminated."));
		}

		return this.#channel
			.sendRequest(type, payload)
			.then((response) => this.#mapResponseToResult(type, response))
			.catch((error) => {
				if (isSerializedError(error)) {
					throw this.#deserializeError(error);
				}
				throw error;
			});
	}

	#handleWorkerFailure(message: string): void {
		if (this.#terminated) {
			return;
		}

		this.#terminated = true;
		this.#detachWorkerHandlers();
		this.#channel.handleFailure(message);
		this.#worker.terminate();
	}

	#detachWorkerHandlers(): void {
		this.#worker.onmessage = null;
		this.#worker.onerror = null;
		this.#worker.onmessageerror = null;
	}

	#enqueueDocumentRequest<T>(operation: () => Promise<T>): Promise<T> {
		const run = this.#documentRequestQueue
			.catch(() => undefined)
			.then(operation);
		this.#documentRequestQueue = run.then(
			() => undefined,
			() => undefined,
		);
		return run;
	}

	#mapResponseToResult<T extends keyof WorkerRequestMap>(
		type: T,
		response: WorkerSuccessResponse,
	): WorkerResultByRequestType[T] {
		switch (type) {
			case "init":
				if (response.type !== "ready") {
					throw new Error(
						`Embedding worker returned ${response.type} for init request.`,
					);
				}
				return response.payload.runtime as WorkerResultByRequestType[T];
			case "embedDocuments": {
				return mapDocumentsEmbeddedResponse(
					response,
				) as WorkerResultByRequestType[T];
			}
			case "embedQuery": {
				return mapQueryEmbeddedResponse(
					response,
				) as WorkerResultByRequestType[T];
			}
		}
	}

	#deserializeError(payload: SerializedError): Error {
		const error = new Error(payload.message);
		error.name = payload.name ?? error.name;
		if (payload.stack) {
			error.stack = payload.stack;
		}
		if (payload.code) {
			(error as Error & { code?: string }).code = payload.code;
		}
		if (payload.cause && payload.cause.length > 0) {
			(error as Error & { cause?: unknown }).cause = payload.cause;
		}
		return error;
	}
}
