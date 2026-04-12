import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { LoadEmbeddingRuntimeOptions } from "./embeddingRuntime.js";
import { isDebugLoggingEnabled } from "./debug.js";
import type {
	WorkerRequestMap,
	WorkerResponseMap,
	SerializedError,
	WorkerSuccessResponse,
} from "./embeddingWorkerProtocol.js";
import {
	isSerializedError,
	isWorkerResponse,
} from "./embeddingWorkerProtocol.js";
import { WorkerChannel } from "./workerChannel.js";

export type WorkerEmbeddingsOptions = {
	runtime?: LoadEmbeddingRuntimeOptions;
	worker?: Worker;
	requestTimeoutMs?: number;
	onProgress?: (progress: WorkerResponseMap["progress"]) => void;
};

type WorkerResponsePayloadByType = {
	[T in WorkerSuccessResponse["type"]]: Extract<
		WorkerSuccessResponse,
		{ type: T }
	>["payload"];
};

type WorkerResultByRequestType = {
	init: WorkerResponsePayloadByType["ready"]["runtime"];
	embedDocuments: WorkerResponsePayloadByType["documentsEmbedded"]["embeddings"];
	embedQuery: WorkerResponsePayloadByType["queryEmbedded"]["embedding"];
};

export class WorkerEmbeddings implements EmbeddingsInterface<number[]> {
	readonly #worker: Worker;
	readonly #channel: WorkerChannel;
	#initialization: Promise<void> | null;
	#documentRequestQueue: Promise<void> = Promise.resolve();
	readonly #runtimeOptions: LoadEmbeddingRuntimeOptions | undefined;
	readonly #onProgress: WorkerEmbeddingsOptions["onProgress"];
	#terminated = false;

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

			if (!isWorkerResponse(event.data)) {
				this.#handleWorkerFailure("Embedding worker sent an invalid response.");
				return;
			}

			if (event.data.type === "progress") {
				this.#onProgress?.(event.data.payload);
				return;
			}

			this.#channel.handleResponse(event.data);
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

	async embedDocuments(documents: string[]): Promise<number[][]> {
		return this.#enqueueDocumentRequest(async () => {
			await this.#ensureInitialized();
			const embeddings = await this.#request("embedDocuments", { documents });
			return embeddings;
		});
	}

	async embedQuery(document: string): Promise<number[]> {
		await this.#ensureInitialized();
		const embedding = await this.#request("embedQuery", { document });
		return embedding;
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
		const run = this.#documentRequestQueue.catch(() => undefined).then(operation);
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
			case "embedDocuments":
				if (response.type !== "documentsEmbedded") {
					throw new Error(
						`Embedding worker returned ${response.type} for embedDocuments request.`,
					);
				}
				return response.payload.embeddings as WorkerResultByRequestType[T];
			case "embedQuery":
				if (response.type !== "queryEmbedded") {
					throw new Error(
						`Embedding worker returned ${response.type} for embedQuery request.`,
					);
				}
				return response.payload.embedding as WorkerResultByRequestType[T];
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
