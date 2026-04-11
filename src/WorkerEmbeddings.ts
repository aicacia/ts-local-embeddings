import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { LoadEmbeddingRuntimeOptions } from "./embeddingRuntime.js";
import type {
	WorkerRequest,
	WorkerRequestMap,
	WorkerResponse,
} from "./embeddingWorkerProtocol.js";

export type WorkerEmbeddingsOptions = {
	runtime?: LoadEmbeddingRuntimeOptions;
	worker?: Worker;
};

type PendingRequest = {
	resolve: (value: unknown) => void;
	reject: (reason?: unknown) => void;
};

type WorkerSuccessType = Exclude<WorkerResponse["type"], "error">;

type WorkerResponsePayloadByType = {
	[T in WorkerSuccessType]: Extract<WorkerResponse, { type: T }>["payload"];
};

type WorkerResultByRequestType = {
	init: WorkerResponsePayloadByType["ready"]["runtime"];
	embedDocuments: WorkerResponsePayloadByType["documentsEmbedded"]["embeddings"];
	embedQuery: WorkerResponsePayloadByType["queryEmbedded"]["embedding"];
};

export class WorkerEmbeddings implements EmbeddingsInterface<number[]> {
	readonly #worker: Worker;
	readonly #pendingRequests = new Map<number, PendingRequest>();
	#requestId = 0;
	#initialization: Promise<void> | null;
	readonly #runtimeOptions: LoadEmbeddingRuntimeOptions | undefined;
	#terminated = false;

	constructor(options: WorkerEmbeddingsOptions = {}) {
		this.#runtimeOptions = options.runtime;

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

		this.#worker.onmessage = (event: MessageEvent<WorkerResponse>): void => {
			this.#handleMessage(event.data);
		};

		this.#worker.onerror = (event): void => {
			this.#handleWorkerFailure(event.message || "Embedding worker crashed.");
		};

		this.#worker.onmessageerror = (): void => {
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
		this.#rejectAllPending(new Error("Embedding worker terminated."));
		this.#worker.terminate();
	}

	async embedDocuments(documents: string[]): Promise<number[][]> {
		await this.#ensureInitialized();
		const embeddings = await this.#request("embedDocuments", { documents });
		return embeddings;
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

		const requestId = ++this.#requestId;
		const request = { type, requestId, payload } as WorkerRequest;

		return new Promise<WorkerResultByRequestType[T]>((resolve, reject) => {
			this.#pendingRequests.set(requestId, {
				resolve: resolve as (value: unknown) => void,
				reject,
			});

			try {
				this.#worker.postMessage(request);
			} catch (error) {
				this.#pendingRequests.delete(requestId);
				reject(error);
			}
		});
	}

	#handleMessage(response: WorkerResponse): void {
		const pending = this.#pendingRequests.get(response.requestId);
		if (!pending) {
			return;
		}

		this.#pendingRequests.delete(response.requestId);

		switch (response.type) {
			case "ready":
				pending.resolve(response.payload.runtime);
				return;
			case "documentsEmbedded":
				pending.resolve(response.payload.embeddings);
				return;
			case "queryEmbedded":
				pending.resolve(response.payload.embedding);
				return;
			case "error": {
				const { message, name, stack } = response.payload;
				const error = new Error(message);
				error.name = name ?? error.name;
				if (stack) {
					error.stack = stack;
				}
				pending.reject(error);
				return;
			}
		}
	}

	#rejectAllPending(error: Error): void {
		for (const pending of this.#pendingRequests.values()) {
			pending.reject(error);
		}
		this.#pendingRequests.clear();
	}

	#handleWorkerFailure(message: string): void {
		if (this.#terminated) {
			return;
		}

		this.#terminated = true;
		this.#rejectAllPending(new Error(message));
		this.#worker.terminate();
	}
}
