import type {
	WorkerRequest,
	WorkerRequestMap,
	WorkerResponse,
	WorkerSuccessResponse,
} from "./embeddingWorkerProtocol.js";

type PendingRequest = {
	resolve: (value: WorkerSuccessResponse) => void;
	reject: (reason?: unknown) => void;
	timeoutHandle: ReturnType<typeof setTimeout> | null;
};

export type WorkerChannelOptions = {
	requestTimeoutMs?: number;
};

export class WorkerChannel {
	readonly #worker: Worker;
	readonly #pendingRequests = new Map<number, PendingRequest>();
	readonly #requestTimeoutMs: number | null;
	#requestId = 0;

	constructor(worker: Worker, options: WorkerChannelOptions = {}) {
		this.#worker = worker;
		this.#requestTimeoutMs =
			typeof options.requestTimeoutMs === "number" &&
			Number.isFinite(options.requestTimeoutMs) &&
			options.requestTimeoutMs > 0
				? options.requestTimeoutMs
				: null;
	}

	sendRequest<T extends keyof WorkerRequestMap>(
		type: T,
		payload: WorkerRequestMap[T],
	): Promise<WorkerSuccessResponse> {
		const requestId = ++this.#requestId;
		const request = { type, requestId, payload } as WorkerRequest;

		return new Promise<WorkerSuccessResponse>((resolve, reject) => {
			const timeoutHandle =
				this.#requestTimeoutMs === null
					? null
					: setTimeout(() => {
							this.#pendingRequests.delete(requestId);
							const error = new Error(
								`Embedding worker request ${requestId} timed out after ${this.#requestTimeoutMs}ms.`,
							);
							error.name = "WorkerRequestTimeoutError";
							reject(error);
						}, this.#requestTimeoutMs);

			this.#pendingRequests.set(requestId, {
				resolve,
				reject,
				timeoutHandle,
			});

			try {
				this.#worker.postMessage(request);
			} catch (error) {
				this.#clearPendingRequest(requestId);
				reject(error);
			}
		});
	}

	handleResponse(response: WorkerResponse): void {
		const pending = this.#clearPendingRequest(response.requestId);
		if (!pending) {
			return;
		}

		if (response.type === "error") {
			pending.reject(response.payload);
			return;
		}

		pending.resolve(response);
	}

	handleFailure(message: string): void {
		const requestIds = Array.from(this.#pendingRequests.keys());
		for (const requestId of requestIds) {
			const pending = this.#clearPendingRequest(requestId);
			pending?.reject(new Error(message));
		}
	}

	#clearPendingRequest(requestId: number): PendingRequest | null {
		const pending = this.#pendingRequests.get(requestId);
		if (!pending) {
			return null;
		}

		this.#pendingRequests.delete(requestId);
		if (pending.timeoutHandle !== null) {
			clearTimeout(pending.timeoutHandle);
		}

		return pending;
	}
}
