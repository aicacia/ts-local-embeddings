/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Suite, Deferred } from "benchmark";
import { WorkerEmbeddings } from "./WorkerEmbeddings.js";
import type { Constructor } from "../types.js";

class FakeWorker {
	onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
	onerror: ((event: ErrorEvent) => void) | null = null;
	onmessageerror: (() => void) | null = null;
	#terminated = false;

	postMessage(request: unknown): void {
		if (this.#terminated) {
			return;
		}

		queueMicrotask(() => {
			if (this.#terminated || !this.onmessage) {
				return;
			}

			const message = request as {
				type: string;
				requestId?: string;
				payload?: Record<string, unknown>;
			};
			switch (message.type) {
				case "init":
					this.onmessage({
						data: {
							type: "ready",
							requestId: message.requestId,
							payload: {
								runtime: {
									modelId: "test-model",
									variant: "cpu",
								},
							},
						},
					} as MessageEvent<unknown>);
					break;
				case "embedDocuments":
					this.onmessage({
						data: {
							type: "progress",
							requestId: message.requestId,
							payload: {
								requestType: "embedDocuments",
								event: {
									type: "batch",
									batchNumber: 1,
									batchDocuments: Array.isArray(message.payload?.documents)
										? message.payload.documents.length
										: 0,
									batchTokens: 0,
									processedAfterBatch: Array.isArray(message.payload?.documents)
										? message.payload.documents.length
										: 0,
									totalDocuments: Array.isArray(message.payload?.documents)
										? message.payload.documents.length
										: 0,
								},
							},
						},
					} as MessageEvent<unknown>);
					this.onmessage({
						data: {
							type: "documentsEmbedded",
							requestId: message.requestId,
							payload: {
								embeddings: Array.isArray(message.payload?.documents)
									? message.payload.documents.map((document: string) => [
											document.length,
										])
									: [],
							},
						},
					} as MessageEvent<unknown>);
					break;
				case "embedQuery":
					this.onmessage({
						data: {
							type: "queryEmbedded",
							requestId: message.requestId,
							payload: {
								embedding: [String(message.payload?.document ?? "").length],
							},
						},
					} as MessageEvent<unknown>);
					break;
			}
		});
	}

	terminate(): void {
		this.#terminated = true;
	}
}

function createWorker(): FakeWorker {
	return new FakeWorker();
}

function warmWorker(): Promise<WorkerEmbeddings> {
	const worker = createWorker();
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
	});
	return embeddings.embedQuery("warmup").then(() => embeddings);
}

export default async function register(Suite: Constructor<Suite>) {
	const warmEmbeddings = await warmWorker();
	return new Promise<void>((resolve, reject) => {
		new Suite()
			.add("worker/embedQuery/cold-init", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const embeddings = new WorkerEmbeddings({
						worker: createWorker() as unknown as Worker,
					});
					await embeddings.embedQuery("hello world");
					embeddings.terminate();
					deferred.resolve();
				},
			})
			.add("worker/embedQuery/warm", {
				defer: true,
				fn: async (deferred: Deferred) => {
					await warmEmbeddings.embedQuery("hello world");
					deferred.resolve();
				},
			})
			.add("worker/embedDocuments/10", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const embeddings = new WorkerEmbeddings({
						worker: createWorker() as unknown as Worker,
					});
					await embeddings.embedDocuments(
						Array.from({ length: 10 }, (_, index) => `document-${index}`),
					);
					embeddings.terminate();
					deferred.resolve();
				},
			})
			.on("cycle", (event: any) => {
				console.log(String(event.target));
			})
			.on("complete", () => {
				warmEmbeddings.terminate();
				resolve();
			})
			.on("error", (event: any) => {
				reject((event.target as Error) ?? new Error("Benchmark suite failed"));
			})
			.run({ async: true });
	});
}
