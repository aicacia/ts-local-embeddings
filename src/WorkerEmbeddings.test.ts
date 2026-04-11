import test from "tape";
import { WorkerEmbeddings } from "./WorkerEmbeddings.js";
import type { WorkerRequest, WorkerResponse } from "./embeddingWorkerProtocol.js";

type FakeWorkerOptions = {
	failFirstInit?: boolean;
};

class FakeWorker {
	onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
	onerror: ((event: ErrorEvent) => void) | null = null;
	onmessageerror: (() => void) | null = null;
	#initAttempts = 0;
	#terminated = false;
	#options: FakeWorkerOptions;

	constructor(options: FakeWorkerOptions = {}) {
		this.#options = options;
	}

	postMessage(request: WorkerRequest): void {
		if (this.#terminated) {
			throw new Error("worker terminated");
		}

		queueMicrotask(() => {
			if (this.#terminated || this.onmessage === null) {
				return;
			}

			switch (request.type) {
				case "init": {
					this.#initAttempts += 1;
					if (this.#options.failFirstInit && this.#initAttempts === 1) {
						this.onmessage({
							data: {
								type: "error",
								requestId: request.requestId,
								payload: { message: "transient init failure" },
							},
						} as MessageEvent<WorkerResponse>);
						return;
					}

					this.onmessage({
						data: {
							type: "ready",
							requestId: request.requestId,
							payload: {
								runtime: {
									modelId: "test-model",
									variant: "q4",
								},
							},
						},
					} as MessageEvent<WorkerResponse>);
					return;
				}
				case "embedDocuments": {
					this.onmessage({
						data: {
							type: "documentsEmbedded",
							requestId: request.requestId,
							payload: {
								embeddings: request.payload.documents.map((document) => [
									document.length,
								]),
							},
						},
					} as MessageEvent<WorkerResponse>);
					return;
				}
				case "embedQuery": {
					this.onmessage({
						data: {
							type: "queryEmbedded",
							requestId: request.requestId,
							payload: {
								embedding: [request.payload.document.length],
							},
						},
					} as MessageEvent<WorkerResponse>);
					return;
				}
			}
		});
	}

	terminate(): void {
		this.#terminated = true;
	}
}

test("WorkerEmbeddings supports injected workers without global Worker", async (assert) => {
	const worker = new FakeWorker();
	const embeddings = new WorkerEmbeddings({ worker: worker as unknown as Worker });

	const queryEmbedding = await embeddings.embedQuery("hello");
	assert.deepEqual(queryEmbedding, [5], "embedQuery uses injected worker successfully");

	embeddings.terminate();
	assert.end();
});

test("WorkerEmbeddings recovers from transient init failures", async (assert) => {
	const worker = new FakeWorker({ failFirstInit: true });
	const embeddings = new WorkerEmbeddings({ worker: worker as unknown as Worker });

	try {
		await embeddings.embedQuery("first");
		assert.fail("expected first embedQuery attempt to fail");
	} catch (error) {
		assert.ok(
			error instanceof Error && /transient init failure/i.test(error.message),
			"first init attempt fails",
		);
	}

	const secondAttempt = await embeddings.embedQuery("second");
	assert.deepEqual(secondAttempt, [6], "second attempt re-initializes and succeeds");

	embeddings.terminate();
	assert.end();
});
