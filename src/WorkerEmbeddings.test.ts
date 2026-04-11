import test from "tape";
import { setDebugLogging } from "./debug.js";
import { WorkerEmbeddings } from "./WorkerEmbeddings.js";
import type {
	WorkerRequest,
	WorkerResponse,
} from "./embeddingWorkerProtocol.js";

type FakeWorkerOptions = {
	failFirstInit?: boolean;
	hangEmbedQuery?: boolean;
	embedQueryErrorPayload?: WorkerResponse;
};

class FakeWorker {
	onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
	onerror: ((event: ErrorEvent) => void) | null = null;
	onmessageerror: (() => void) | null = null;
	#initAttempts = 0;
	#terminated = false;
	#options: FakeWorkerOptions;
	lastInitOptions: WorkerRequest["payload"]["options"] | null = null;

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
						this.lastInitOptions = request.payload.options;
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
							type: "progress",
							requestId: request.requestId,
							payload: {
								requestType: "embedDocuments",
								event: {
									type: "batch",
									batchNumber: 1,
									batchDocuments: request.payload.documents.length,
									batchTokens: request.payload.documents.join("").length,
									processedAfterBatch: request.payload.documents.length,
									totalDocuments: request.payload.documents.length,
								},
							},
						},
					} as MessageEvent<WorkerResponse>);
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
					if (this.#options.hangEmbedQuery) {
						return;
					}

					if (
						this.#options.embedQueryErrorPayload &&
						this.#options.embedQueryErrorPayload.type === "error"
					) {
						this.onmessage({
							data: {
								...this.#options.embedQueryErrorPayload,
								requestId: request.requestId,
							},
						} as MessageEvent<WorkerResponse>);
						return;
					}

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
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
	});

	const queryEmbedding = await embeddings.embedQuery("hello");
	assert.deepEqual(
		queryEmbedding,
		[5],
		"embedQuery uses injected worker successfully",
	);

	embeddings.terminate();
	assert.end();
});

test("WorkerEmbeddings recovers from transient init failures", async (assert) => {
	const worker = new FakeWorker({ failFirstInit: true });
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
	});

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
	assert.deepEqual(
		secondAttempt,
		[6],
		"second attempt re-initializes and succeeds",
	);

	embeddings.terminate();
	assert.end();
});

test("WorkerEmbeddings supports request timeouts", async (assert) => {
	const worker = new FakeWorker({ hangEmbedQuery: true });
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
		requestTimeoutMs: 5,
	});

	try {
		await embeddings.embedQuery("timeout");
		assert.fail("expected timeout error");
	} catch (error) {
		assert.ok(
			error instanceof Error && /timed out/i.test(error.message),
			"request timeout rejects with a timeout error",
		);
	}

	embeddings.terminate();
	assert.end();
});

test("WorkerEmbeddings propagates debug flag to worker init options", async (assert) => {
	const worker = new FakeWorker();
	setDebugLogging(true);

	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
	});

	await embeddings.embedQuery("hello");

	assert.deepEqual(
		worker.lastInitOptions,
		{ debugLogging: true },
		"init request includes debugLogging=true when global debug is enabled",
	);

	embeddings.terminate();
	setDebugLogging(false);
	assert.end();
});

test("WorkerEmbeddings deserializes error metadata", async (assert) => {
	const worker = new FakeWorker({
		embedQueryErrorPayload: {
			type: "error",
			requestId: 0,
			payload: {
				message: "model failed",
				name: "ModelError",
				stack: "stack",
				code: "E_MODEL",
				cause: [{ message: "root", name: "RootCause", code: "E_ROOT" }],
			},
		},
	});
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
	});

	try {
		await embeddings.embedQuery("error");
		assert.fail("expected embedQuery to throw");
	} catch (error) {
		assert.ok(error instanceof Error, "returns an Error instance");
		assert.equal((error as Error).name, "ModelError", "preserves error name");
		assert.equal(
			(error as Error & { code?: string }).code,
			"E_MODEL",
			"preserves error code",
		);
		assert.deepEqual(
			(error as Error & { cause?: unknown }).cause,
			[{ message: "root", name: "RootCause", code: "E_ROOT" }],
			"preserves serialized cause summary",
		);
	}

	embeddings.terminate();
	assert.end();
});

test("WorkerEmbeddings emits progress updates", async (assert) => {
	const worker = new FakeWorker();
	const events: Array<{ processedAfterBatch: number; totalDocuments: number }> = [];
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
		onProgress: (progress) => {
			events.push({
				processedAfterBatch: progress.event.processedAfterBatch,
				totalDocuments: progress.event.totalDocuments,
			});
		},
	});

	await embeddings.embedDocuments(["hello", "world"]);

	assert.deepEqual(
		events,
		[{ processedAfterBatch: 2, totalDocuments: 2 }],
		"progress callback receives worker batch progress events",
	);

	embeddings.terminate();
	assert.end();
});
