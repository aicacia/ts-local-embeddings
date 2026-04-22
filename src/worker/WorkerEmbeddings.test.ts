import test from "tape";
import { setDebugLogging } from "../debug.js";
import { WorkerEmbeddings } from "./WorkerEmbeddings.js";
import { normalizeLoadEmbeddingRuntimeOptions } from "./workerRuntimeOptions.js";
import type {
	WorkerRequest,
	WorkerResponse,
} from "./embeddingWorkerProtocol.js";

type FakeWorkerOptions = {
	failFirstInit?: boolean;
	hangEmbedQuery?: boolean;
	embedQueryErrorPayload?: WorkerResponse;
	autoRespondToEmbedDocuments?: boolean;
	deliverMessagesAfterTerminate?: boolean;
};

class FakeWorker {
	onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
	onerror: ((event: ErrorEvent) => void) | null = null;
	onmessageerror: (() => void) | null = null;
	#initAttempts = 0;
	#terminated = false;
	#options: FakeWorkerOptions;
	lastInitOptions: WorkerRequest["payload"]["options"] | null = null;
	requests: WorkerRequest[] = [];

	constructor(options: FakeWorkerOptions = {}) {
		this.#options = options;
	}

	postMessage(request: WorkerRequest): void {
		this.requests.push(request);
		if (this.#terminated) {
			throw new Error("worker terminated");
		}

		queueMicrotask(() => {
			if (
				(this.#terminated && !this.#options.deliverMessagesAfterTerminate) ||
				this.onmessage === null
			) {
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
					if (this.#options.autoRespondToEmbedDocuments === false) {
						return;
					}

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

	emit(response: WorkerResponse): void {
		if (
			(this.#terminated && !this.#options.deliverMessagesAfterTerminate) ||
			this.onmessage === null
		) {
			return;
		}

		this.onmessage({ data: response } as MessageEvent<WorkerResponse>);
	}

	requestCount(type: WorkerRequest["type"]): number {
		return this.requests.filter((request) => request.type === type).length;
	}

	latestRequest(type: WorkerRequest["type"]): WorkerRequest {
		const request = [...this.requests]
			.reverse()
			.find((entry) => entry.type === type);
		if (!request) {
			throw new Error(`expected a ${type} request`);
		}

		return request;
	}
}

async function flushMicrotasks(iterations = 4): Promise<void> {
	for (let index = 0; index < iterations; index += 1) {
		await Promise.resolve();
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

test("WorkerEmbeddings supports raw worker message data without MessageEvent wrapper", async (assert) => {
	const worker = new FakeWorker();
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
	});

	const queryPromise = embeddings.embedQuery("hello");
	await new Promise((resolve) => setTimeout(resolve, 0));
	const request = worker.latestRequest("embedQuery");

	worker.onmessage?.({
		type: "queryEmbedded",
		requestId: request.requestId,
		payload: {
			embedding: [5],
		},
	} as unknown as MessageEvent<unknown>);

	const queryEmbedding = await queryPromise;
	assert.deepEqual(
		queryEmbedding,
		[5],
		"embedQuery uses raw worker message data successfully",
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

test("WorkerEmbeddings treats undefined and equivalent explicit runtime init options as equal", async (assert) => {
	const defaultOptions = normalizeLoadEmbeddingRuntimeOptions(undefined);
	const explicitEmptyOptions = normalizeLoadEmbeddingRuntimeOptions({});
	const explicitDebugFalseOptions = normalizeLoadEmbeddingRuntimeOptions({
		debugLogging: false,
	});

	assert.deepEqual(
		explicitEmptyOptions,
		defaultOptions,
		"empty runtime options normalize to the same defaults as undefined",
	);
	assert.deepEqual(
		explicitDebugFalseOptions,
		defaultOptions,
		"explicit debugLogging=false normalizes to the same defaults as undefined",
	);
	assert.notDeepEqual(
		normalizeLoadEmbeddingRuntimeOptions({ debugLogging: true }),
		defaultOptions,
		"debugLogging=true differs from default runtime options",
	);

	assert.end();
});

test("WorkerEmbeddings supports repeated init requests with identical runtime options", async (assert) => {
	const worker = new FakeWorker();
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
		runtime: { modelId: "test-model", debugLogging: false },
	});

	await embeddings.embedQuery("hello");
	const provenance = await embeddings.getEmbeddingProvenance();

	assert.equal(
		worker.requestCount("init"),
		2,
		"WorkerEmbeddings sends init again for getEmbeddingProvenance",
	);
	assert.equal(
		provenance,
		"test-model:q4",
		"getEmbeddingProvenance returns runtime metadata after repeated init",
	);

	embeddings.terminate();
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
	const events: Array<{ processedAfterBatch: number; totalDocuments: number }> =
		[];
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

test("WorkerEmbeddings serializes embedDocuments requests to keep progress ordered", async (assert) => {
	const worker = new FakeWorker({ autoRespondToEmbedDocuments: false });
	const events: number[] = [];
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
		onProgress: (progress) => {
			events.push(progress.event.totalDocuments);
		},
	});

	const firstRequestPromise = embeddings.embedDocuments(["first"]);
	await flushMicrotasks();
	const secondRequestPromise = embeddings.embedDocuments(["second", "pair"]);
	await flushMicrotasks();

	assert.equal(
		worker.requestCount("embedDocuments"),
		1,
		"only the first embedDocuments request is sent while it is in flight",
	);

	const firstRequest = worker.latestRequest("embedDocuments");
	worker.emit({
		type: "progress",
		requestId: firstRequest.requestId,
		payload: {
			requestType: "embedDocuments",
			event: {
				type: "batch",
				batchNumber: 1,
				batchDocuments: 1,
				batchTokens: 5,
				processedAfterBatch: 1,
				totalDocuments: 1,
			},
		},
	});
	worker.emit({
		type: "documentsEmbedded",
		requestId: firstRequest.requestId,
		payload: {
			embeddings: [[5]],
		},
	});

	assert.deepEqual(await firstRequestPromise, [[5]], "first request resolves");
	await flushMicrotasks();

	assert.equal(
		worker.requestCount("embedDocuments"),
		2,
		"second embedDocuments request is sent after the first one completes",
	);

	const secondRequest = worker.latestRequest("embedDocuments");
	worker.emit({
		type: "progress",
		requestId: secondRequest.requestId,
		payload: {
			requestType: "embedDocuments",
			event: {
				type: "batch",
				batchNumber: 1,
				batchDocuments: 2,
				batchTokens: 10,
				processedAfterBatch: 2,
				totalDocuments: 2,
			},
		},
	});
	worker.emit({
		type: "documentsEmbedded",
		requestId: secondRequest.requestId,
		payload: {
			embeddings: [[6], [4]],
		},
	});

	assert.deepEqual(
		await secondRequestPromise,
		[[6], [4]],
		"second request resolves after it is dispatched",
	);
	assert.deepEqual(
		events,
		[1, 2],
		"progress events stay sequential across requests",
	);

	embeddings.terminate();
	assert.end();
});

test("WorkerEmbeddings ignores late worker messages after terminate", async (assert) => {
	const worker = new FakeWorker({ deliverMessagesAfterTerminate: true });
	const events: number[] = [];
	const embeddings = new WorkerEmbeddings({
		worker: worker as unknown as Worker,
		onProgress: (progress) => {
			events.push(progress.event.totalDocuments);
		},
	});

	await embeddings.embedQuery("ready");
	embeddings.terminate();

	worker.emit({
		type: "progress",
		requestId: 999,
		payload: {
			requestType: "embedDocuments",
			event: {
				type: "batch",
				batchNumber: 1,
				batchDocuments: 1,
				batchTokens: 1,
				processedAfterBatch: 1,
				totalDocuments: 1,
			},
		},
	});
	worker.emit({
		type: "queryEmbedded",
		requestId: 1000,
		payload: {
			embedding: [1],
		},
	});

	assert.deepEqual(
		events,
		[],
		"late messages do not trigger progress callbacks",
	);
	assert.end();
});
