import test from "tape";
import type {
	SerializedError,
	WorkerRequest,
	WorkerResponse,
	WorkerSuccessResponse,
} from "./embeddingWorkerProtocol.js";
import type { WorkerPort } from "./workerPort.js";
import { WorkerChannel } from "./workerChannel.js";

class FakeWorker implements WorkerPort {
	onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
	onerror: ((event: ErrorEvent) => void) | null = null;
	onmessageerror: (() => void) | null = null;
	requests: WorkerRequest[] = [];

	postMessage(request: WorkerRequest): void {
		this.requests.push(request);
	}

	terminate(): void {}

	emit(response: WorkerResponse): void {
		this.onmessage?.({ data: response } as MessageEvent<WorkerResponse>);
	}
}

function extractRequests(worker: FakeWorker): WorkerRequest[] {
	if (worker.requests.length === 0) {
		throw new Error("expected at least one request");
	}
	return worker.requests;
}

test("WorkerChannel resolves pending requests out of order", async (assert) => {
	const worker = new FakeWorker();
	const channel = new WorkerChannel(worker as unknown as WorkerPort);

	const initPromise = channel.sendRequest("init", {});
	const queryPromise = channel.sendRequest("embedQuery", { document: "abc" });
	const [initRequest, queryRequest] = extractRequests(worker);

	channel.handleResponse({
		type: "queryEmbedded",
		requestId: queryRequest.requestId,
		payload: { embedding: [3] },
	});
	channel.handleResponse({
		type: "ready",
		requestId: initRequest.requestId,
		payload: { runtime: { modelId: "m", variant: "v" } },
	});

	const initResponse = await initPromise;
	const queryResponse = await queryPromise;

	assert.equal(initResponse.type, "ready", "init resolves to ready response");
	assert.equal(
		queryResponse.type,
		"queryEmbedded",
		"query resolves to queryEmbedded response",
	);
	assert.end();
});

test("WorkerChannel ignores unknown request ids safely", async (assert) => {
	const worker = new FakeWorker();
	const channel = new WorkerChannel(worker as unknown as WorkerPort);

	const requestPromise = channel.sendRequest("embedQuery", { document: "x" });
	const [request] = extractRequests(worker);

	channel.handleResponse({
		type: "queryEmbedded",
		requestId: request.requestId + 999,
		payload: { embedding: [999] },
	});
	channel.handleResponse({
		type: "queryEmbedded",
		requestId: request.requestId,
		payload: { embedding: [1] },
	});

	const response = await requestPromise;
	assert.deepEqual(
		response.payload.embedding,
		[1],
		"unknown response id does not affect pending request",
	);
	assert.end();
});

test("WorkerChannel rejects pending requests once on failure", async (assert) => {
	const worker = new FakeWorker();
	const channel = new WorkerChannel(worker as unknown as WorkerPort);

	const first = channel.sendRequest("embedQuery", { document: "first" });
	const second = channel.sendRequest("embedQuery", { document: "second" });
	let rejectedCount = 0;
	const collect = (promise: Promise<WorkerSuccessResponse>) =>
		promise.catch((error) => {
			rejectedCount += 1;
			return error;
		});

	const firstResultPromise = collect(first);
	const secondResultPromise = collect(second);

	channel.handleFailure("worker crashed");
	channel.handleFailure("worker crashed again");

	const [firstResult, secondResult] = await Promise.all([
		firstResultPromise,
		secondResultPromise,
	]);

	assert.equal(rejectedCount, 2, "pending requests are each rejected once");
	assert.ok(firstResult instanceof Error, "first rejection is an Error");
	assert.ok(secondResult instanceof Error, "second rejection is an Error");
	assert.end();
});

test("WorkerChannel timeout rejects hung requests and late responses are ignored", async (assert) => {
	const worker = new FakeWorker();
	const channel = new WorkerChannel(worker as unknown as WorkerPort, {
		requestTimeoutMs: 10,
	});
	const requestPromise = channel.sendRequest("embedQuery", {
		document: "stuck",
	});
	const [request] = extractRequests(worker);

	let timeoutError: unknown;
	try {
		await requestPromise;
		assert.fail("expected timeout rejection");
	} catch (error) {
		timeoutError = error;
	}

	channel.handleResponse({
		type: "queryEmbedded",
		requestId: request.requestId,
		payload: { embedding: [5] },
	});

	assert.ok(timeoutError instanceof Error, "timeout rejects with an Error");
	assert.ok(
		timeoutError instanceof Error && /timed out/i.test(timeoutError.message),
		"timeout message is descriptive",
	);
	assert.end();
});

test("WorkerChannel surfaces serialized error payload metadata", async (assert) => {
	const worker = new FakeWorker();
	const channel = new WorkerChannel(worker as unknown as WorkerPort);
	const requestPromise = channel.sendRequest("embedQuery", {
		document: "error",
	});
	const [request] = extractRequests(worker);

	const payload: SerializedError = {
		message: "model failed",
		name: "ModelError",
		stack: "stack",
		code: "E_MODEL",
		cause: [{ message: "inner", name: "InnerError", code: "E_INNER" }],
	};
	channel.handleResponse({
		type: "error",
		requestId: request.requestId,
		payload,
	});

	try {
		await requestPromise;
		assert.fail("expected serialized error rejection");
	} catch (error) {
		assert.deepEqual(error, payload, "serialized error payload is preserved");
	}
	assert.end();
});
