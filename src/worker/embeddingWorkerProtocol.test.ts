import test from "tape";
import { isWorkerResponse } from "./embeddingWorkerProtocol.js";

test("isWorkerResponse accepts valid embedDocuments progress payloads", (assert) => {
	const response = {
		type: "progress",
		requestId: 1,
		payload: {
			requestType: "embedDocuments",
			event: {
				type: "batch",
				batchNumber: 1,
				batchDocuments: 3,
				batchTokens: 15,
				processedAfterBatch: 3,
				totalDocuments: 3,
			},
		},
	};

	assert.ok(isWorkerResponse(response), "valid progress payload is accepted");
	assert.end();
});

test("isWorkerResponse rejects malformed embedDocuments progress payloads", (assert) => {
	const response = {
		type: "progress",
		requestId: 1,
		payload: {
			requestType: "embedDocuments",
			event: {
				type: "batch",
				batchNumber: "one",
				batchDocuments: 3,
				batchTokens: 15,
				processedAfterBatch: 3,
				totalDocuments: 3,
			},
		},
	};

	assert.notOk(
		isWorkerResponse(response),
		"invalid progress payload is rejected",
	);
	assert.end();
});
