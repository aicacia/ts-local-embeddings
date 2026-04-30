import test from "tape";
import {
	packEmbeddings,
	packEmbeddingsForTransfer,
	serializeEmbeddingForTransfer,
} from "./transfer.js";

test("packEmbeddings packs rows into a contiguous Float32Array buffer", (assert) => {
	const embeddings = [
		[1, 2, 3],
		[4, 5, 6],
	];
	const result = packEmbeddings(embeddings);
	assert.equal(result.rows, 2, "rows count is preserved");
	assert.equal(result.dims, 3, "dimension count is preserved");
	assert.equal(
		result.buffer.byteLength,
		2 * 3 * 4,
		"buffer byte length is correct",
	);
	const view = new Float32Array(result.buffer);
	assert.deepEqual(
		Array.from(view),
		[1, 2, 3, 4, 5, 6],
		"buffer contains packed embedding values",
	);
	assert.end();
});

test("packEmbeddingsForTransfer includes the packed buffer in transferList when threshold is exceeded", (assert) => {
	const embeddings = [
		[1, 2, 3],
		[4, 5, 6],
	];
	const result = packEmbeddingsForTransfer(embeddings, {
		transferThreshold: 5,
	});
	assert.equal(result.rows, 2, "rows count is preserved");
	assert.equal(result.dims, 3, "dimension count is preserved");
	assert.equal(
		result.transferList.length,
		1,
		"buffer is included in transfer list",
	);
	assert.equal(
		result.transferList[0],
		result.buffer,
		"transfer list contains the packed buffer",
	);
	assert.end();
});

test("serializeEmbeddingForTransfer serializes small Float32Array embeddings as an array", (assert) => {
	const embedding = new Float32Array([1, 2, 3]);
	const result = serializeEmbeddingForTransfer(embedding, {
		transferThreshold: 10,
	});
	assert.deepEqual(
		result.serializedEmbedding,
		{ type: "array", array: [1, 2, 3] },
		"small embeddings are serialized as numeric arrays",
	);
	assert.equal(
		result.transferList.length,
		0,
		"no transfer list for small embeddings",
	);
	assert.end();
});

test("serializeEmbeddingForTransfer transfers large Float32Array embeddings when allowed", (assert) => {
	const embedding = new Float32Array(32).fill(1);
	const result = serializeEmbeddingForTransfer(embedding, {
		transferThreshold: 16,
		transferOwnership: true,
	});
	assert.equal(
		result.transferList.length,
		1,
		"large embeddings use transfer list",
	);
	assert.equal(
		(result.serializedEmbedding as { type: string }).type,
		"buffer",
		"large embeddings are serialized as a buffer",
	);
	assert.equal(
		result.transferList[0],
		(result.serializedEmbedding as { type: string; buffer: ArrayBuffer })
			.buffer,
		"transfer list contains the same buffer as payload",
	);
	assert.end();
});

test("serializeEmbeddingForTransfer transfers large numeric arrays as packed buffers", (assert) => {
	const embedding = Array.from({ length: 32 }, (_, index) => index + 1);
	const result = serializeEmbeddingForTransfer(embedding, {
		transferThreshold: 16,
	});
	assert.equal(
		result.transferList.length,
		1,
		"large numeric arrays use transfer list",
	);
	assert.equal(
		(result.serializedEmbedding as { type: string }).type,
		"buffer",
		"large numeric arrays are serialized as a buffer",
	);
	assert.end();
});
