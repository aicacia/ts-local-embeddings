import { Document } from "@langchain/core/documents";
import test from "tape";
import {
	createVectorWritePipeline,
	resolveRecordId,
	type StoredVectorRecord,
} from "./vectorWritePipeline.js";

function createDocument(
	pageContent: string,
	metadata: Record<string, unknown> = {},
) {
	return new Document({ pageContent, metadata });
}

test("vectorWritePipeline dedups groups and fans out a single embedding", async (assert) => {
	let embedCalls = 0;
	let embeddedDocuments: string[] = [];
	const writes: StoredVectorRecord[] = [];
	const pipeline = createVectorWritePipeline({
		embeddings: {
			embedDocuments: async (documents: string[]) => {
				embedCalls += 1;
				embeddedDocuments = documents;
				return documents.map((value) => [value.length]);
			},
			embedQuery: async () => [0],
		},
		resolveEmbeddingSpace: async () => "space-1",
		getCachedRecords: async () => [],
		putRecords: async (records) => {
			writes.push(...records);
		},
		dedupStrategy: "contentAndText",
	});

	const docs = [
		createDocument("repeat"),
		createDocument("repeat"),
		createDocument("unique"),
	];
	const result = await pipeline.addDocuments(docs);

	assert.equal(embedCalls, 1, "embedDocuments invoked once");
	assert.deepEqual(
		embeddedDocuments,
		["repeat", "unique"],
		"only representative documents are embedded",
	);
	assert.equal(writes.length, 3, "all input documents are written");
	assert.equal(result.insertedCount, 3, "summary reports inserted count");
	assert.equal(result.dedupGroupCount, 2, "summary reports dedup group count");
	assert.end();
});

test("resolveRecordId uses deterministic fallback when randomUUID is unavailable", async (assert) => {
	const document = createDocument("fallback-content", { foo: "bar" });
	const originalRandomUUID = (globalThis as any).crypto?.randomUUID;

	try {
		if (globalThis.crypto) {
			(globalThis.crypto as any).randomUUID = undefined;
		}

		const firstId = resolveRecordId(document, 0);
		const secondId = resolveRecordId(document, 1);

		assert.equal(
			firstId.startsWith("doc-fallback-"),
			true,
			"fallback id is generated when randomUUID is unavailable",
		);
		assert.notEqual(
			firstId,
			secondId,
			"different fallbackIndex yields distinct deterministic ids",
		);
	} finally {
		if (globalThis.crypto && typeof originalRandomUUID !== "undefined") {
			(globalThis.crypto as any).randomUUID = originalRandomUUID;
		}
	}

	assert.end();
});

test("vectorWritePipeline strict mode rejects repeated content vector mismatch", async (assert) => {
	const pipeline = createVectorWritePipeline({
		embeddings: {
			embedDocuments: async () => [],
			embedQuery: async () => [0],
		},
		resolveEmbeddingSpace: async () => "space-1",
		getCachedRecords: async () => [],
		putRecords: async () => undefined,
		strictDeterminism: true,
		dedupStrategy: "contentAndText",
	});

	try {
		await pipeline.addVectors(
			[
				[1, 2],
				[9, 9],
			],
			[createDocument("repeat"), createDocument("repeat")],
		);
		assert.fail("expected strict determinism failure");
	} catch (error) {
		assert.ok(
			error instanceof Error &&
				/deterministic embedding guard/i.test(error.message),
			"throws with deterministic guard message",
		);
	}

	assert.end();
});

test("vectorWritePipeline addVectors reports unique dedup groups", async (assert) => {
	const writes: StoredVectorRecord[] = [];
	const pipeline = createVectorWritePipeline({
		embeddings: {
			embedDocuments: async () => [],
			embedQuery: async () => [0],
		},
		resolveEmbeddingSpace: async () => "space-1",
		getCachedRecords: async () => [],
		putRecords: async (records) => {
			writes.push(...records);
		},
		dedupStrategy: "contentAndText",
	});

	const result = await pipeline.addVectors(
		[[1], [2], [3]],
		[createDocument("repeat"), createDocument("repeat"), createDocument("unique")],
	);

	assert.equal(result.insertedCount, 3, "insertedCount still reflects all written records");
	assert.equal(result.dedupGroupCount, 2, "dedupGroupCount reports unique groups");
	assert.equal(writes.length, 3, "all records are still written for addVectors");
	assert.end();
});

test("vectorWritePipeline reuses cached embeddings by content hash and space", async (assert) => {
	let embedCalls = 0;
	const writes: StoredVectorRecord[] = [];
	const pipeline = createVectorWritePipeline({
		embeddings: {
			embedDocuments: async (documents: string[]) => {
				embedCalls += 1;
				return documents.map((value) => [value.length]);
			},
			embedQuery: async () => [0],
		},
		resolveEmbeddingSpace: async () => "space-1",
		getCachedRecords: async (_space, _hashes, contents) =>
			contents.map((content, index) =>
				index === 0
					? {
							id: "cached",
							content,
							embeddingSpace: "space-1",
							contentHash: "h1",
							cacheKey: "space-1:h1",
							embedding: [42],
							metadata: {},
						}
					: null,
			),
		putRecords: async (records) => {
			writes.push(...records);
		},
	});

	const result = await pipeline.addDocuments([
		createDocument("cached-content"),
		createDocument("new-content"),
	]);

	assert.equal(embedCalls, 1, "only non-cached documents are embedded");
	assert.deepEqual(writes[0]?.embedding, [42], "cached embedding is reused");
	assert.equal(result.reusedEmbeddingCount, 1, "summary reports reuse count");
	assert.end();
});

test("vectorWritePipeline record mapper is equivalent for addDocuments and addVectors", async (assert) => {
	const docs = [
		createDocument("alpha", { rank: 1 }),
		createDocument("beta", { rank: 2 }),
	];
	const writesFromDocuments: StoredVectorRecord[] = [];
	const writesFromVectors: StoredVectorRecord[] = [];

	const pipelineForDocuments = createVectorWritePipeline({
		embeddings: {
			embedDocuments: async (documents: string[]) =>
				documents.map((d) => [d.length]),
			embedQuery: async () => [0],
		},
		resolveEmbeddingSpace: async () => "space-1",
		getCachedRecords: async () => [null, null],
		putRecords: async (records) => {
			writesFromDocuments.push(...records);
		},
	});
	const pipelineForVectors = createVectorWritePipeline({
		embeddings: {
			embedDocuments: async () => [],
			embedQuery: async () => [0],
		},
		resolveEmbeddingSpace: async () => "space-1",
		getCachedRecords: async () => [],
		putRecords: async (records) => {
			writesFromVectors.push(...records);
		},
	});

	await pipelineForDocuments.addDocuments(docs);
	await pipelineForVectors.addVectors([[5], [4]], docs);

	assert.equal(writesFromDocuments.length, 2, "addDocuments writes 2 records");
	assert.equal(writesFromVectors.length, 2, "addVectors writes 2 records");
	assert.deepEqual(
		writesFromDocuments.map((record) => ({
			content: record.content,
			embeddingSpace: record.embeddingSpace,
			cacheKey: record.cacheKey,
			metadata: record.metadata,
		})),
		writesFromVectors.map((record) => ({
			content: record.content,
			embeddingSpace: record.embeddingSpace,
			cacheKey: record.cacheKey,
			metadata: record.metadata,
		})),
		"both APIs produce compatible record shapes",
	);
	assert.end();
});
