import test from "tape";
import {
	IndexedDbStoreGateway,
	VECTOR_STORE_SCHEMA,
} from "./indexedDbStoreGateway.js";
import type { StoredVectorRecord } from "./vectorWritePipeline.js";
import { createVectorWritePipeline } from "./vectorWritePipeline.js";
import { Document } from "@langchain/core/documents";
import {
	installFakeIndexedDb,
	installFailingOpenOnceIndexedDb,
	patchMissingGetAll,
	uniqueDbName,
} from "./testUtils.js";

test("IndexedDbStoreGateway open creates expected schema", async (assert) => {
	installFakeIndexedDb();
	const dbName = uniqueDbName();
	const gateway = new IndexedDbStoreGateway({ dbName, storeName: "vectors" });

	const db = await gateway.open();
	assert.ok(db.objectStoreNames.contains("vectors"), "creates vectors store");

	const tx = db.transaction("vectors", "readonly");
	const store = tx.objectStore("vectors");
	assert.ok(
		store.indexNames.contains(VECTOR_STORE_SCHEMA.contentHashIndex),
		"creates content hash index",
	);

	await gateway.close();
	assert.end();
});

test("IndexedDbStoreGateway put/get/query/count/clear lifecycle", async (assert) => {
	installFakeIndexedDb();
	const dbName = uniqueDbName();
	const gateway = new IndexedDbStoreGateway({ dbName, storeName: "vectors" });
	const records: StoredVectorRecord[] = [
		{
			id: "1",
			content: "hello world",
			embeddingSpace: "space-1",
			contentHash: "h1",
			cacheKey: "space-1:h1",
			embedding: [1, 0],
			metadata: {},
		},
		{
			id: "2",
			content: "goodbye",
			embeddingSpace: "space-2",
			contentHash: "h2",
			cacheKey: "space-2:h2",
			embedding: [0, 1],
			metadata: {},
		},
	];

	await gateway.put(records);
	assert.equal(await gateway.count(), 2, "count reflects inserted records");

	const all = await gateway.getAll();
	assert.equal(all.length, 2, "getAll returns all records");

	const queried = await gateway.queryByContentHash(
		["h1", "h2"],
		["hello world", "goodbye"],
		(record, index) =>
			(index < 0 || record.content === ["hello world", "goodbye"][index]) &&
			record.embeddingSpace === ["space-1", "space-2"][Math.max(0, index)],
	);
	assert.equal(queried[0]?.id, "1", "query returns first matching record");
	assert.equal(queried[1]?.id, "2", "query returns second matching record");

	await gateway.clear();
	assert.equal(await gateway.count(), 0, "clear removes all records");

	await gateway.close();
	assert.end();
});

test("IndexedDbStoreGateway works with getAll unavailable", async (assert) => {
	installFakeIndexedDb();
	const { restore } = patchMissingGetAll();
	try {
		const dbName = uniqueDbName();
		const gateway = new IndexedDbStoreGateway({ dbName, storeName: "vectors" });
		const records: StoredVectorRecord[] = [
			{
				id: "1",
				content: "hello world",
				embeddingSpace: "space-1",
				contentHash: "h1",
				cacheKey: "space-1:h1",
				embedding: [1, 0],
				metadata: {},
			},
			{
				id: "2",
				content: "hello world",
				embeddingSpace: "space-2",
				contentHash: "h1",
				cacheKey: "space-2:h1",
				embedding: [0, 1],
				metadata: {},
			},
		];

		await gateway.put(records);
		assert.equal(await gateway.count(), 2, "count still works without getAll");

		const all = await gateway.getAll();
		assert.equal(all.length, 2, "fallback getAll returns all records");

		const queried = await gateway.queryByContentHash(
			["h1"],
			["hello world"],
			(record, index) => record.embeddingSpace === "space-1" && index === 0,
		);
		assert.equal(
			queried[0]?.id,
			"1",
			"fallback queryByContentHash returns matching record",
		);

		await gateway.close();
	} finally {
		restore();
	}

	assert.end();
});
test("IndexedDbStoreGateway close invalidates and reopen keeps persisted data", async (assert) => {
	installFakeIndexedDb();
	const dbName = uniqueDbName();
	const gateway = new IndexedDbStoreGateway({ dbName, storeName: "vectors" });

	await gateway.put([
		{
			id: "1",
			content: "persisted",
			embeddingSpace: "space",
			contentHash: "hp",
			cacheKey: "space:hp",
			embedding: [1],
			metadata: {},
		},
	]);
	await gateway.close();

	const reopened = await gateway.open();
	assert.ok(reopened.objectStoreNames.contains("vectors"), "reopens database");
	assert.equal(await gateway.count(), 1, "data remains after close/reopen");

	await gateway.close();
	assert.end();
});

test("IndexedDbStoreGateway retries open after an initial failure", async (assert) => {
	installFailingOpenOnceIndexedDb();
	const dbName = uniqueDbName();
	const gateway = new IndexedDbStoreGateway({ dbName, storeName: "vectors" });

	try {
		await gateway.open();
		assert.fail("expected first open to fail");
	} catch (error) {
		assert.ok(
			error instanceof Error && /simulated open failure/i.test(error.message),
			"first open surfaces the initial failure",
		);
	}

	const reopened = await gateway.open();
	assert.ok(
		reopened.objectStoreNames.contains("vectors"),
		"second open retries instead of reusing stale rejected state",
	);

	await gateway.close();
	installFakeIndexedDb();
	assert.end();
});

test("IndexedDbStoreGateway persists Float32Array embeddings", async (assert) => {
	installFakeIndexedDb();
	const dbName = uniqueDbName();
	const gateway = new IndexedDbStoreGateway({ dbName, storeName: "vectors" });

	const emb = new Float32Array([1.1, 2.2, 3.3]);
	await gateway.put([
		{
			id: "f1",
			content: "doc1",
			embeddingSpace: "space",
			contentHash: "hf1",
			cacheKey: "space:hf1",
			embedding: emb,
			metadata: {},
		},
	]);

	const all = await gateway.getAll();
	assert.equal(all.length, 1, "one record persisted");
	assert.ok(
		all[0].embedding instanceof Float32Array,
		"embedding is Float32Array",
	);
	assert.deepEqual(
		Array.from(all[0].embedding as Float32Array),
		Array.from(emb),
		"values preserved",
	);

	await gateway.close();
	assert.end();
});

test("VectorWritePipeline converts large number[] to Float32Array", async (assert) => {
	// Use an embedding runtime that returns plain number[] vectors larger than threshold
	const docs = [
		new Document({ pageContent: "d1", metadata: {} }),
		new Document({ pageContent: "d2", metadata: {} }),
	];

	const embeddingsRuntime = {
		embedDocuments: async (inputs: string[]) =>
			inputs.map(() => Array.from({ length: 20 }, (_, i) => i + 0.5)),
		embedQuery: async (doc: string) => [1],
	} as any;

	const captured: any[] = [];
	const pipeline = createVectorWritePipeline({
		embeddings: embeddingsRuntime,
		resolveEmbeddingSpace: async () => "space",
		getCachedRecords: async () => docs.map(() => null),
		putRecords: async (records) => {
			captured.push(...records);
		},
	});

	await pipeline.addDocuments(docs);
	assert.equal(captured.length, docs.length, "putRecords called with records");
	assert.ok(
		captured[0].embedding instanceof Float32Array,
		"embedding converted to Float32Array",
	);

	assert.end();
});
