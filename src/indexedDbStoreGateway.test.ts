import test from "tape";
import { indexedDB as fakeIndexedDB } from "fake-indexeddb";
import {
	IndexedDbStoreGateway,
	VECTOR_STORE_SCHEMA,
} from "./indexedDbStoreGateway.js";
import type { StoredVectorRecord } from "./vectorWritePipeline.js";

function uniqueDbName(): string {
	return `test-db-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function installFakeIndexedDb(): void {
	(globalThis as { indexedDB?: IDBFactory }).indexedDB =
		fakeIndexedDB as unknown as IDBFactory;
}

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
