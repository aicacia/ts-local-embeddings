import { Document } from "@langchain/core/documents";
import test from "tape";
import { indexedDB as fakeIndexedDB } from "fake-indexeddb";
import { IndexedDBVectorStore } from "./IndexedDBVectorStore.js";
import { IndexedDbStoreGateway } from "./indexedDbStoreGateway.js";

function uniqueDbName(): string {
	return `vector-store-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function installFakeIndexedDb(): void {
	(globalThis as { indexedDB?: IDBFactory }).indexedDB =
		fakeIndexedDB as unknown as IDBFactory;
}

function countChars(value: string): [number, number] {
	const a = (value.match(/a/gi) ?? []).length;
	const b = (value.match(/b/gi) ?? []).length;
	return [a, b];
}

const embeddings = {
	embedDocuments: async (documents: string[]) =>
		documents.map((doc) => countChars(doc)),
	embedQuery: async (document: string) => countChars(document),
};

test("IndexedDBVectorStore addDocuments and similarity search work through gateway", async (assert) => {
	installFakeIndexedDb();
	const store = new IndexedDBVectorStore(embeddings, {
		dbName: uniqueDbName(),
	});

	await store.addDocuments([
		new Document({ pageContent: "aaaa" }),
		new Document({ pageContent: "bbbb" }),
	]);

	const results = await store.similaritySearch("aaa", 1);
	assert.equal(results.length, 1, "returns one result");
	assert.equal(results[0]?.pageContent, "aaaa", "returns nearest document");
	assert.equal(await store.count(), 2, "count delegates correctly");

	await store.close();
	assert.end();
});

test("IndexedDBVectorStore maxMarginalRelevanceSearch returns Document instances", async (assert) => {
	installFakeIndexedDb();
	const store = new IndexedDBVectorStore(embeddings, {
		dbName: uniqueDbName(),
	});

	await store.addDocuments([
		new Document({ pageContent: "ab" }),
		new Document({ pageContent: "aaa" }),
	]);

	const results: Document[] = await store.maxMarginalRelevanceSearch("ab", {
		k: 1,
		fetchK: 2,
	});

	assert.equal(results.length, 1, "returns one result");
	assert.equal(results[0]?.pageContent, "ab", "returns nearest document");

	await store.close();
	assert.end();
});

test(
	"IndexedDBVectorStore similaritySearchWithScore does not use bulk getAll",
	async (assert) => {
		installFakeIndexedDb();
		const store = new IndexedDBVectorStore(embeddings, {
			dbName: uniqueDbName(),
		});

		await store.addDocuments([
			new Document({ pageContent: "aaaa" }),
			new Document({ pageContent: "bbbb" }),
		]);

		const originalGetAll = IndexedDbStoreGateway.prototype.getAll;
		IndexedDbStoreGateway.prototype.getAll = async () => {
			throw new Error("getAll should not be called for queryVectors");
		};

		try {
			const results = await store.similaritySearchWithScore("aaa", 1);
			assert.equal(results.length, 1, "returns one result");
			assert.equal(results[0]?.[0].pageContent, "aaaa", "returns nearest document");
		} finally {
			IndexedDbStoreGateway.prototype.getAll = originalGetAll;
		}

		await store.close();
		assert.end();
	},
);

test("IndexedDBVectorStore close and reopen via fromExistingIndex preserves data", async (assert) => {
	installFakeIndexedDb();
	const dbName = uniqueDbName();
	const store = new IndexedDBVectorStore(embeddings, { dbName });
	await store.addDocuments([new Document({ pageContent: "ab" })]);
	await store.close();

	const reopened = await IndexedDBVectorStore.fromExistingIndex(embeddings, {
		dbName,
	});
	assert.equal(await reopened.count(), 1, "reopened index still has data");

	await reopened.close();
	assert.end();
});
