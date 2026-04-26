/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Suite, Deferred } from "benchmark";
import { Document } from "@langchain/core/documents";
import { IndexedDBVectorStore } from "./IndexedDBVectorStore.js";
import type { Constructor } from "../types.js";
import { createDocuments } from "../utils/documentUtils.js";
import { installFakeIndexedDb, uniqueDbName } from "./testUtils.js";

const embeddings = {
	embedDocuments: async (documents: string[]) =>
		documents.map((content) => [content.length]),
	embedQuery: async (document: string) => [document.length],
};

export default async function register(Suite: Constructor<Suite>) {
	installFakeIndexedDb();

	const uniqueDocs = createDocuments(100, "unique-document");
	const duplicateDocs = createDocuments(100, "duplicate-document");
	const identicalDocs = createDocuments(100, "duplicate-document");

	const populatedStore = new IndexedDBVectorStore(embeddings, {
		dbName: uniqueDbName(),
	});

	await populatedStore.addDocuments(uniqueDocs);

	return new Promise<void>((resolve, reject) => {
		new Suite()
			.add("store/addDocuments/100-unique", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const store = new IndexedDBVectorStore(embeddings, {
						dbName: uniqueDbName(),
					});
					await store.addDocuments(uniqueDocs);
					await store.close();
					deferred.resolve();
				},
			})
			.add("store/addDocuments/100-duplicate", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const store = new IndexedDBVectorStore(embeddings, {
						dbName: uniqueDbName(),
					});
					await store.addDocuments(identicalDocs);
					await store.close();
					deferred.resolve();
				},
			})
			.add("store/similaritySearch/100", {
				defer: true,
				fn: async (deferred: Deferred) => {
					await populatedStore.similaritySearch("duplicate-document", 5);
					deferred.resolve();
				},
			})
			.add("store/maxMarginalRelevanceSearch/100", {
				defer: true,
				fn: async (deferred: Deferred) => {
					await populatedStore.maxMarginalRelevanceSearch(
						"duplicate-document",
						{
							k: 5,
							fetchK: 20,
						},
					);
					deferred.resolve();
				},
			})
			.on("cycle", (event: any) => {
				console.log(String(event.target));
			})
			.on("complete", async () => {
				await populatedStore.close();
				resolve();
			})
			.on("error", (event: any) => {
				reject((event.target as Error) ?? new Error("Benchmark suite failed"));
			})
			.run({ async: true });
	});
}
