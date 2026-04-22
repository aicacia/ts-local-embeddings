/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Suite, Deferred } from "benchmark";
import { indexedDB as fakeIndexedDB } from "fake-indexeddb";
import { IndexedDbStoreGateway } from "./indexedDbStoreGateway.js";
import type { StoredVectorRecord } from "./vectorWritePipeline.js";
import type { Constructor } from "../types.js";

function installFakeIndexedDb(): void {
	try {
		const g = globalThis as { indexedDB?: IDBFactory };
		if (typeof g.indexedDB === "undefined") {
			g.indexedDB = fakeIndexedDB as unknown as IDBFactory;
		}
	} catch (err) {
		// In some browsers `indexedDB` is non-writable; skip if we can't set it
		console.warn("installFakeIndexedDb: cannot overwrite indexedDB, skipping");
	}
}

function uniqueDbName(): string {
	return `gateway-bench-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createRecord(index: number): StoredVectorRecord {
	const content = `document-${index}`;
	const contentHash = index.toString(16).padStart(8, "0");

	return {
		id: `id-${index}`,
		content,
		embeddingSpace: "default",
		contentHash,
		cacheKey: `default:${contentHash}`,
		embedding: Array.from({ length: 16 }, () => Math.random()),
		metadata: { index },
	};
}

export default async function register(Suite: Constructor<Suite>) {
	installFakeIndexedDb();

	const gateway = new IndexedDbStoreGateway({
		dbName: uniqueDbName(),
		storeName: "vectors",
	});

	const records = Array.from({ length: 128 }, (_, index) =>
		createRecord(index),
	);

	await gateway.open();
	await gateway.put(records);

	return new Promise<void>((resolve, reject) => {
		new Suite()
			.add("gateway/open", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const oneOffGateway = new IndexedDbStoreGateway({
						dbName: uniqueDbName(),
						storeName: "vectors",
					});
					await oneOffGateway.open();
					await oneOffGateway.close();
					deferred.resolve();
				},
			})
			.add("gateway/put-128-records", {
				defer: true,
				fn: async (deferred: Deferred) => {
					await gateway.put(records);
					deferred.resolve();
				},
			})
			.add("gateway/getAll-128-records", {
				defer: true,
				fn: async (deferred: Deferred) => {
					await gateway.getAll();
					deferred.resolve();
				},
			})
			.add("gateway/queryByContentHash-16", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const hashes = records
						.slice(0, 16)
						.map((record) => record.contentHash ?? "");
					const contents = records.slice(0, 16).map((record) => record.content);
					await gateway.queryByContentHash(hashes, contents, () => true);
					deferred.resolve();
				},
			})
			.add("gateway/clear", {
				defer: true,
				fn: async (deferred: Deferred) => {
					await gateway.clear();
					deferred.resolve();
				},
			})
			.on("cycle", (event: any) => {
				console.log(String(event.target));
			})
			.on("complete", async () => {
				await gateway.close();
				resolve();
			})
			.on("error", (event: any) => {
				reject((event.target as Error) ?? new Error("Benchmark suite failed"));
			})
			.run({ async: true });
	});
}
