import {
	IDBIndex as FakeIDBIndex,
	IDBObjectStore as FakeIDBObjectStore,
	indexedDB as fakeIndexedDB,
} from "fake-indexeddb";

export function uniqueDbName(prefix = "test"): string {
	return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function installFakeIndexedDb(): void {
	try {
		const globalObject = globalThis as { indexedDB?: IDBFactory };
		globalObject.indexedDB = fakeIndexedDB as unknown as IDBFactory;
	} catch (error) {
		console.warn(
			"installFakeIndexedDb: cannot overwrite indexedDB, skipping",
			error,
		);
	}
}

export function installFailingOpenOnceIndexedDb(): void {
	const baseFactory = fakeIndexedDB as unknown as IDBFactory;
	let failed = false;

	(globalThis as { indexedDB?: IDBFactory }).indexedDB = {
		...baseFactory,
		cmp: baseFactory.cmp.bind(baseFactory),
		deleteDatabase: baseFactory.deleteDatabase.bind(baseFactory),
		open: (name: string, version?: number) => {
			if (failed) {
				return baseFactory.open(name, version);
			}

			failed = true;
			const request = {
				error: new Error("simulated open failure"),
				onerror: null,
				onsuccess: null,
				onblocked: null,
				onupgradeneeded: null,
				readyState: "pending",
				result: undefined,
				source: null,
				transaction: null,
			} as unknown as IDBOpenDBRequest;

			queueMicrotask(() => {
				(
					request as IDBOpenDBRequest & {
						readyState: IDBRequestReadyState;
						onerror: ((event: Event) => void) | null;
					}
				).readyState = "done";
				(
					request as IDBOpenDBRequest & {
						onerror: ((event: Event) => void) | null;
					}
				).onerror?.(new Event("error"));
			});

			return request;
		},
	} as unknown as IDBFactory;
}

export function patchMissingGetAll(): { restore: () => void } {
	const originalObjectStoreGetAll = FakeIDBObjectStore.prototype.getAll;
	const originalIndexGetAll = FakeIDBIndex.prototype.getAll;

	// @ts-expect-error
	FakeIDBObjectStore.prototype.getAll = undefined;
	// @ts-expect-error
	FakeIDBIndex.prototype.getAll = undefined;

	return {
		restore: () => {
			FakeIDBObjectStore.prototype.getAll = originalObjectStoreGetAll;
			FakeIDBIndex.prototype.getAll = originalIndexGetAll;
		},
	};
}
