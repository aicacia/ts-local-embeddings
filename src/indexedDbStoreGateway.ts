import type { StoredVectorRecord } from "./vectorWritePipeline.js";

export const VECTOR_STORE_SCHEMA = {
	defaultDbName: "langchain-indexeddb-vectorstore",
	defaultStoreName: "vectors",
	contentHashIndex: "by_content_hash",
	currentVersion: 2,
} as const;

type Migration = {
	fromVersion: number;
	toVersion: number;
	apply: (args: {
		request: IDBOpenDBRequest;
		storeName: string;
		contentHashIndex: string;
	}) => void;
};

const MIGRATIONS: Migration[] = [
	{
		fromVersion: 0,
		toVersion: 1,
		apply: ({ request, storeName }) => {
			const database = request.result;
			if (!database.objectStoreNames.contains(storeName)) {
				database.createObjectStore(storeName, {
					keyPath: "id",
				});
			}
		},
	},
	{
		fromVersion: 1,
		toVersion: 2,
		apply: ({ request, storeName, contentHashIndex }) => {
			const database = request.result;
			const store = request.transaction?.objectStore(storeName);
			if (database.objectStoreNames.contains(storeName)) {
				if (store && !store.indexNames.contains(contentHashIndex)) {
					store.createIndex(contentHashIndex, "contentHash", {
						unique: false,
					});
				}
				return;
			}

			const createdStore = database.createObjectStore(storeName, {
				keyPath: "id",
			});
			createdStore.createIndex(contentHashIndex, "contentHash", {
				unique: false,
			});
		},
	},
];

export function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		request.onsuccess = () => resolve(request.result);
		request.onerror = () =>
			reject(request.error ?? new Error("IndexedDB request failed."));
	});
}

export function transactionDone(transaction: IDBTransaction): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		transaction.oncomplete = () => resolve();
		transaction.onerror = () =>
			reject(transaction.error ?? new Error("IndexedDB transaction failed."));
		transaction.onabort = () =>
			reject(transaction.error ?? new Error("IndexedDB transaction aborted."));
	});
}

function hasIndex(store: IDBObjectStore, indexName: string): boolean {
	const indexNames = store.indexNames;
	if (
		typeof (indexNames as { contains?: unknown }).contains === "function"
	) {
		return (
			indexNames as {
				contains(name: string): boolean;
			}
		).contains(indexName);
	}

	for (let index = 0; index < indexNames.length; index += 1) {
		if (indexNames[index] === indexName) {
			return true;
		}
	}

	return false;
}

function getAllRecords<T>(
	source: IDBObjectStore | IDBIndex,
	query?: IDBValidKey | IDBKeyRange,
): Promise<T[]> {
	const getAll = (source as { getAll?: unknown }).getAll;
	if (typeof getAll === "function") {
		return requestToPromise(
			(getAll as {
				(query?: IDBValidKey | IDBKeyRange): IDBRequest<T[]>;
			}).call(source, query),
		);
	}

	return new Promise<T[]>((resolve, reject) => {
		const records: T[] = [];
		const request = source.openCursor(query);
		request.onsuccess = () => {
			const cursor = request.result;
			if (!cursor) {
				resolve(records);
				return;
			}

			records.push(cursor.value);
			cursor.continue();
		};
		request.onerror = () =>
			reject(
				request.error ?? new Error("IndexedDB cursor iteration failed."),
			);
	});
}

function iterateRecords<T>(
	source: IDBObjectStore | IDBIndex,
	callback: (record: T) => Promise<boolean | void> | boolean | void,
	query?: IDBValidKey | IDBKeyRange,
): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		const request = source.openCursor(query);
		request.onsuccess = async () => {
			const cursor = request.result;
			if (!cursor) {
				resolve();
				return;
			}

			try {
				const shouldContinue = await callback(cursor.value);
				if (shouldContinue === false) {
					resolve();
					return;
				}
			} catch (error) {
				reject(error);
				return;
			}

			cursor.continue();
		};
		request.onerror = () =>
			reject(
				request.error ?? new Error("IndexedDB cursor iteration failed."),
			);
	});
}

function applyMigrations(
	request: IDBOpenDBRequest,
	oldVersion: number,
	newVersion: number,
	storeName: string,
): void {
	const start = Number.isFinite(oldVersion) ? oldVersion : 0;
	const end = Number.isFinite(newVersion)
		? newVersion
		: VECTOR_STORE_SCHEMA.currentVersion;

	for (const migration of MIGRATIONS) {
		if (migration.fromVersion < start || migration.toVersion > end) {
			continue;
		}

		migration.apply({
			request,
			storeName,
			contentHashIndex: VECTOR_STORE_SCHEMA.contentHashIndex,
		});
	}
}

export type IndexedDbStoreGatewayArgs = {
	dbName?: string;
	storeName?: string;
};

export class IndexedDbStoreGateway {
	readonly #dbName: string;
	readonly #storeName: string;
	#dbPromise: Promise<IDBDatabase> | null = null;

	constructor(args: IndexedDbStoreGatewayArgs = {}) {
		this.#dbName = args.dbName ?? VECTOR_STORE_SCHEMA.defaultDbName;
		this.#storeName = args.storeName ?? VECTOR_STORE_SCHEMA.defaultStoreName;
	}

	open(): Promise<IDBDatabase> {
		if (typeof indexedDB === "undefined") {
			throw new Error("IndexedDB is not available in this environment.");
		}

		if (this.#dbPromise) {
			return this.#dbPromise;
		}

		const openPromise = new Promise<IDBDatabase>((resolve, reject) => {
			const rejectOpen = (error: Error): void => {
				if (this.#dbPromise === openPromise) {
					this.#dbPromise = null;
				}
				reject(error);
			};

			const request = indexedDB.open(
				this.#dbName,
				VECTOR_STORE_SCHEMA.currentVersion,
			);

			request.onupgradeneeded = (event) => {
				const previousVersion = event.oldVersion;
				const nextVersion =
					event.newVersion ?? VECTOR_STORE_SCHEMA.currentVersion;
				applyMigrations(request, previousVersion, nextVersion, this.#storeName);
			};

			request.onsuccess = () => {
				const database = request.result;
				const hasStore = database.objectStoreNames.contains(this.#storeName);
				if (!hasStore) {
					rejectOpen(
						new Error(
							`IndexedDB schema mismatch: missing object store '${this.#storeName}'.`,
						),
					);
					return;
				}
				resolve(database);
			};

			request.onerror = () => {
				rejectOpen(
					request.error ?? new Error("Failed to open IndexedDB database."),
				);
			};
		});

		this.#dbPromise = openPromise;

		return this.#dbPromise;
	}

	async close(): Promise<void> {
		if (!this.#dbPromise) {
			return;
		}

		const database = await this.#dbPromise;
		database.close();
		this.#dbPromise = null;
	}

	async getAll(): Promise<StoredVectorRecord[]> {
		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readonly");
		const records = await getAllRecords<StoredVectorRecord>(
			transaction.objectStore(this.#storeName),
		);
		await transactionDone(transaction);
		return records;
	}

	async iterateAll<T>(
		callback: (record: T) => Promise<boolean | void> | boolean | void,
	): Promise<void> {
		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readonly");
		await iterateRecords<T>(
			transaction.objectStore(this.#storeName),
			callback,
		);
		await transactionDone(transaction);
	}

	async queryByContentHash(
		contentHashes: string[],
		contents: string[],
		predicate: (record: StoredVectorRecord, index: number) => boolean,
	): Promise<Array<StoredVectorRecord | null>> {
		if (contentHashes.length === 0) {
			return [];
		}

		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readonly");
		const store = transaction.objectStore(this.#storeName);

		if (hasIndex(store, VECTOR_STORE_SCHEMA.contentHashIndex)) {
			const directMatches = await Promise.all(
				contentHashes.map(async (contentHash, index) => {
					const matches = await getAllRecords<StoredVectorRecord>(
						store.index(VECTOR_STORE_SCHEMA.contentHashIndex),
						contentHash,
					);

					return (
						matches.find(
							(match) =>
								match.content === contents[index] && predicate(match, index),
						) ?? null
					);
				}),
			);
			await transactionDone(transaction);
			return directMatches;
		}

		const allRecords = await getAllRecords<StoredVectorRecord>(store);
		// Fallback path: if the contentHash index is missing, this performs a full
		// scan of the store so old database versions remain compatible.
		const recordByHash = new Map<string, StoredVectorRecord>();
		const recordByContent = new Map<string, StoredVectorRecord>();

		for (const record of allRecords) {
			if (!predicate(record, -1)) {
				continue;
			}

			if (record.contentHash) {
				recordByHash.set(record.contentHash, record);
			}
			recordByContent.set(record.content, record);
		}

		const records = contentHashes.map((contentHash, index) => {
			const byHash = recordByHash.get(contentHash);
			if (byHash && byHash.content === contents[index]) {
				return byHash;
			}

			return recordByContent.get(contents[index]) ?? null;
		});

		await transactionDone(transaction);
		return records;
	}

	async count(): Promise<number> {
		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readonly");
		const count = await requestToPromise(
			transaction.objectStore(this.#storeName).count(),
		);
		await transactionDone(transaction);
		return count;
	}

	async put(records: StoredVectorRecord[]): Promise<void> {
		if (records.length === 0) {
			return;
		}

		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readwrite");
		const store = transaction.objectStore(this.#storeName);

		for (const record of records) {
			store.put(record);
		}

		await transactionDone(transaction);
	}

	async clear(): Promise<void> {
		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readwrite");
		transaction.objectStore(this.#storeName).clear();
		await transactionDone(transaction);
	}
}
