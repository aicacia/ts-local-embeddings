import type { StoredVectorRecord } from "./vectorWritePipeline.js";
import { serializeEmbeddingForTransfer } from "../serialization/transfer.js";

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
	if (typeof (indexNames as { contains?: unknown }).contains === "function") {
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
		return requestToPromise(getAll.call(source, query));
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
			reject(request.error ?? new Error("IndexedDB cursor iteration failed."));
	});
}
function iterateRecords<T>(
	source: IDBObjectStore | IDBIndex,
	callback: (record: T) => Promise<boolean | undefined> | boolean | undefined,
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
			reject(request.error ?? new Error("IndexedDB cursor iteration failed."));
	});
}

function normalizeStoredVectorRecord(record: any): StoredVectorRecord {
	if (!record || record.embedding == null) return record;
	const emb = record.embedding;
	try {
		if (emb instanceof ArrayBuffer) {
			record.embedding = new Float32Array(emb);
		} else if (ArrayBuffer.isView(emb)) {
			if (!(emb instanceof Float32Array)) {
				const length =
					(emb as any).length ?? Math.floor((emb as any).byteLength / 4);
				record.embedding = new Float32Array(
					(emb as any).buffer,
					(emb as any).byteOffset ?? 0,
					length,
				);
			}
		}
	} catch (_e) {
		// If normalization fails, return record as-is.
	}
	return record;
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

export type StorageGatewayPort = {
	open(): Promise<IDBDatabase>;
	close(): Promise<void>;
	getAll(): Promise<StoredVectorRecord[]>;
	iterateAll<T>(
		callback: (record: T) => Promise<boolean | undefined> | boolean | undefined,
	): Promise<void>;
	queryByContentHash(
		contentHashes: string[],
		contents: string[],
		predicate: (record: StoredVectorRecord, index: number) => boolean,
	): Promise<Array<StoredVectorRecord | null>>;
	count(): Promise<number>;
	put(records: StoredVectorRecord[]): Promise<void>;
	clear(): Promise<void>;
};

// Helper: compute a safe chunk size for batched IndexedDB `put()` writes at
// runtime. We sample a small subset of records, estimate the average
// serialized size via `JSON.stringify`, and target ~256KB per transaction to
// balance throughput and transaction latency. Values are clamped to a
// conservative range to avoid extremely small or huge transactions.
/* eslint-disable @typescript-eslint/no-explicit-any */
function computePutChunkSize(records?: StoredVectorRecord[]): number {
	const DEFAULT = 64;
	const MIN = 4;
	// Allow larger chunk sizes for high-throughput workloads so fewer
	// transactions are created when storing larger embeddings.
	const MAX = 4096;
	const SAMPLE_LIMIT = 16;
	// Increase target transaction bytes to reduce transaction overhead by
	// grouping more records per transaction (512KB target).
	const TARGET_BYTES = 512 * 1024; // 512KB

	if (!records || records.length === 0) {
		return DEFAULT;
	}

	const sampleCount = Math.min(SAMPLE_LIMIT, records.length);
	const step = Math.max(1, Math.floor(records.length / sampleCount));
	const samples: StoredVectorRecord[] = [];
	for (
		let i = 0, picked = 0;
		picked < sampleCount && i < records.length;
		i += step, picked += 1
	) {
		samples.push(records[i]);
	}

	let totalBytes = 0;
	for (const rec of samples) {
		try {
			let bytes = 0;
			// Estimate embedding size: prefer TypedArray.byteLength when present,
			// otherwise approximate numeric arrays as Float32 (4 bytes per entry).
			const emb = rec.embedding;
			if (emb && ArrayBuffer.isView(emb)) {
				// Typed arrays expose `byteLength` via ArrayBufferView
				const view = emb as unknown as ArrayBufferView;
				bytes += view.byteLength;
			} else if (Array.isArray(emb)) {
				bytes += emb.length * 4; // approximate Float32 per element
			}

			// Estimate content size (string characters as bytes approximation).
			if (typeof rec.content === "string") {
				bytes += rec.content.length;
			}

			// Estimate metadata size roughly by number of keys.
			const meta = rec.metadata;
			if (meta && typeof meta === "object") {
				bytes += Math.min(1024, Object.keys(meta).length * 32);
			}

			// Small per-record overhead estimate.
			bytes += 64;
			totalBytes += Math.max(0, Math.floor(bytes));
		} catch (_) {
			totalBytes += 0;
		}
	}

	const avgBytes = Math.max(1, Math.floor(totalBytes / samples.length));
	const raw = Math.floor(TARGET_BYTES / avgBytes);
	const clamped = Math.max(MIN, Math.min(MAX, raw));
	/* eslint-enable @typescript-eslint/no-explicit-any */
	if (!Number.isFinite(clamped) || clamped <= 0) {
		return DEFAULT;
	}
	return clamped;
}

export type IndexedDbStoreGatewayPort = StorageGatewayPort;

export type IndexedDbStoreGatewayArgs = {
	dbName?: string;
	storeName?: string;
	putChunkSize?: number;
	// When true, allow transferring the original ArrayBuffer from the
	// caller to the worker when it is safe (may detach source buffers).
	transferOwnership?: boolean;
	// Threshold (in elements) above which numeric arrays / typed arrays
	// will be converted to `Float32Array` and transferred to the write
	// worker. Smaller vectors remain as `number[]` to avoid copy overhead.
	typedArrayTransferThreshold?: number;
	// Control how embeddings are persisted in IndexedDB. 'auto' leaves
	// the choice to the gateway, 'arraybuffer' forces ArrayBuffer storage,
	// 'typedarray' forces a typed-array view.
	persistEmbeddingAs?: "auto" | "arraybuffer" | "typedarray";
};

export class IndexedDbStoreGateway {
	readonly #dbName: string;
	readonly #storeName: string;
	#dbPromise: Promise<IDBDatabase> | null = null;
	#putChunkSize?: number;
	#cachedPutChunkSize: number | null = null;
	// Pending coalesced put requests queued to be flushed in a single batch.
	#pendingPutRequests: Array<{
		records: StoredVectorRecord[];
		resolve: () => void;
		reject: (err: unknown) => void;
	}> = [];
	#flushScheduled = false;
	// Optional dedicated worker for performing IndexedDB writes off the main
	// thread. Created lazily when first needed and when Worker/IndexedDB are
	// available in the environment.
	#useWriteWorker =
		typeof Worker !== "undefined" &&
		typeof indexedDB !== "undefined" &&
		typeof URL !== "undefined";
	#worker: any | null = null;
	#workerMsgId = 0;
	#pendingWorkerResponses: Map<
		number,
		{
			resolve: () => void;
			reject: (err: unknown) => void;
			pendingRequests: Array<{
				records: StoredVectorRecord[];
				resolve: () => void;
				reject: (err: unknown) => void;
			}>;
		}
	> = new Map();
	// New configurable behavior
	#transferOwnership = true;
	#typedArrayTransferThreshold = 16;
	#persistEmbeddingAs: "auto" | "arraybuffer" | "typedarray" = "auto";

	constructor(args: IndexedDbStoreGatewayArgs = {}) {
		this.#dbName = args.dbName ?? VECTOR_STORE_SCHEMA.defaultDbName;
		this.#storeName = args.storeName ?? VECTOR_STORE_SCHEMA.defaultStoreName;
		this.#putChunkSize = args.putChunkSize;
		this.#transferOwnership = args.transferOwnership ?? true;
		this.#typedArrayTransferThreshold =
			typeof args.typedArrayTransferThreshold === "number"
				? Math.max(1, Math.floor(args.typedArrayTransferThreshold))
				: 16;
		this.#persistEmbeddingAs = args.persistEmbeddingAs ?? "auto";
	}

	// Lazily create an inline worker that performs batched IndexedDB writes.
	// The worker is implemented as an inline blob to avoid requiring extra
	// build-time artifacts; it will open the same DB/schema and apply writes
	// sent from the main thread. If worker creation fails, we silently
	// fall back to main-thread writes.
	async #ensureWorker(): Promise<void> {
		if (!this.#useWriteWorker) return;
		if (this.#worker) return;

		try {
			const workerSource = `
        const CONTENT_HASH_INDEX = ${JSON.stringify(VECTOR_STORE_SCHEMA.contentHashIndex)};
        const DB_VERSION = ${VECTOR_STORE_SCHEMA.currentVersion};

        let dbPromise = null;

        function requestToPromise(request){
          return new Promise((resolve,reject)=>{
            request.onsuccess = ()=> resolve(request.result);
            request.onerror = ()=> reject(request.error || new Error('IndexedDB request failed'));
          });
        }

        function transactionDone(transaction){
          return new Promise((resolve,reject)=>{
            transaction.oncomplete = ()=> resolve();
            transaction.onerror = ()=> reject(transaction.error || new Error('IndexedDB transaction failed'));
            transaction.onabort = ()=> reject(transaction.error || new Error('IndexedDB transaction aborted'));
          });
        }

        function openDb(dbName, storeName, version){
          if (dbPromise) return dbPromise;
          dbPromise = new Promise((resolve,reject)=>{
            const request = indexedDB.open(dbName, version || DB_VERSION);
            request.onupgradeneeded = ()=>{
              const database = request.result;
              if (database.objectStoreNames.contains(storeName)){
                try{
                  const store = request.transaction?.objectStore(storeName);
                  if (store && typeof store.indexNames?.contains === 'function' && !store.indexNames.contains(CONTENT_HASH_INDEX)){
                    store.createIndex(CONTENT_HASH_INDEX, 'contentHash', { unique: false });
                  }
                }catch(e){/* ignore */}
                return;
              }
              const created = database.createObjectStore(storeName, { keyPath: 'id' });
              try{ created.createIndex(CONTENT_HASH_INDEX, 'contentHash', { unique: false }); }catch(e){}
            };
            request.onsuccess = ()=> resolve(request.result);
            request.onerror = ()=> reject(request.error || new Error('Failed to open IndexedDB in worker'));
          });
          return dbPromise;
        }

        self.onmessage = async (ev)=>{
          const msg = ev.data;
          try{
            if (!msg) return;
            if (msg.type === 'putBatch'){
              const db = await openDb(msg.dbName, msg.storeName, msg.version);
              const records = msg.records || [];
              const chunkSize = msg.chunkSize || 64;
              for (let i = 0; i < records.length; i += chunkSize){
                const end = Math.min(i + chunkSize, records.length);
                const tx = db.transaction(msg.storeName, 'readwrite');
                const store = tx.objectStore(msg.storeName);
                for (let j = i; j < end; j++){
                  const r = records[j];
                  let embedding = null;
                  if (r && r.embedding && r.embedding.type === 'buffer'){
                    // reconstruct typed array view from transferred buffer
                    try{ embedding = new Float32Array(r.embedding.buffer, 0, r.embedding.length); }catch(e){ embedding = new Float32Array(r.embedding.buffer); }
                  } else if (r && r.embedding && r.embedding.type === 'array'){
                    embedding = r.embedding.array;
                  } else {
                    embedding = r.embedding;
                  }

                  const recToPut = {
                    id: r.id,
                    content: r.content,
                    embedding: embedding,
                    metadata: r.metadata,
                    contentHash: r.contentHash,
                    cacheKey: r.cacheKey,
                    embeddingSpace: r.embeddingSpace,
                  };
                  try{ store.put(recToPut); }catch(e){ /* swallow per-record put errors to allow tx to fail/propagate */ }
                }
                await transactionDone(tx);
              }
              self.postMessage({ type: 'putBatchAck', id: msg.id });
            } else if (msg.type === 'init'){
              await openDb(msg.dbName, msg.storeName, msg.version);
              self.postMessage({ type: 'initAck' });
            } else if (msg.type === 'close'){
              if (dbPromise){
                try{ const db = await dbPromise; db.close(); }catch(e){}
                dbPromise = null;
              }
              self.postMessage({ type: 'closeAck', id: msg.id });
            }
          }catch(err){
            try{ self.postMessage({ type: 'putBatchError', id: msg.id, error: String(err) }); }catch(e){}
          }
        };
      `;

			const blob = new Blob([workerSource], { type: "application/javascript" });
			// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
			this.#worker = new Worker(URL.createObjectURL(blob));

			this.#worker.onmessage = (ev: MessageEvent) => {
				const data = ev.data as { type?: string; id?: number; error?: string };
				if (!data || typeof data.type !== "string") return;
				if (data.type === "putBatchAck" && typeof data.id === "number") {
					const entry = this.#pendingWorkerResponses.get(data.id);
					if (entry) {
						try {
							for (const req of entry.pendingRequests) {
								try {
									req.resolve();
								} catch (_) {}
							}
							entry.resolve();
						} catch (_) {}
						this.#pendingWorkerResponses.delete(data.id);
					}
					return;
				}

				if (data.type === "putBatchError" && typeof data.id === "number") {
					const entry = this.#pendingWorkerResponses.get(data.id);
					if (entry) {
						try {
							entry.reject(new Error(String(data.error ?? "worker error")));
						} catch (_) {}
						for (const req of entry.pendingRequests) {
							try {
								req.reject(new Error(String(data.error ?? "worker error")));
							} catch (_) {}
						}
						this.#pendingWorkerResponses.delete(data.id);
					}
					return;
				}
			};
		} catch (err) {
			// If worker creation fails, disable worker usage and continue with
			// main-thread writes as a fallback.
			// eslint-disable-next-line no-console
			console.warn(
				"IndexedDB worker creation failed, falling back to main-thread writes",
				err,
			);
			this.#useWriteWorker = false;
			this.#worker = null;
		}
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
		return records.map((r) => normalizeStoredVectorRecord(r as unknown as any));
	}

	async iterateAll<T>(
		callback: (record: T) => Promise<boolean | undefined> | boolean | undefined,
	): Promise<void> {
		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readonly");
		await iterateRecords<any>(
			transaction.objectStore(this.#storeName),
			(rec) => {
				const normalized = normalizeStoredVectorRecord(rec as any);
				return callback(normalized as unknown as T);
			},
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
			const uniqueContentHashes = Array.from(new Set(contentHashes));
			const matchesByHash = new Map<string, StoredVectorRecord[]>();
			const indexSource = store.index(VECTOR_STORE_SCHEMA.contentHashIndex);

			// Try to query the index per-hash to avoid scanning the entire index.
			// If per-hash queries are not supported or fail, fall back to a single
			// index scan for compatibility.
			let usedFallback = false;
			try {
				await Promise.all(
					uniqueContentHashes.map(async (hash) => {
						try {
							const recs = await getAllRecords<StoredVectorRecord>(
								indexSource,
								hash,
							);
							if (recs && recs.length > 0) {
								matchesByHash.set(
									hash,
									recs.map((r) => normalizeStoredVectorRecord(r as any)),
								);
							}
						} catch (err) {
							usedFallback = true;
						}
					}),
				);
			} catch (_err) {
				usedFallback = true;
			}

			if (usedFallback) {
				const uniqueSet = new Set(uniqueContentHashes);
				const indexRecords =
					await getAllRecords<StoredVectorRecord>(indexSource);
				for (const record of indexRecords) {
					const normalized = normalizeStoredVectorRecord(record as any);
					const contentHash = normalized.contentHash;
					if (!contentHash || !uniqueSet.has(contentHash)) continue;
					const arr = matchesByHash.get(contentHash);
					if (arr) arr.push(normalized);
					else matchesByHash.set(contentHash, [normalized]);
				}
			}

			const directMatches = contentHashes.map((contentHash, index) => {
				const matches = matchesByHash.get(contentHash) ?? [];
				return (
					matches.find(
						(match) =>
							match.content === contents[index] && predicate(match, index),
					) ?? null
				);
			});

			await transactionDone(transaction);
			return directMatches;
		}

		const allRecords = (await getAllRecords<StoredVectorRecord>(store)).map(
			(r) => normalizeStoredVectorRecord(r as any),
		);
		// Fallback path: if the contentHash index is missing, this performs a full
		// scan of the store so old database versions remain compatible.
		const recordByHash = new Map<string, StoredVectorRecord[]>();
		const recordByContent = new Map<string, StoredVectorRecord[]>();

		for (const record of allRecords) {
			if (record.contentHash) {
				const entries = recordByHash.get(record.contentHash) ?? [];
				entries.push(record);
				recordByHash.set(record.contentHash, entries);
			}

			const contentEntries = recordByContent.get(record.content) ?? [];
			contentEntries.push(record);
			recordByContent.set(record.content, contentEntries);
		}

		const records = contentHashes.map((contentHash, index) => {
			const matchesByHash = recordByHash.get(contentHash) ?? [];
			const hashMatch = matchesByHash.find(
				(record) =>
					record.content === contents[index] && predicate(record, index),
			);
			if (hashMatch) {
				return hashMatch;
			}

			const matchesByContent = recordByContent.get(contents[index]) ?? [];
			return (
				matchesByContent.find((record) => predicate(record, index)) ?? null
			);
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
		// Coalesce multiple concurrent `put()` calls that occur within the same
		// event loop tick into a single batched flush. This reduces the number of
		// transactions and structured-clone operations performed by IndexedDB.
		return new Promise<void>((resolve, reject) => {
			this.#pendingPutRequests.push({ records, resolve, reject });
			if (!this.#flushScheduled) {
				this.#flushScheduled = true;
				// Schedule flush on next macrotask to allow other puts to join.
				setTimeout(() => {
					void this.#flushPendingPuts();
				}, 0);
			}
		});
	}

	async #flushPendingPuts(): Promise<void> {
		this.#flushScheduled = false;
		const pending = this.#pendingPutRequests.splice(
			0,
			this.#pendingPutRequests.length,
		);
		if (pending.length === 0) return;

		// Flatten records (shallow) without cloning record objects.
		const combined: StoredVectorRecord[] = [];
		for (const req of pending) {
			for (let i = 0; i < req.records.length; i++) {
				combined.push(req.records[i]);
			}
		}

		try {
			// Prefer offloading writes to a worker when available to avoid
			// main-thread structured-clone costs for large numeric arrays.
			if (this.#useWriteWorker) {
				await this.#ensureWorker();
			}

			// Determine chunk size (respect explicit override, cached value,
			// otherwise compute from combined records).
			let chunkSize: number;
			if (
				typeof this.#putChunkSize === "number" &&
				Number.isFinite(this.#putChunkSize) &&
				this.#putChunkSize > 0
			) {
				chunkSize = Math.max(1, Math.floor(this.#putChunkSize));
			} else {
				if (this.#cachedPutChunkSize == null) {
					this.#cachedPutChunkSize = computePutChunkSize(combined);
				}
				chunkSize = this.#cachedPutChunkSize ?? 64;
			}

			// If worker is available, serialize records and transfer large
			// Float32Array buffers to the worker; otherwise perform writes on
			// the main thread as before.
			if (this.#worker) {
				const serialized: any[] = [];
				const transferList: ArrayBuffer[] = [];

				for (let i = 0; i < combined.length; i++) {
					const rec = combined[i] as StoredVectorRecord & { embedding?: any };
					const out: any = {
						id: rec.id,
						content: rec.content,
						metadata: rec.metadata,
						contentHash: rec.contentHash,
						cacheKey: rec.cacheKey,
						embeddingSpace: rec.embeddingSpace,
					};

					const { serializedEmbedding, transferList: embeddingTransferList } =
						serializeEmbeddingForTransfer(rec.embedding, {
							transferThreshold: this.#typedArrayTransferThreshold,
							transferOwnership: this.#transferOwnership,
							persistEmbeddingAs: this.#persistEmbeddingAs,
						});

					out.embedding = serializedEmbedding;
					transferList.push(...embeddingTransferList);
					serialized.push(out);
				}

				const msgId = ++this.#workerMsgId;
				// track pending so worker response can resolve/reject original promises
				const ackPromise = new Promise<void>((resolve, reject) => {
					this.#pendingWorkerResponses.set(msgId, {
						resolve,
						reject,
						pendingRequests: pending,
					});
				});

				try {
					this.#worker.postMessage(
						{
							type: "putBatch",
							id: msgId,
							dbName: this.#dbName,
							storeName: this.#storeName,
							version: VECTOR_STORE_SCHEMA.currentVersion,
							chunkSize,
							records: serialized,
						},
						transferList,
					);

					// await worker ack and then resolve individual promises
					await ackPromise;
				} catch (err) {
					// On worker failure, reject pending and fall back to main-thread
					// writes for correctness.
					for (const req of pending) {
						try {
							req.reject(err);
						} catch (_e) {}
					}
					// disable worker usage for future attempts
					this.#useWriteWorker = false;
					try {
						if (this.#worker) {
							try {
								this.#worker.terminate();
							} catch (_) {}
						}
					} finally {
						this.#worker = null;
					}
					throw err;
				}
			} else {
				const database = await this.open();
				const total = combined.length;
				for (let i = 0; i < total; i += chunkSize) {
					const end = Math.min(i + chunkSize, total);
					const transaction = database.transaction(
						this.#storeName,
						"readwrite",
					);
					const store = transaction.objectStore(this.#storeName);
					for (let j = i; j < end; j++) {
						// Write records directly from the combined array to avoid extra
						// allocations or cloning.
						store.put(combined[j]);
					}

					try {
						const txAny = transaction as unknown as { commit?: () => void };
						if (typeof txAny.commit === "function") {
							txAny.commit();
						}
					} catch (_err) {
						// ignore commit errors
					}

					await transactionDone(transaction);
				}

				// All writes succeeded; resolve individual promises.
				for (const req of pending) {
					try {
						req.resolve();
					} catch (_err) {
						// ignore resolver errors
					}
				}
			}
		} catch (err) {
			// Propagate error to all pending requests.
			for (const req of pending) {
				try {
					req.reject(err);
				} catch (_e) {
					// ignore
				}
			}
			throw err;
		}
	}

	async clear(): Promise<void> {
		const database = await this.open();
		const transaction = database.transaction(this.#storeName, "readwrite");
		transaction.objectStore(this.#storeName).clear();
		await transactionDone(transaction);
	}
}
