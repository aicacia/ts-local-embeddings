import { Document, type DocumentInterface } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import {
	cosineSimilarity,
	maximalMarginalRelevance,
} from "@langchain/core/utils/math";

type StoredVectorRecord = {
	id: string;
	content: string;
	contentHash?: string;
	embedding: number[];
	metadata: Record<string, unknown>;
};

type PendingEmbeddingGroup = {
	representativeDocument: Document;
	indices: number[];
};

export type IndexedDBVectorStoreFilter = (doc: Document) => boolean;

export type IndexedDBVectorStoreArgs = {
	dbName?: string;
	storeName?: string;
	similarity?: (a: number[], b: number[]) => number;
};

const DEFAULT_DB_NAME = "langchain-indexeddb-vectorstore";
const DEFAULT_STORE_NAME = "vectors";
const DB_VERSION = 2;
const CONTENT_HASH_INDEX = "by_content_hash";

function cosine(a: number[], b: number[]): number {
	const similarityMatrix = cosineSimilarity([a], [b]);
	const score = similarityMatrix[0]?.[0];
	return Number.isFinite(score) ? score : 0;
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		request.onsuccess = () => resolve(request.result);
		request.onerror = () =>
			reject(request.error ?? new Error("IndexedDB request failed."));
	});
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		transaction.oncomplete = () => resolve();
		transaction.onerror = () =>
			reject(transaction.error ?? new Error("IndexedDB transaction failed."));
		transaction.onabort = () =>
			reject(transaction.error ?? new Error("IndexedDB transaction aborted."));
	});
}

function createDocument(record: StoredVectorRecord): Document {
	return new Document({
		id: record.id,
		pageContent: record.content,
		metadata: record.metadata,
	});
}

function toStoredVectorRecord(args: {
	id: string;
	document: Document;
	contentHash: string;
	embedding: number[];
}): StoredVectorRecord {
	const metadata =
		(args.document.metadata as Record<string, unknown> | undefined) ?? {};

	return {
		id: args.id,
		content: args.document.pageContent,
		contentHash: args.contentHash,
		embedding: args.embedding,
		metadata,
	};
}

function fallbackHash(input: string): string {
	let hash = 5381;
	for (let index = 0; index < input.length; index += 1) {
		hash = (hash * 33) ^ input.charCodeAt(index);
	}

	return `fallback-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

async function sha256(input: string): Promise<string> {
	if (typeof crypto === "undefined" || !crypto.subtle) {
		return fallbackHash(input);
	}

	const data = new TextEncoder().encode(input);
	const digest = await crypto.subtle.digest("SHA-256", data);
	return Array.from(new Uint8Array(digest), (byte) =>
		byte.toString(16).padStart(2, "0"),
	).join("");
}

function resolveRecordId(document: Document, fallbackIndex: number): string {
	const metadataId = (document.metadata as { id?: unknown } | undefined)?.id;
	if (typeof metadataId === "string" && metadataId.trim().length > 0) {
		return metadataId;
	}

	if (typeof metadataId === "number" && Number.isFinite(metadataId)) {
		return String(metadataId);
	}

	if (
		typeof crypto !== "undefined" &&
		typeof crypto.randomUUID === "function"
	) {
		return crypto.randomUUID();
	}

	return `doc-${Date.now()}-${fallbackIndex}`;
}

export class IndexedDBVectorStore {
	readonly #embeddings: EmbeddingsInterface<number[]>;
	readonly #dbName: string;
	readonly #storeName: string;
	readonly #similarity: (a: number[], b: number[]) => number;
	#dbPromise: Promise<IDBDatabase> | null = null;

	constructor(
		embeddings: EmbeddingsInterface<number[]>,
		args: IndexedDBVectorStoreArgs = {},
	) {
		this.#embeddings = embeddings;
		this.#dbName = args.dbName ?? DEFAULT_DB_NAME;
		this.#storeName = args.storeName ?? DEFAULT_STORE_NAME;
		this.#similarity = args.similarity ?? cosine;
	}

	async addDocuments(documents: Document[]): Promise<void> {
		if (documents.length === 0) {
			return;
		}

		const resolvedIds = documents.map((document, index) =>
			resolveRecordId(document, index),
		);
		const contentHashes = await Promise.all(
			documents.map((document) => sha256(document.pageContent)),
		);
		const cachedRecords = await this.#getRecordsByContentHash(
			contentHashes,
			documents.map((document) => document.pageContent),
		);

		const recordsToWrite: StoredVectorRecord[] = [];
		const pendingEmbeddingGroups = new Map<string, PendingEmbeddingGroup>();

		for (let index = 0; index < documents.length; index += 1) {
			const document = documents[index];
			const id = resolvedIds[index];
			const contentHash = contentHashes[index];
			const cachedRecord = cachedRecords[index];

			if (cachedRecord) {
				recordsToWrite.push(
					toStoredVectorRecord({
						id,
						document,
						contentHash,
						embedding: cachedRecord.embedding,
					}),
				);
				continue;
			}

			const groupKey = `${contentHash}:${document.pageContent}`;
			const existingGroup = pendingEmbeddingGroups.get(groupKey);
			if (existingGroup) {
				existingGroup.indices.push(index);
				continue;
			}

			pendingEmbeddingGroups.set(groupKey, {
				representativeDocument: document,
				indices: [index],
			});
		}

		const uniquePendingGroups = Array.from(pendingEmbeddingGroups.values());
		if (uniquePendingGroups.length > 0) {
			const embeddedVectors = await this.#embeddings.embedDocuments(
				uniquePendingGroups.map(
					(group) => group.representativeDocument.pageContent,
				),
			);

			if (embeddedVectors.length !== uniquePendingGroups.length) {
				throw new Error(
					`Embedding runtime produced ${embeddedVectors.length} vectors for ${uniquePendingGroups.length} documents.`,
				);
			}

			for (
				let groupIndex = 0;
				groupIndex < uniquePendingGroups.length;
				groupIndex += 1
			) {
				const group = uniquePendingGroups[groupIndex];
				const embedding = embeddedVectors[groupIndex] ?? [];

				for (const originalIndex of group.indices) {
					const document = documents[originalIndex];
					recordsToWrite.push(
						toStoredVectorRecord({
							id: resolvedIds[originalIndex],
							document,
							contentHash: contentHashes[originalIndex],
							embedding,
						}),
					);
				}
			}
		}

		await this.#putRecords(recordsToWrite);
	}

	async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
		if (vectors.length !== documents.length) {
			throw new Error(
				`Expected vectors/documents lengths to match, got vectors=${vectors.length}, documents=${documents.length}.`,
			);
		}

		if (vectors.length === 0) {
			return;
		}

		const resolvedIds = documents.map((document, index) =>
			resolveRecordId(document, index),
		);
		const contentHashes = await Promise.all(
			documents.map((document) => sha256(document.pageContent)),
		);

		const database = await this.#getDatabase();
		const transaction = database.transaction(this.#storeName, "readwrite");
		const store = transaction.objectStore(this.#storeName);

		for (let index = 0; index < vectors.length; index += 1) {
			const document = documents[index];
			const embedding = vectors[index] ?? [];
			const id = resolvedIds[index];
			const contentHash = contentHashes[index];

			const record = toStoredVectorRecord({
				id,
				document,
				contentHash,
				embedding,
			});

			store.put(record);
		}

		await transactionDone(transaction);
	}

	async #putRecords(records: StoredVectorRecord[]): Promise<void> {
		if (records.length === 0) {
			return;
		}

		const database = await this.#getDatabase();
		const transaction = database.transaction(this.#storeName, "readwrite");
		const store = transaction.objectStore(this.#storeName);

		for (const record of records) {
			store.put(record);
		}

		await transactionDone(transaction);
	}

	async similaritySearch(
		query: string,
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<Document[]> {
		const queryEmbedding = await this.#embeddings.embedQuery(query);
		const matches = await this.similaritySearchVectorWithScore(
			queryEmbedding,
			k,
			filter,
		);
		return matches.map(([document]) => document);
	}

	async similaritySearchWithScore(
		query: string,
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<Array<[Document, number]>> {
		const queryEmbedding = await this.#embeddings.embedQuery(query);
		return this.similaritySearchVectorWithScore(queryEmbedding, k, filter);
	}

	async similaritySearchVectorWithScore(
		query: number[],
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<Array<[Document, number]>> {
		const matches = await this.#queryVectors(query, k, filter);
		return matches.map((match) => [
			createDocument({
				id: match.id,
				content: match.content,
				embedding: match.embedding,
				metadata: match.metadata,
			}),
			match.similarity,
		]);
	}

	async maxMarginalRelevanceSearch(
		query: string,
		options: {
			k: number;
			fetchK?: number;
			lambda?: number;
			filter?: IndexedDBVectorStoreFilter;
		},
	): Promise<DocumentInterface[]> {
		const queryEmbedding = await this.#embeddings.embedQuery(query);
		const searches = await this.#queryVectors(
			queryEmbedding,
			options.fetchK ?? 20,
			options.filter,
		);
		const selectedIndices = maximalMarginalRelevance(
			queryEmbedding,
			searches.map((search) => search.embedding),
			options.lambda,
			options.k,
		);

		return selectedIndices.map(
			(index) =>
				new Document({
					metadata: searches[index].metadata,
					pageContent: searches[index].content,
					id: searches[index].id,
				}),
		);
	}

	async clear(): Promise<void> {
		const database = await this.#getDatabase();
		const transaction = database.transaction(this.#storeName, "readwrite");
		transaction.objectStore(this.#storeName).clear();
		await transactionDone(transaction);
	}

	async count(): Promise<number> {
		const database = await this.#getDatabase();
		const transaction = database.transaction(this.#storeName, "readonly");
		const count = await requestToPromise(
			transaction.objectStore(this.#storeName).count(),
		);
		await transactionDone(transaction);
		return count;
	}

	async close(): Promise<void> {
		if (!this.#dbPromise) {
			return;
		}

		const database = await this.#dbPromise;
		database.close();
		this.#dbPromise = null;
	}

	static async fromTexts(
		texts: string[],
		metadatas: object[] | object,
		embeddings: EmbeddingsInterface<number[]>,
		args?: IndexedDBVectorStoreArgs,
	): Promise<IndexedDBVectorStore> {
		const metadataArray = Array.isArray(metadatas)
			? metadatas
			: Array.from({ length: texts.length }, () => metadatas);

		const documents = texts.map(
			(text, index) =>
				new Document({
					pageContent: text,
					metadata:
						(metadataArray[index] as Record<string, unknown> | undefined) ?? {},
				}),
		);

		return IndexedDBVectorStore.fromDocuments(documents, embeddings, args);
	}

	static async fromDocuments(
		documents: Document[],
		embeddings: EmbeddingsInterface<number[]>,
		args?: IndexedDBVectorStoreArgs,
	): Promise<IndexedDBVectorStore> {
		const store = new IndexedDBVectorStore(embeddings, args);
		await store.addDocuments(documents);
		return store;
	}

	static async fromExistingIndex(
		embeddings: EmbeddingsInterface<number[]>,
		args?: IndexedDBVectorStoreArgs,
	): Promise<IndexedDBVectorStore> {
		const store = new IndexedDBVectorStore(embeddings, args);
		await store.#getDatabase();
		return store;
	}

	#getDatabase(): Promise<IDBDatabase> {
		if (typeof indexedDB === "undefined") {
			throw new Error("IndexedDB is not available in this environment.");
		}

		if (this.#dbPromise) {
			return this.#dbPromise;
		}

		this.#dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
			const request = indexedDB.open(this.#dbName, DB_VERSION);

			request.onupgradeneeded = () => {
				const database = request.result;
				const store = database.objectStoreNames.contains(this.#storeName)
					? request.transaction?.objectStore(this.#storeName)
					: database.createObjectStore(this.#storeName, { keyPath: "id" });

				if (store && !store.indexNames.contains(CONTENT_HASH_INDEX)) {
					store.createIndex(CONTENT_HASH_INDEX, "contentHash", {
						unique: false,
					});
				}
			};

			request.onsuccess = () => resolve(request.result);
			request.onerror = () => {
				this.#dbPromise = null;
				reject(
					request.error ?? new Error("Failed to open IndexedDB database."),
				);
			};
		});

		return this.#dbPromise;
	}

	async #getRecordsByContentHash(
		contentHashes: string[],
		contents: string[],
	): Promise<Array<StoredVectorRecord | null>> {
		if (contentHashes.length === 0) {
			return [];
		}

		const database = await this.#getDatabase();
		const transaction = database.transaction(this.#storeName, "readonly");
		const store = transaction.objectStore(this.#storeName);

		let directMatches: Array<StoredVectorRecord | undefined> = Array.from(
			{ length: contentHashes.length },
			() => undefined,
		);
		if (store.indexNames.contains(CONTENT_HASH_INDEX)) {
			directMatches = await Promise.all(
				contentHashes.map((contentHash) =>
					requestToPromise(store.index(CONTENT_HASH_INDEX).get(contentHash)),
				),
			);
			await transactionDone(transaction);
			return directMatches.map((match) => match ?? null);
		}

		const allRecords = await requestToPromise(store.getAll());
		const recordByHash = new Map<string, StoredVectorRecord>();
		const recordByContent = new Map<string, StoredVectorRecord>();

		for (const record of allRecords) {
			if (record.contentHash) {
				recordByHash.set(record.contentHash, record);
			}
			recordByContent.set(record.content, record);
		}

		const records = contentHashes.map((contentHash, index) => {
			const directMatch = directMatches[index];
			if (directMatch) {
				return directMatch;
			}

			return (
				recordByHash.get(contentHash) ??
				recordByContent.get(contents[index]) ??
				null
			);
		});

		await transactionDone(transaction);
		return records;
	}

	async #getAllRecords(): Promise<StoredVectorRecord[]> {
		const database = await this.#getDatabase();
		const transaction = database.transaction(this.#storeName, "readonly");
		const records = await requestToPromise(
			transaction.objectStore(this.#storeName).getAll(),
		);
		await transactionDone(transaction);
		return records;
	}

	async #queryVectors(
		query: number[],
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<
		Array<{
			similarity: number;
			metadata: Record<string, unknown>;
			content: string;
			embedding: number[];
			id: string;
		}>
	> {
		const records = await this.#getAllRecords();
		return records
			.filter((record) => {
				if (!filter) {
					return true;
				}

				return filter(
					new Document({
						metadata: record.metadata,
						pageContent: record.content,
						id: record.id,
					}),
				);
			})
			.map((record) => ({
				similarity: this.#similarity(query, record.embedding),
				metadata: record.metadata,
				content: record.content,
				embedding: record.embedding,
				id: record.id,
			}))
			.sort((left, right) => right.similarity - left.similarity)
			.slice(0, Math.max(0, k));
	}
}
