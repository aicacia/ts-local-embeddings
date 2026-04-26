import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { maximalMarginalRelevance } from "@langchain/core/utils/math";
import {
	createVectorWritePipeline,
	type StoredVectorRecord,
	type VectorWritePipeline,
} from "./vectorWritePipeline.js";
import {
	IndexedDbStoreGateway,
	VECTOR_STORE_SCHEMA,
} from "./indexedDbStoreGateway.js";
import type { VectorStoreGateway } from "./VectorStoreGateway.js";
import type { StorageGatewayPort } from "./indexedDbStoreGateway.js";
import { computeVectorNorm, computeSimilarity } from "./vectorMathUtils.js";
import { selectTopK } from "../utils/topKUtils.js";
import { embedDocuments, embedQuery } from "../utils/embeddingUtils.js";
import { createDocument, normalizeMetadata } from "../utils/documentUtils.js";
import {
	assertEqualLengthVectors,
	calculateDotAndNorms,
	cosineSimilarity,
	cosineSimilarityWithQueryNorm,
} from "./mathUtils.js";

export type IndexedDBVectorStoreFilter = (doc: Document) => boolean;

export type IndexedDBVectorStoreArgs = {
	dbName?: string;
	storeName?: string;
	gateway?: StorageGatewayPort;
	similarity?: (a: ArrayLike<number>, b: ArrayLike<number>) => number;
	// When the store has more than `getAllThreshold` records, queries will
	// stream via `iterateAll` instead of materializing the entire DB via
	// `getAll()` to reduce memory pressure. Default: 10000.
	getAllThreshold?: number;
};

// Vector math helpers are provided by ./mathUtils.js

// Use shared createDocument from documentUtils

type Match = {
	similarity: number;
	metadata: Record<string, unknown>;
	content: string;
	embedding: ArrayLike<number>;
	id: string;
};

type EmbeddingProvenanceSource = {
	getEmbeddingProvenance?: () => Promise<string>;
};

// Bounded min-heap implementation to maintain the top-K matches more
// efficiently than repeated splices for larger K. Heap stores the lowest
// similarity at the top.
// Using shared `MinHeap` from utils/heapUtils.ts

const DEFAULT_EMBEDDING_SPACE = "default";

/**
 * IndexedDBVectorStore provides persistent vector storage using IndexedDB.
 * Supports similarity search, MMR, and batch operations.
 */
export class IndexedDBVectorStore {
	readonly #embeddings: EmbeddingsInterface<number[] | Float32Array>;
	readonly #dbName: string;
	readonly #storeName: string;
	readonly #similarity: (a: ArrayLike<number>, b: ArrayLike<number>) => number;
	readonly #writePipeline: VectorWritePipeline;
	readonly #gateway: VectorStoreGateway;
	#embeddingSpacePromise: Promise<string> | null = null;
	#getAllThreshold = 10000;

	/**
	 * Create an IndexedDBVectorStore instance.
	 * @param embeddings - Embeddings interface for vectorization.
	 * @param args - Optional configuration for DB, store, and similarity.
	 */
	constructor(
		embeddings: EmbeddingsInterface<number[] | Float32Array>,
		args: IndexedDBVectorStoreArgs = {},
	) {
		if (
			args.gateway &&
			(args.dbName !== undefined || args.storeName !== undefined)
		) {
			throw new Error(
				"Cannot specify dbName or storeName when a custom gateway is provided.",
			);
		}

		this.#embeddings = embeddings;
		this.#dbName = args.dbName ?? VECTOR_STORE_SCHEMA.defaultDbName;
		this.#storeName = args.storeName ?? VECTOR_STORE_SCHEMA.defaultStoreName;
		this.#similarity = args.similarity ?? cosineSimilarity;
		this.#gateway =
			args.gateway ??
			new IndexedDbStoreGateway({
				dbName: this.#dbName,
				storeName: this.#storeName,
			});
		this.#writePipeline = createVectorWritePipeline({
			embeddings: this.#embeddings,
			resolveEmbeddingSpace: () => this.#resolveEmbeddingSpace(),
			getCachedRecords: (embeddingSpace, contentHashes, contents) =>
				this.#getRecordsByContentHash(embeddingSpace, contentHashes, contents),
			putRecords: (records) => this.#putRecords(records),
		});

		this.#getAllThreshold =
			typeof args.getAllThreshold === "number" &&
			Number.isFinite(args.getAllThreshold) &&
			args.getAllThreshold > 0
				? Math.max(1, Math.floor(args.getAllThreshold))
				: 10000;
	}

	/**
	 * Add an array of documents to the store.
	 * @param documents - Documents to add.
	 */
	async addDocuments(documents: Document[]): Promise<void> {
		await this.#writePipeline.addDocuments(documents);
	}

	/**
	 * Add precomputed vectors and their documents to the store.
	 * @param vectors - Array of vectors.
	 * @param documents - Corresponding documents.
	 */
	async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
		await this.#writePipeline.addVectors(vectors, documents);
	}

	async #putRecords(records: StoredVectorRecord[]): Promise<void> {
		await this.#gateway.put(records);
	}

	/**
	 * Perform a similarity search for documents matching a query string.
	 * @param query - Query string.
	 * @param k - Number of top results.
	 * @param filter - Optional filter function.
	 * @returns Array of matching documents.
	 */
	async similaritySearch(
		query: string,
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<Document[]> {
		const queryEmbedding = await embedQuery(this.#embeddings, query);
		const matches = await this.similaritySearchVectorWithScore(
			queryEmbedding,
			k,
			filter,
		);
		return matches.map(([document]) => document);
	}

	/**
	 * Perform a similarity search and return documents with their similarity scores.
	 * @param query - Query string.
	 * @param k - Number of top results.
	 * @param filter - Optional filter function.
	 * @returns Array of [Document, score] pairs.
	 */
	async similaritySearchWithScore(
		query: string,
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<Array<[Document, number]>> {
		const queryEmbedding = await embedQuery(this.#embeddings, query);
		return this.similaritySearchVectorWithScore(queryEmbedding, k, filter);
	}

	async similaritySearchVectorWithScore(
		query: ArrayLike<number>,
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<Array<[Document, number]>> {
		const matches = await this.#queryVectors(query, k, filter);
		return matches.map((match) => [
			createDocument(match.content, match.metadata, match.id),
			match.similarity,
		]);
	}

	/**
	 * Perform a Maximal Marginal Relevance (MMR) search for diverse results.
	 * @param query - Query string.
	 * @param options - MMR options (k, fetchK, lambda, filter).
	 * @returns Array of matching documents.
	 */
	async maxMarginalRelevanceSearch(
		query: string,
		options: {
			k: number;
			fetchK?: number;
			lambda?: number;
			filter?: IndexedDBVectorStoreFilter;
		},
	): Promise<Document[]> {
		const queryEmbedding = await embedQuery(this.#embeddings, query);
		const searches = await this.#queryVectors(
			queryEmbedding as ArrayLike<number>,
			options.fetchK ?? 20,
			options.filter,
		);
		const selectedIndices = maximalMarginalRelevance(
			Array.from(queryEmbedding as ArrayLike<number>),
			searches.map((search) =>
				Array.from(search.embedding as ArrayLike<number>),
			),
			options.lambda,
			options.k,
		);
		return selectedIndices.map((index) =>
			createDocument(
				searches[index].content,
				searches[index].metadata,
				searches[index].id,
			),
		);
	}

	/**
	 * Clear all records from the store.
	 */
	async clear(): Promise<void> {
		await this.#gateway.clear();
	}

	/**
	 * Get the number of records in the store.
	 * @returns Number of records.
	 */
	async count(): Promise<number> {
		return this.#gateway.count();
	}

	/**
	 * Close the underlying database connection.
	 */
	async close(): Promise<void> {
		await this.#gateway.close();
	}

	static async fromTexts(
		texts: string[],
		metadatas: object[] | object,
		embeddings: EmbeddingsInterface<number[] | Float32Array>,
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
		embeddings: EmbeddingsInterface<number[] | Float32Array>,
		args?: IndexedDBVectorStoreArgs,
	): Promise<IndexedDBVectorStore> {
		const store = new IndexedDBVectorStore(embeddings, args);
		await store.addDocuments(documents);
		return store;
	}

	static async fromExistingIndex(
		embeddings: EmbeddingsInterface<number[] | Float32Array>,
		args?: IndexedDBVectorStoreArgs,
	): Promise<IndexedDBVectorStore> {
		const store = new IndexedDBVectorStore(embeddings, args);
		await store.#gateway.open();
		return store;
	}

	async #getRecordsByContentHash(
		embeddingSpace: string,
		contentHashes: string[],
		contents: string[],
	): Promise<StoredVectorRecord[]> {
		const records = await this.#gateway.queryByContentHash(
			contentHashes,
			contents,
			(record, index) =>
				(index < 0 || record.content === contents[index]) &&
				this.#matchesEmbeddingSpace(record, embeddingSpace),
		);

		return records.filter((r): r is StoredVectorRecord => r !== null);
	}

	async #getAllRecords(): Promise<StoredVectorRecord[]> {
		return this.#gateway.getAll();
	}

	async #queryVectors(
		query: ArrayLike<number>,
		k: number,
		filter?: IndexedDBVectorStoreFilter,
	): Promise<
		Array<{
			similarity: number;
			metadata: Record<string, unknown>;
			content: string;
			embedding: ArrayLike<number>;
			id: string;
		}>
	> {
		if (k <= 0) return [];

		const useOptimizedCosine = this.#similarity === cosineSimilarity;
		const queryNorm = useOptimizedCosine ? computeVectorNorm(query) : 0;
		const computeSimilarity = (
			embedding: ArrayLike<number>,
			recordNorm?: number,
		): number => {
			if (useOptimizedCosine && typeof recordNorm === "number") {
				if (recordNorm === 0 || queryNorm === 0) return 0;
				let dot = 0;
				for (let i = 0; i < embedding.length; i += 1) {
					dot += (query[i] as number) * (embedding[i] as number);
				}
				const score = dot / (queryNorm * recordNorm);
				return Number.isFinite(score) ? score : 0;
			}
			return useOptimizedCosine
				? cosineSimilarityWithQueryNorm(query, queryNorm, embedding)
				: this.#similarity(query, embedding);
		};

		// Decide whether to fetch all records or stream via cursor. Streaming avoids materializing the entire DB into memory when the store is large.
		let all: StoredVectorRecord[] = [];
		let useStream = false;
		try {
			const total = await this.#gateway.count();
			useStream = total > this.#getAllThreshold;
		} catch {
			useStream = true;
		}

		if (useStream) {
			await this.#gateway.iterateAll<StoredVectorRecord>(async (record) => {
				all.push(record);
				return true;
			});
		} else {
			try {
				all = await this.#gateway.getAll();
			} catch {
				await this.#gateway.iterateAll<StoredVectorRecord>(async (record) => {
					all.push(record);
					return true;
				});
			}
		}

		const scored = all
			.filter((record) => {
				if (!filter) return true;
				const doc = createDocument(record.content, record.metadata, record.id);
				return filter(doc);
			})
			.map((record) => {
				const embedding = record.embedding as ArrayLike<number>;
				const recordNorm = (record as any).embeddingNorm as number | undefined;
				return {
					similarity: computeSimilarity(embedding, recordNorm),
					metadata: record.metadata,
					content: record.content,
					embedding: embedding,
					id: record.id,
				};
			});
		return selectTopK(scored, k);
	}

	#matchesEmbeddingSpace(
		record: StoredVectorRecord,
		embeddingSpace: string,
	): boolean {
		const recordSpace = record.embeddingSpace ?? DEFAULT_EMBEDDING_SPACE;
		return recordSpace === embeddingSpace;
	}

	#resolveEmbeddingSpace(): Promise<string> {
		if (!this.#embeddingSpacePromise) {
			this.#embeddingSpacePromise = Promise.resolve(DEFAULT_EMBEDDING_SPACE);
		}
		return this.#embeddingSpacePromise;
	}
}
