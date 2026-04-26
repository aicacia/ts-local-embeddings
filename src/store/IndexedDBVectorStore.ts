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
	type StorageGatewayPort,
	VECTOR_STORE_SCHEMA,
} from "./indexedDbStoreGateway.js";
import { computeVectorNorm } from "./utils.js";
import { MinHeap } from "../utils/heapUtils.js";
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

function createDocument(record: StoredVectorRecord): Document {
	return new Document({
		id: record.id,
		pageContent: record.content,
		metadata: record.metadata,
	});
}

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

export class IndexedDBVectorStore {
	readonly #embeddings: EmbeddingsInterface<number[] | Float32Array>;
	readonly #dbName: string;
	readonly #storeName: string;
	readonly #similarity: (a: ArrayLike<number>, b: ArrayLike<number>) => number;
	readonly #writePipeline: VectorWritePipeline;
	readonly #gateway: StorageGatewayPort;
	#embeddingSpacePromise: Promise<string> | null = null;
	#getAllThreshold = 10000;

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

	async addDocuments(documents: Document[]): Promise<void> {
		await this.#writePipeline.addDocuments(documents);
	}

	async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
		await this.#writePipeline.addVectors(vectors, documents);
	}

	async #putRecords(records: StoredVectorRecord[]): Promise<void> {
		await this.#gateway.put(records);
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
		query: ArrayLike<number>,
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
			} as unknown as StoredVectorRecord),
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
	): Promise<Document[]> {
		const queryEmbedding = await this.#embeddings.embedQuery(query);
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
		await this.#gateway.clear();
	}

	async count(): Promise<number> {
		return this.#gateway.count();
	}

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
		if (k <= 0) {
			return [];
		}

		const heap = new MinHeap<Match>((a, b) => a.similarity - b.similarity);

		const useOptimizedCosine = this.#similarity === cosineSimilarity;
		const queryNorm = useOptimizedCosine ? computeVectorNorm(query) : 0;
		const computeSimilarity = (
			embedding: ArrayLike<number>,
			recordNorm?: number,
		): number => {
			if (useOptimizedCosine && typeof recordNorm === "number") {
				if (recordNorm === 0 || queryNorm === 0) {
					return 0;
				}

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

		const processRecord = (record: StoredVectorRecord): void => {
			if (filter) {
				const doc = new Document({
					metadata: record.metadata,
					pageContent: record.content,
					id: record.id,
				});
				if (!filter(doc)) return;
			}

			const embedding = record.embedding as ArrayLike<number>;
			let similarity: number;
			const recordNorm = (record as any).embeddingNorm as number | undefined;
			similarity = computeSimilarity(embedding, recordNorm);

			if (heap.size() < k) {
				heap.push({
					similarity,
					metadata: record.metadata,
					content: record.content,
					embedding: record.embedding as unknown as ArrayLike<number>,
					id: record.id,
				});
				return;
			}

			const top = heap.peek();
			if (top && similarity > top.similarity) {
				heap.pop();
				heap.push({
					similarity,
					metadata: record.metadata,
					content: record.content,
					embedding: record.embedding as unknown as ArrayLike<number>,
					id: record.id,
				});
			}
		};

		// Decide whether to fetch all records or stream via cursor. Streaming
		// avoids materializing the entire DB into memory when the store is large.
		let useStream = false;
		try {
			const total = await this.#gateway.count();
			useStream = total > this.#getAllThreshold;
		} catch (_err) {
			// If counting fails, fall back to streaming for safety.
			useStream = true;
		}

		if (useStream) {
			await this.#gateway.iterateAll<StoredVectorRecord>(async (record) => {
				processRecord(record);
				return true;
			});
			return heap.toArrayDesc();
		}

		// Small store: materialize all records for faster in-memory iteration.
		let all: StoredVectorRecord[];
		try {
			all = await this.#gateway.getAll();
		} catch (_) {
			await this.#gateway.iterateAll<StoredVectorRecord>(async (record) => {
				processRecord(record);
				return true;
			});
			return heap.toArrayDesc();
		}

		for (const record of all) processRecord(record);
		return heap.toArrayDesc();
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
