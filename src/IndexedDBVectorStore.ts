import { Document, type DocumentInterface } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import {
	cosineSimilarity,
	maximalMarginalRelevance,
} from "@langchain/core/utils/math";
import {
	createVectorWritePipeline,
	type StoredVectorRecord,
	type VectorWritePipeline,
} from "./vectorWritePipeline.js";
import { IndexedDbStoreGateway } from "./indexedDbStoreGateway.js";

export type IndexedDBVectorStoreFilter = (doc: Document) => boolean;

export type IndexedDBVectorStoreArgs = {
	dbName?: string;
	storeName?: string;
	similarity?: (a: number[], b: number[]) => number;
};

const DEFAULT_DB_NAME = "langchain-indexeddb-vectorstore";
const DEFAULT_STORE_NAME = "vectors";

function cosine(a: number[], b: number[]): number {
	const similarityMatrix = cosineSimilarity([a], [b]);
	const score = similarityMatrix[0]?.[0];
	return Number.isFinite(score) ? score : 0;
}

function createDocument(record: StoredVectorRecord): Document {
	return new Document({
		id: record.id,
		pageContent: record.content,
		metadata: record.metadata,
	});
}

type EmbeddingProvenanceSource = {
	getEmbeddingProvenance?: () => Promise<string>;
};

const DEFAULT_EMBEDDING_SPACE = "default";

export class IndexedDBVectorStore {
	readonly #embeddings: EmbeddingsInterface<number[]>;
	readonly #dbName: string;
	readonly #storeName: string;
	readonly #similarity: (a: number[], b: number[]) => number;
	readonly #writePipeline: VectorWritePipeline;
	readonly #gateway: IndexedDbStoreGateway;
	#embeddingSpacePromise: Promise<string> | null = null;

	constructor(
		embeddings: EmbeddingsInterface<number[]>,
		args: IndexedDBVectorStoreArgs = {},
	) {
		this.#embeddings = embeddings;
		this.#dbName = args.dbName ?? DEFAULT_DB_NAME;
		this.#storeName = args.storeName ?? DEFAULT_STORE_NAME;
		this.#similarity = args.similarity ?? cosine;
		this.#gateway = new IndexedDbStoreGateway({
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
	): Promise<Document[]> {
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
		await store.#gateway.open();
		return store;
	}

	async #getRecordsByContentHash(
		embeddingSpace: string,
		contentHashes: string[],
		contents: string[],
	): Promise<Array<StoredVectorRecord | null>> {
		return this.#gateway.queryByContentHash(
			contentHashes,
			contents,
			(record, index) =>
				(index < 0 || record.content === contents[index]) &&
				this.#matchesEmbeddingSpace(record, embeddingSpace),
		);
	}

	async #getAllRecords(): Promise<StoredVectorRecord[]> {
		return this.#gateway.getAll();
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
		if (k <= 0) {
			return [];
		}

		const topMatches: Array<{
			similarity: number;
			metadata: Record<string, unknown>;
			content: string;
			embedding: number[];
			id: string;
		}> = [];

		const maybePushMatch = (
			match: {
				similarity: number;
				metadata: Record<string, unknown>;
				content: string;
				embedding: number[];
				id: string;
			},
		) => {
			if (topMatches.length < k) {
				topMatches.push(match);
				topMatches.sort((left, right) => right.similarity - left.similarity);
				return;
			}

			const lowest = topMatches[topMatches.length - 1];
			if (match.similarity <= lowest.similarity) {
				return;
			}

			topMatches[topMatches.length - 1] = match;
			topMatches.sort((left, right) => right.similarity - left.similarity);
		};

		await this.#gateway.iterateAll<StoredVectorRecord>(async (record) => {
			if (filter) {
				const doc = new Document({
					metadata: record.metadata,
					pageContent: record.content,
					id: record.id,
				});
				if (!filter(doc)) {
					return;
				}
			}

			maybePushMatch({
				similarity: this.#similarity(query, record.embedding),
				metadata: record.metadata,
				content: record.content,
				embedding: record.embedding,
				id: record.id,
			});
		});

		return topMatches;
	}

	#matchesEmbeddingSpace(
		record: StoredVectorRecord,
		embeddingSpace: string,
	): boolean {
		const recordSpace = record.embeddingSpace ?? DEFAULT_EMBEDDING_SPACE;
		return recordSpace === embeddingSpace;
	}

	#resolveEmbeddingSpace(): Promise<string> {
		if (this.#embeddingSpacePromise) {
			return this.#embeddingSpacePromise;
		}

		this.#embeddingSpacePromise = (async () => {
			const source = this.#embeddings as EmbeddingProvenanceSource;
			if (typeof source.getEmbeddingProvenance === "function") {
				const provenance = await source.getEmbeddingProvenance();
				if (typeof provenance === "string" && provenance.length > 0) {
					return provenance;
				}
			}

			return DEFAULT_EMBEDDING_SPACE;
		})();

		return this.#embeddingSpacePromise;
	}
}

