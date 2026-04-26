/* eslint-disable @typescript-eslint/no-explicit-any */
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import {
	createEmbeddingPipeline,
	type EmbeddingPipeline,
	type EmbeddingPipelineHooks,
	type LocalEmbeddingsRuntime,
} from "./createEmbeddingPipeline.js";
import {
	embedDocuments,
	embedQuery,
	packEmbeddings,
} from "../utils/embeddingUtils.js";

export type { LocalEmbeddingsRuntime } from "./createEmbeddingPipeline.js";
export type {
	EmbeddingPipelineEvent,
	EmbeddingPipelineHooks,
} from "./createEmbeddingPipeline.js";

/**
 * LocalEmbeddings provides local embedding generation using a runtime pipeline.
 * Implements EmbeddingsInterface for compatibility with LangChain.
 */
export class LocalEmbeddings
	implements EmbeddingsInterface<number[] | Float32Array>
{
	#pipeline: EmbeddingPipeline;

	/**
	 * Create a LocalEmbeddings instance.
	 * @param runtime - The runtime configuration for the embedding pipeline.
	 * @param hooks - Optional hooks for pipeline events.
	 */
	constructor(
		runtime: LocalEmbeddingsRuntime,
		hooks: EmbeddingPipelineHooks = {},
	) {
		this.#pipeline = createEmbeddingPipeline(runtime, hooks);
	}

	/**
	 * Embed an array of documents.
	 * @param documents - Array of strings to embed.
	 * @returns Array of embeddings (number[] or Float32Array).
	 */
	async embedDocuments(
		documents: string[],
	): Promise<Array<number[] | Float32Array>> {
		return embedDocuments(this.#pipeline, documents);
	}

	/**
	 * Return a packed ArrayBuffer with embeddings laid out row-major (rows * dims).
	 * This can be transferred or used to create typed-array views without additional allocations.
	 * @param documents - Array of strings to embed.
	 * @returns Object with buffer, rows, and dims.
	 */
	async embedDocumentsRaw(
		documents: string[],
	): Promise<{ buffer: ArrayBuffer; rows: number; dims: number }> {
		const result = await embedDocuments(this.#pipeline, documents);
		if (!Array.isArray(result) || result.length === 0) {
			return { buffer: new ArrayBuffer(0), rows: 0, dims: 0 };
		}
		const packed = packEmbeddings(result as ArrayLike<number>[]);
		return { buffer: packed.buffer, rows: packed.rows, dims: packed.dims };
	}

	/**
	 * Embed a single query/document.
	 * @param document - String to embed.
	 * @returns Embedding (number[] or Float32Array).
	 */
	async embedQuery(document: string): Promise<number[] | Float32Array> {
		return embedQuery(this.#pipeline, document);
	}
}
