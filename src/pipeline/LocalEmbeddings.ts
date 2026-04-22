/* eslint-disable @typescript-eslint/no-explicit-any */
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import {
	createEmbeddingPipeline,
	type EmbeddingPipeline,
	type EmbeddingPipelineHooks,
	type LocalEmbeddingsRuntime,
} from "./embeddingPipeline.js";
import { packRowsToFloat32 } from "../utils/typedArrayUtils.js";

export type { LocalEmbeddingsRuntime } from "./embeddingPipeline.js";
export type {
	EmbeddingPipelineEvent,
	EmbeddingPipelineHooks,
} from "./embeddingPipeline.js";

export class LocalEmbeddings
	implements EmbeddingsInterface<number[] | Float32Array>
{
	#pipeline: EmbeddingPipeline;

	constructor(
		runtime: LocalEmbeddingsRuntime,
		hooks: EmbeddingPipelineHooks = {},
	) {
		this.#pipeline = createEmbeddingPipeline(runtime, hooks);
	}

	async embedDocuments(
		documents: string[],
	): Promise<Array<number[] | Float32Array>> {
		return this.#pipeline.embedDocuments(documents);
	}

	/**
	 * Return a packed `ArrayBuffer` with embeddings laid out row-major
	 * (rows * dims). This can be transferred or used to create typed-array
	 * views without additional allocations.
	 */
	async embedDocumentsRaw(
		documents: string[],
	): Promise<{ buffer: ArrayBuffer; rows: number; dims: number }> {
		const result = await this.#pipeline.embedDocuments(documents);

		if (!Array.isArray(result) || result.length === 0) {
			return { buffer: new ArrayBuffer(0), rows: 0, dims: 0 };
		}

		const packed = packRowsToFloat32(result as ArrayLike<number>[]);
		return { buffer: packed.buffer, rows: packed.rows, dims: packed.dims };
	}

	async embedQuery(document: string): Promise<number[] | Float32Array> {
		return this.#pipeline.embedQuery(document);
	}
}
