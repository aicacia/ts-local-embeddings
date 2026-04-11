import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import {
	createEmbeddingPipeline,
	type EmbeddingPipeline,
	type EmbeddingPipelineHooks,
	type LocalEmbeddingsRuntime,
} from "./embeddingPipeline.js";

export type { LocalEmbeddingsRuntime } from "./embeddingPipeline.js";
export type { EmbeddingPipelineEvent, EmbeddingPipelineHooks } from "./embeddingPipeline.js";

export class LocalEmbeddings implements EmbeddingsInterface<number[]> {
	#pipeline: EmbeddingPipeline;

	constructor(runtime: LocalEmbeddingsRuntime, hooks: EmbeddingPipelineHooks = {}) {
		this.#pipeline = createEmbeddingPipeline(runtime, hooks);
	}

	async embedDocuments(documents: string[]): Promise<number[][]> {
		return this.#pipeline.embedDocuments(documents);
	}

	async embedQuery(document: string): Promise<number[]> {
		return this.#pipeline.embedQuery(document);
	}
}
