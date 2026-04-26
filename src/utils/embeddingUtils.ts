// Shared embedding and packing utilities for LocalEmbeddings, WorkerEmbeddings, IndexedDBVectorStore
import { packRowsToFloat32 } from "./typedArrayUtils.js";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";

/**
 * Embed an array of documents using the provided embeddings interface.
 */
export async function embedDocuments(
	embeddings: EmbeddingsInterface<number[] | Float32Array>,
	documents: string[],
): Promise<Array<number[] | Float32Array>> {
	return embeddings.embedDocuments(documents);
}

/**
 * Embed a single query/document using the provided embeddings interface.
 */
export async function embedQuery(
	embeddings: EmbeddingsInterface<number[] | Float32Array>,
	document: string,
): Promise<number[] | Float32Array> {
	return embeddings.embedQuery(document);
}

/**
 * Pack an array of embeddings into a contiguous Float32Array buffer.
 */
export function packEmbeddings(embeddings: ArrayLike<number>[]): {
	buffer: ArrayBuffer;
	rows: number;
	dims: number;
} {
	return packRowsToFloat32(embeddings);
}
