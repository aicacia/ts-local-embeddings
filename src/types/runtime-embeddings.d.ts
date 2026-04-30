export interface RuntimeEmbeddings {
  embedDocumentsRaw?: (documents: unknown) => Promise<EmbeddingsResult> | EmbeddingsResult
  embedQueryRaw?: (query: unknown) => Promise<EmbeddingsResult> | EmbeddingsResult
}

export type EmbeddingsResult = number[] | Float32Array | ArrayBuffer

export interface EmbeddingMessage {
  embeddings?: EmbeddingsResult[]
  embeddingsBuffer?: ArrayBuffer | ArrayBufferLike | { buffer: ArrayBuffer }
  embedding?: EmbeddingsResult
  embeddingBuffer?: ArrayBuffer | ArrayBufferLike | { buffer: ArrayBuffer }
}

export type SerializedRecord = {
  id: string
  embedding: EmbeddingsResult | number[]
  metadata?: Record<string, unknown>
}

export interface NodeProcessLike {
  versions?: { node?: string }
}

export {}
