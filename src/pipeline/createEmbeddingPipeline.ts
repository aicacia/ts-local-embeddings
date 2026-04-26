import type {
	ModelInstance,
	TokenizerInstance,
} from "../runtime/embeddingRuntime.js";
import { isDebugLoggingEnabled } from "../debug.js";
import { resolveTokenizerCall, resolveModelCall } from "./tokenizerModel.js";
import { resolveBatchLimits, estimateDocumentTokenLength } from "./batching.js";
import { parseEmbeddingMatrix } from "./matrixUtils.js";

type TokenizerCallOptions = {
	padding?: boolean;
	truncation?: boolean;
	max_length: number;
};

export type LocalEmbeddingsRuntime = {
	tokenizer: TokenizerInstance;
	model: ModelInstance;
};

export type EmbeddingPipelineEvent = {
	type: "batch";
	batchNumber: number;
	batchDocuments: number;
	batchTokens: number;
	processedAfterBatch: number;
	totalDocuments: number;
};

export type EmbeddingPipelineHooks = {
	onEvent?: (event: EmbeddingPipelineEvent) => void;
};

export type EmbeddingPipeline = {
	embedDocuments(documents: string[]): Promise<number[][]>;
	embedQuery(document: string): Promise<number[]>;
};

export function createEmbeddingPipeline(
	runtime: LocalEmbeddingsRuntime,
	options: EmbeddingPipelineHooks = {},
): EmbeddingPipeline {
	const tokenizer = runtime.tokenizer;
	const model = runtime.model;
	const tokenizerCall = resolveTokenizerCall(tokenizer);
	const modelCall = resolveModelCall(model);
	const onEvent = options.onEvent;
	const debugLoggingEnabled = isDebugLoggingEnabled();
	const shouldTrackEvents = typeof onEvent === "function";
	const maxInputTokens = (tokenizer as any).model_max_length ?? 512;
	const tokenizerOptions: TokenizerCallOptions = {
		padding: true,
		truncation: true,
		max_length: maxInputTokens,
	};
	const { targetBatchTokens, maxDocumentsPerBatch } =
		resolveBatchLimits(maxInputTokens);

	return {
		async embedDocuments(documents: string[]): Promise<number[][]> {
			if (documents.length === 0) return [];

			const embeddings: number[][] = new Array(documents.length);
			let embeddingsIndex = 0;
			let batch: string[] = [];
			let batchTokens = 0;
			let batchesProcessed = 0;
			let documentsProcessed = 0;

			if (debugLoggingEnabled) {
				console.debug("[LocalEmbeddings] Starting document embedding.", {
					totalDocuments: documents.length,
					maxInputTokens,
					targetBatchTokens,
					maxDocumentsPerBatch,
				});
			}

			const flushBatch = async (): Promise<void> => {
				if (batch.length === 0) return;

				const currentBatchSize = batch.length;
				const currentBatchTokens = batchTokens;
				const nextBatchNumber = batchesProcessed + 1;
				const projectedDocumentsProcessed =
					documentsProcessed + currentBatchSize;

				if (shouldTrackEvents) {
					onEvent?.({
						type: "batch",
						batchNumber: nextBatchNumber,
						batchDocuments: currentBatchSize,
						batchTokens: currentBatchTokens,
						processedAfterBatch: projectedDocumentsProcessed,
						totalDocuments: documents.length,
					});
				}

				if (debugLoggingEnabled) {
					console.debug("[LocalEmbeddings] Running embedding batch.", {
						batchNumber: nextBatchNumber,
						batchDocuments: currentBatchSize,
						batchTokens: currentBatchTokens,
						processedAfterBatch: projectedDocumentsProcessed,
						totalDocuments: documents.length,
						progressPercent:
							Math.round(
								(projectedDocumentsProcessed / documents.length) * 100 * 10,
							) / 10,
					});
				}

				const inputs = tokenizerCall(batch, tokenizerOptions);

				const { sentence_embedding } = await modelCall(inputs);
				const batchEmbeddings = await parseEmbeddingMatrix(
					sentence_embedding.tolist(),
					currentBatchSize,
					"Embedding batch",
				);

				for (let i = 0; i < batchEmbeddings.length; i++)
					embeddings[embeddingsIndex++] = batchEmbeddings[i];

				batchesProcessed += 1;
				documentsProcessed = projectedDocumentsProcessed;
				batch = [];
				batchTokens = 0;
			};

			for (let di = 0; di < documents.length; di++) {
				const document = documents[di];
				const documentTokens = estimateDocumentTokenLength(
					document,
					maxInputTokens,
				);
				const wouldExceedBatch =
					batch.length > 0 &&
					(batch.length >= maxDocumentsPerBatch ||
						batchTokens + documentTokens > targetBatchTokens);

				if (wouldExceedBatch) await flushBatch();

				batch.push(document);
				batchTokens += documentTokens;
			}

			await flushBatch();

			if (isDebugLoggingEnabled()) {
				console.debug("[LocalEmbeddings] Completed document embedding.", {
					totalDocuments: documents.length,
					batchesProcessed,
					embeddingsProduced: embeddings.length,
				});
			}

			return embeddings.slice(0, embeddingsIndex);
		},

		async embedQuery(document: string): Promise<number[]> {
			const inputs = tokenizerCall([document], tokenizerOptions);
			const { sentence_embedding } = await modelCall(inputs);
			const embeddings = await parseEmbeddingMatrix(
				sentence_embedding.tolist(),
				1,
				"Query embedding",
			);
			return embeddings[0];
		},
	};
}
