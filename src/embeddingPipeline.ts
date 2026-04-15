import type { ModelInstance, TokenizerInstance } from "./embeddingRuntime.js";
import { isDebugLoggingEnabled } from "./debug.js";

export type LocalEmbeddingsRuntime = {
	tokenizer: TokenizerInstance | Promise<TokenizerInstance>;
	model: ModelInstance | Promise<ModelInstance>;
};

const MAX_INPUT_TOKENS_FALLBACK = 512;
const MIN_DOCUMENTS_PER_BATCH_FALLBACK = 32;
const TARGET_BATCH_TOKENS_FALLBACK = 4096;
const TARGET_BATCH_TOKENS_MAX = 16384;

type TokenizerCallOptions = {
	padding?: boolean;
	truncation?: boolean;
	max_length: number;
};

type SentenceEmbeddingResult = {
	sentence_embedding: {
		tolist: () => unknown;
	};
};

type BatchDecisionEvent = {
	type: "batch";
	batchNumber: number;
	batchDocuments: number;
	batchTokens: number;
	processedAfterBatch: number;
	totalDocuments: number;
};

export type EmbeddingPipelineEvent = BatchDecisionEvent;

export type EmbeddingPipelineHooks = {
	onEvent?: (event: EmbeddingPipelineEvent) => void;
};

export type EmbeddingPipeline = {
	embedDocuments(documents: string[]): Promise<number[][]>;
	embedQuery(document: string): Promise<number[]>;
};

function hasSentenceEmbeddingWithToList(
	value: unknown,
): value is SentenceEmbeddingResult {
	if (
		typeof value !== "object" ||
		value === null ||
		!("sentence_embedding" in value)
	) {
		return false;
	}

	const sentenceEmbedding = (value as { sentence_embedding?: unknown })
		.sentence_embedding;
	if (typeof sentenceEmbedding !== "object" || sentenceEmbedding === null) {
		return false;
	}

	return (
		typeof (sentenceEmbedding as { tolist?: unknown }).tolist === "function"
	);
}

function asFinitePositiveInteger(value: unknown): number | null {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return null;
	}

	const parsed = Math.floor(value);
	return parsed > 0 ? parsed : null;
}

export function resolveMaxInputTokens(
	tokenizer: TokenizerInstance,
	model: ModelInstance,
): number {
	const tokenizerLimit = asFinitePositiveInteger(
		(tokenizer as { model_max_length?: unknown }).model_max_length,
	);
	const modelConfig =
		(
			model as unknown as {
				config?: {
					max_position_embeddings?: unknown;
					n_positions?: unknown;
					max_seq_len?: unknown;
					seq_length?: unknown;
				};
			}
		).config ?? {};

	const modelLimitCandidates = [
		asFinitePositiveInteger(modelConfig.max_position_embeddings),
		asFinitePositiveInteger(modelConfig.n_positions),
		asFinitePositiveInteger(modelConfig.max_seq_len),
		asFinitePositiveInteger(modelConfig.seq_length),
	].filter((value): value is number => value !== null);

	const modelLimit =
		modelLimitCandidates.length > 0 ? Math.min(...modelLimitCandidates) : null;

	const resolved = [tokenizerLimit, modelLimit]
		.filter((value): value is number => value !== null)
		.reduce<number | null>(
			(min, value) => (min === null ? value : Math.min(min, value)),
			null,
		);

	return resolved ?? MAX_INPUT_TOKENS_FALLBACK;
}

export function resolveBatchLimits(maxInputTokens: number): {
	targetBatchTokens: number;
	maxDocumentsPerBatch: number;
} {
	// `navigator.deviceMemory` is browser-only. When unavailable, fall back to
	// conservative batch limits rather than assuming large device memory.
	const nav =
		typeof navigator !== "undefined"
			? (navigator as Navigator & { deviceMemory?: number })
			: null;
	const deviceMemory = asFinitePositiveInteger(nav?.deviceMemory);

	const tokenMultiplier =
		deviceMemory === null
			? 8
			: deviceMemory >= 16
				? 12
				: deviceMemory >= 8
					? 10
					: deviceMemory >= 4
						? 8
						: 6;

	const computedTargetBatchTokens = Math.min(
		TARGET_BATCH_TOKENS_MAX,
		Math.max(TARGET_BATCH_TOKENS_FALLBACK, maxInputTokens * tokenMultiplier),
	);

	const computedMaxDocumentsPerBatch = Math.max(
		4,
		Math.min(
			64,
			Math.floor(computedTargetBatchTokens / Math.max(1, maxInputTokens)),
		),
	);

	const maxDocumentsPerBatch = Number.isFinite(computedMaxDocumentsPerBatch)
		? computedMaxDocumentsPerBatch
		: MIN_DOCUMENTS_PER_BATCH_FALLBACK;

	return {
		targetBatchTokens: computedTargetBatchTokens,
		maxDocumentsPerBatch,
	};
}

export function estimateDocumentTokenLength(
	document: string,
	maxInputTokens: number,
): number {
	// Estimate token count heuristically. Multibyte text often consumes more tokens,
	// so use a conservative character-per-token ratio to avoid overfilling batches.
	const likelyMultibyteText = Array.from(document).some(
		(char) => char.charCodeAt(0) > 255,
	);
	const charsPerToken = likelyMultibyteText ? 2 : 4;
	const estimatedTokens = Math.ceil(document.length / charsPerToken);
	return Math.max(1, Math.min(maxInputTokens, estimatedTokens));
}

function isFiniteNumberArray(value: unknown): value is number[] {
	return (
		Array.isArray(value) &&
		value.every((entry) => typeof entry === "number" && Number.isFinite(entry))
	);
}

export function ensureEmbeddingMatrix(
	value: unknown,
	expectedLength: number,
	errorContext: string,
): number[][] {
	if (!Array.isArray(value)) {
		throw new Error(`${errorContext} output is not an array.`);
	}

	if (value.length !== expectedLength) {
		throw new Error(
			`${errorContext} produced ${value.length} vectors for ${expectedLength} documents.`,
		);
	}

	for (const vector of value) {
		if (!isFiniteNumberArray(vector)) {
			throw new Error(
				`${errorContext} output contains a non-numeric embedding vector.`,
			);
		}
	}

	return value;
}

export function invokeTokenizer(
	tokenizer: TokenizerInstance,
	documents: string[],
	options: TokenizerCallOptions,
): unknown {
	const maybeCallable = tokenizer as unknown as
		| ((documents: string[], options: TokenizerCallOptions) => unknown)
		| {
				_call?: (documents: string[], options: TokenizerCallOptions) => unknown;
		  };

	if (typeof maybeCallable === "function") {
		return maybeCallable(documents, options);
	}

	if (typeof maybeCallable._call === "function") {
		return maybeCallable._call(documents, options);
	}

	throw new Error("Embedding tokenizer is not callable.");
}

export async function invokeModel(
	model: ModelInstance,
	inputs: unknown,
): Promise<SentenceEmbeddingResult> {
	const maybeCallable = model as unknown as
		| ((modelInputs: unknown) => Promise<unknown>)
		| {
				_call?: (modelInputs: unknown) => Promise<unknown>;
		  };

	let output: unknown;
	if (typeof maybeCallable === "function") {
		output = await maybeCallable(inputs);
	} else if (typeof maybeCallable._call === "function") {
		output = await maybeCallable._call(inputs);
	} else {
		throw new Error("Embedding model is not callable.");
	}

	if (hasSentenceEmbeddingWithToList(output)) {
		return output as SentenceEmbeddingResult;
	}

	throw new Error(
		"Embedding model output is missing sentence_embedding.tolist().",
	);
}

export function createEmbeddingPipeline(
	runtime: LocalEmbeddingsRuntime,
	hooks: EmbeddingPipelineHooks = {},
): EmbeddingPipeline {
	const tokenizerPromise = Promise.resolve(runtime.tokenizer);
	const modelPromise = Promise.resolve(runtime.model);

	return {
		async embedDocuments(documents: string[]): Promise<number[][]> {
			if (documents.length === 0) {
				return [];
			}

			const [tokenizer, model] = await Promise.all([
				tokenizerPromise,
				modelPromise,
			]);
			const maxInputTokens = resolveMaxInputTokens(tokenizer, model);
			const { targetBatchTokens, maxDocumentsPerBatch } =
				resolveBatchLimits(maxInputTokens);
			const embeddings: number[][] = [];
			let batch: string[] = [];
			let batchTokens = 0;
			let batchesProcessed = 0;
			let documentsProcessed = 0;

			if (isDebugLoggingEnabled()) {
				console.debug("[LocalEmbeddings] Starting document embedding.", {
					totalDocuments: documents.length,
					maxInputTokens,
					targetBatchTokens,
					maxDocumentsPerBatch,
				});
			}

			const flushBatch = async (): Promise<void> => {
				if (batch.length === 0) {
					return;
				}

				const currentBatchSize = batch.length;
				const currentBatchTokens = batchTokens;
				const nextBatchNumber = batchesProcessed + 1;
				const projectedDocumentsProcessed =
					documentsProcessed + currentBatchSize;

				hooks.onEvent?.({
					type: "batch",
					batchNumber: nextBatchNumber,
					batchDocuments: currentBatchSize,
					batchTokens: currentBatchTokens,
					processedAfterBatch: projectedDocumentsProcessed,
					totalDocuments: documents.length,
				});

				if (isDebugLoggingEnabled()) {
					console.debug("[LocalEmbeddings] Running embedding batch.", {
						batchNumber: nextBatchNumber,
						batchDocuments: currentBatchSize,
						batchTokens: currentBatchTokens,
						processedAfterBatch: projectedDocumentsProcessed,
						totalDocuments: documents.length,
						progressPercent: Number(
							((projectedDocumentsProcessed / documents.length) * 100).toFixed(
								1,
							),
						),
					});
				}

				const inputs = invokeTokenizer(tokenizer, batch, {
					padding: true,
					truncation: true,
					max_length: maxInputTokens,
				});

				const { sentence_embedding } = await invokeModel(model, inputs);
				const batchEmbeddings = ensureEmbeddingMatrix(
					sentence_embedding.tolist(),
					currentBatchSize,
					"Embedding batch",
				);

				embeddings.push(...batchEmbeddings);
				batchesProcessed += 1;
				documentsProcessed = projectedDocumentsProcessed;
				batch = [];
				batchTokens = 0;
			};

			for (const document of documents) {
				const documentTokens = estimateDocumentTokenLength(
					document,
					maxInputTokens,
				);
				const wouldExceedBatch =
					batch.length > 0 &&
					(batch.length >= maxDocumentsPerBatch ||
						batchTokens + documentTokens > targetBatchTokens);

				if (wouldExceedBatch) {
					await flushBatch();
				}

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

			return embeddings;
		},

		async embedQuery(document: string): Promise<number[]> {
			const [tokenizer, model] = await Promise.all([
				tokenizerPromise,
				modelPromise,
			]);
			const maxInputTokens = resolveMaxInputTokens(tokenizer, model);

			const inputs = invokeTokenizer(tokenizer, [document], {
				padding: true,
				truncation: true,
				max_length: maxInputTokens,
			});

			const { sentence_embedding } = await invokeModel(model, inputs);

			const embeddings = ensureEmbeddingMatrix(
				sentence_embedding.tolist(),
				1,
				"Query embedding",
			);

			return embeddings[0];
		},
	};
}
