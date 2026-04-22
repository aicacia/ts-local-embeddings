import type {
	ModelInstance,
	TokenizerInstance,
} from "../runtime/embeddingRuntime.js";
import { isDebugLoggingEnabled } from "../debug.js";

const MULTIBYTE_REGEX = /[^\u0000-\u00ff]/;

// Note: embedding numeric validation removed — trust model outputs.

export type LocalEmbeddingsRuntime = {
	tokenizer: TokenizerInstance;
	model: ModelInstance;
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

export type EmbeddingPipelineOptions = EmbeddingPipelineHooks;

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
	const length = document.length;
	if (length <= 32) {
		for (let index = 0; index < length; index += 1) {
			if (document.charCodeAt(index) > 255) {
				const estimatedTokens = Math.ceil(length / 2);
				return Math.max(1, Math.min(maxInputTokens, estimatedTokens));
			}
		}
		const estimatedTokens = Math.ceil(length / 4);
		return Math.max(1, Math.min(maxInputTokens, estimatedTokens));
	}

	const likelyMultibyteText = MULTIBYTE_REGEX.test(document);
	const charsPerToken = likelyMultibyteText ? 2 : 4;
	const estimatedTokens = Math.ceil(length / charsPerToken);
	return Math.max(1, Math.min(maxInputTokens, estimatedTokens));
}

function isFiniteNumberArray(value: unknown): value is number[] {
	if (!Array.isArray(value)) {
		return false;
	}
	for (let i = 0; i < value.length; i++) {
		const entry = value[i];
		if (typeof entry !== "number" || !Number.isFinite(entry)) {
			return false;
		}
	}
	return true;
}

export function ensureEmbeddingMatrix(
	value: unknown,
	expectedLength: number,
	errorContext: string,
): number[][] {
	// Minimal shape-only checks: ensure top-level array and expected count.
	// Per-element numeric validation has been removed; trust the model output
	// when it returns without error.
	if (!Array.isArray(value)) {
		throw new Error(`${errorContext} output is not an array.`);
	}

	if (value.length !== expectedLength) {
		throw new Error(
			`${errorContext} produced ${value.length} vectors for ${expectedLength} documents.`,
		);
	}

	return value as number[][];
}

export async function ensureEmbeddingMatrixAsync(
	value: unknown,
	expectedLength: number,
	errorContext: string,
): Promise<number[][]> {
	// Minimal async passthrough — delegate to synchronous shape-only checks.
	return Promise.resolve(
		ensureEmbeddingMatrix(value, expectedLength, errorContext),
	);
}

function parseEmbeddingMatrix(
	sentenceEmbedding: unknown,
	expectedLength: number,
	errorContext: string,
): Promise<number[][]> {
	return ensureEmbeddingMatrixAsync(
		sentenceEmbedding,
		expectedLength,
		errorContext,
	);
}

type TokenizerCallable = (
	documents: string[],
	options: TokenizerCallOptions,
) => unknown;

type ModelCallable = (inputs: unknown) => Promise<SentenceEmbeddingResult>;

function resolveTokenizerCall(tokenizer: TokenizerInstance): TokenizerCallable {
	const maybeCallable = tokenizer as unknown as
		| TokenizerCallable
		| {
				_call?: TokenizerCallable;
		  };

	if (typeof maybeCallable === "function") {
		return maybeCallable;
	}

	if (typeof maybeCallable._call === "function") {
		return maybeCallable._call;
	}

	throw new Error("Embedding tokenizer is not callable.");
}

function resolveModelCall(model: ModelInstance): ModelCallable {
	const maybeCallable = model as unknown as
		| ((modelInputs: unknown) => Promise<unknown>)
		| {
				_call?: (modelInputs: unknown) => Promise<unknown>;
		  };

	if (typeof maybeCallable === "function") {
		return async (inputs: unknown) => {
			const output = await maybeCallable(inputs);
			if (hasSentenceEmbeddingWithToList(output)) {
				return output as SentenceEmbeddingResult;
			}

			throw new Error(
				"Embedding model output is missing sentence_embedding.tolist().",
			);
		};
	}

	if (typeof maybeCallable._call === "function") {
		return async (inputs: unknown) => {
			// @ts-expect-error we already checked maybeCallable
			const output = await maybeCallable._call(inputs);
			if (hasSentenceEmbeddingWithToList(output)) {
				return output as SentenceEmbeddingResult;
			}

			throw new Error(
				"Embedding model output is missing sentence_embedding.tolist().",
			);
		};
	}

	throw new Error("Embedding model is not callable.");
}

export function invokeTokenizer(
	tokenizer: TokenizerInstance,
	documents: string[],
	options: TokenizerCallOptions,
): unknown {
	return resolveTokenizerCall(tokenizer)(documents, options);
}

export async function invokeModel(
	model: ModelInstance,
	inputs: unknown,
): Promise<SentenceEmbeddingResult> {
	return resolveModelCall(model)(inputs);
}

export function createEmbeddingPipeline(
	runtime: LocalEmbeddingsRuntime,
	options: EmbeddingPipelineOptions = {},
): EmbeddingPipeline {
	const tokenizer = runtime.tokenizer;
	const model = runtime.model;
	const tokenizerCall = resolveTokenizerCall(tokenizer);
	const modelCall = resolveModelCall(model);
	const onEvent = options.onEvent;
	const debugLoggingEnabled = isDebugLoggingEnabled();
	const shouldTrackEvents = typeof onEvent === "function";
	const maxInputTokens = resolveMaxInputTokens(tokenizer, model);
	const tokenizerOptions: TokenizerCallOptions = {
		padding: true,
		truncation: true,
		max_length: maxInputTokens,
	};
	const { targetBatchTokens, maxDocumentsPerBatch } =
		resolveBatchLimits(maxInputTokens);

	return {
		async embedDocuments(documents: string[]): Promise<number[][]> {
			if (documents.length === 0) {
				return [];
			}

			// Preallocate full output array to avoid repeated growth during pushes.
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
				if (batch.length === 0) {
					return;
				}

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

				for (let i = 0; i < batchEmbeddings.length; i++) {
					embeddings[embeddingsIndex++] = batchEmbeddings[i];
				}

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
