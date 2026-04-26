import type {
	ModelInstance,
	TokenizerInstance,
} from "../runtime/embeddingRuntime.js";

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

type TokenizerCallable = (
	documents: string[],
	options: TokenizerCallOptions,
) => unknown;

type ModelCallable = (inputs: unknown) => Promise<SentenceEmbeddingResult>;

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
	if (typeof sentenceEmbedding !== "object" || sentenceEmbedding === null)
		return false;

	return (
		typeof (sentenceEmbedding as { tolist?: unknown }).tolist === "function"
	);
}

function resolveCallable<T extends (...args: any[]) => any>(
	value: unknown,
	errorMessage: string,
): T {
	const maybeCallable = value as unknown as T | { _call?: T };

	if (typeof maybeCallable === "function") {
		return maybeCallable;
	}

	if (typeof (maybeCallable as any)._call === "function") {
		return (maybeCallable as any)._call;
	}

	throw new Error(errorMessage);
}

export function resolveTokenizerCall(
	tokenizer: TokenizerInstance,
): TokenizerCallable {
	return resolveCallable<TokenizerCallable>(
		tokenizer,
		"Embedding tokenizer is not callable.",
	);
}

export function resolveModelCall(model: ModelInstance): ModelCallable {
	const modelCall = resolveCallable<(modelInputs: unknown) => Promise<unknown>>(
		model,
		"Embedding model is not callable.",
	);

	return async (inputs: unknown) => {
		const output = await modelCall(inputs);
		if (hasSentenceEmbeddingWithToList(output)) {
			return output as SentenceEmbeddingResult;
		}
		throw new Error(
			"Embedding model output is missing sentence_embedding.tolist().",
		);
	};
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

export function resolveMaxInputTokens(
	tokenizer: TokenizerInstance,
	model: ModelInstance,
): number {
	function asFinitePositiveInteger(value: unknown): number | null {
		if (typeof value !== "number" || !Number.isFinite(value)) return null;
		const parsed = Math.floor(value);
		return parsed > 0 ? parsed : null;
	}

	const tokenizerLimit = asFinitePositiveInteger(
		(tokenizer as any).model_max_length,
	);
	const modelConfig = ((model as any).config ?? {}) as Record<string, unknown>;

	const modelLimitCandidates = [
		asFinitePositiveInteger(modelConfig.max_position_embeddings),
		asFinitePositiveInteger(modelConfig.n_positions),
		asFinitePositiveInteger(modelConfig.max_seq_len),
		asFinitePositiveInteger(modelConfig.seq_length),
	].filter((v): v is number => v !== null);

	const modelLimit =
		modelLimitCandidates.length > 0 ? Math.min(...modelLimitCandidates) : null;

	const resolved = [tokenizerLimit, modelLimit]
		.filter((v): v is number => v !== null)
		.reduce<number | null>(
			(min, value) => (min === null ? value : Math.min(min, value)),
			null,
		);

	const MAX_INPUT_TOKENS_FALLBACK = 512;
	return resolved ?? MAX_INPUT_TOKENS_FALLBACK;
}
