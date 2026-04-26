const MULTIBYTE_REGEX = /[^\u0000-\u00ff]/;

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

export function resolveBatchLimits(maxInputTokens: number): {
	targetBatchTokens: number;
	maxDocumentsPerBatch: number;
} {
	const MAX_INPUT_TOKENS_FALLBACK = 512;
	const MIN_DOCUMENTS_PER_BATCH_FALLBACK = 32;
	const TARGET_BATCH_TOKENS_FALLBACK = 4096;
	const TARGET_BATCH_TOKENS_MAX = 16384;

	const nav =
		typeof navigator !== "undefined"
			? (navigator as Navigator & { deviceMemory?: number })
			: null;
	const deviceMemory =
		typeof nav?.deviceMemory === "number" && Number.isFinite(nav.deviceMemory)
			? Math.floor(nav.deviceMemory)
			: null;

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
