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

	return value as number[][];
}

export async function ensureEmbeddingMatrixAsync(
	value: unknown,
	expectedLength: number,
	errorContext: string,
): Promise<number[][]> {
	return Promise.resolve(
		ensureEmbeddingMatrix(value, expectedLength, errorContext),
	);
}

export function parseEmbeddingMatrix(
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
