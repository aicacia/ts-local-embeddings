export function assertEqualLengthVectors(
	a: ArrayLike<number>,
	b: ArrayLike<number>,
): void {
	if (a.length !== b.length) {
		throw new Error("Embedding vectors must have equal length.");
	}
}

export function calculateDotAndNorms(
	a: ArrayLike<number>,
	b: ArrayLike<number>,
): { dot: number; normA: number; normB: number } {
	let dot = 0;
	let normA = 0;
	let normB = 0;
	for (let i = 0; i < a.length; i += 1) {
		const ai = a[i];
		const bi = b[i];
		dot += ai * bi;
		normA += ai * ai;
		normB += bi * bi;
	}
	return { dot, normA, normB };
}

export function cosineSimilarity(
	a: ArrayLike<number>,
	b: ArrayLike<number>,
): number {
	assertEqualLengthVectors(a, b);
	if (a.length === 0) return 0;

	const { dot, normA, normB } = calculateDotAndNorms(a, b);
	if (normA === 0 || normB === 0) return 0;

	const score = dot / (Math.sqrt(normA) * Math.sqrt(normB));
	return Number.isFinite(score) ? score : 0;
}

export function cosineSimilarityWithQueryNorm(
	query: ArrayLike<number>,
	queryNorm: number,
	vector: ArrayLike<number>,
): number {
	assertEqualLengthVectors(query, vector);
	if (queryNorm === 0) return 0;

	const { dot, normB } = calculateDotAndNorms(query, vector);
	if (normB === 0) return 0;

	const score = dot / (queryNorm * Math.sqrt(normB));
	return Number.isFinite(score) ? score : 0;
}
