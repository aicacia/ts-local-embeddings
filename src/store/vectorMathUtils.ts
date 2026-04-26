// Vector math utilities for vector stores
import {
	cosineSimilarity,
	cosineSimilarityWithQueryNorm,
} from "./mathUtils.js";

/**
 * Compute the norm of a vector.
 */
export function computeVectorNorm(vec: ArrayLike<number>): number {
	let sum = 0;
	for (let i = 0; i < vec.length; i++) sum += (vec[i] as number) ** 2;
	return Math.sqrt(sum);
}

/**
 * Compute similarity between two vectors.
 */
export function computeSimilarity(
	a: ArrayLike<number>,
	b: ArrayLike<number>,
	normA?: number,
	normB?: number,
	useOptimizedCosine = false,
): number {
	if (
		useOptimizedCosine &&
		typeof normA === "number" &&
		typeof normB === "number"
	) {
		if (normA === 0 || normB === 0) return 0;
		let dot = 0;
		for (let i = 0; i < a.length; i++)
			dot += (a[i] as number) * (b[i] as number);
		const score = dot / (normA * normB);
		return Number.isFinite(score) ? score : 0;
	}
	return useOptimizedCosine
		? cosineSimilarityWithQueryNorm(a, normA ?? 0, b)
		: cosineSimilarity(a, b);
}
