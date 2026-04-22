export function computeVectorNorm(vector: ArrayLike<number>): number {
	let norm = 0;
	for (let i = 0; i < vector.length; i += 1) {
		const v = vector[i] ?? 0;
		norm += v * v;
	}
	return Math.sqrt(norm);
}
