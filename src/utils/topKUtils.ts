// Shared utility for top-K selection using a min-heap
// Used by vector stores and other modules needing efficient top-K
import { MinHeap } from "./heapUtils.js";

export type TopKItem<T> = T & { similarity: number };

/**
 * Returns the top-K items by similarity, using a bounded min-heap.
 * Items must have a `similarity` property (higher is better).
 */
export function selectTopK<T extends { similarity: number }>(
	items: T[],
	k: number,
): T[] {
	if (k <= 0) return [];
	const heap = new MinHeap<T>((a, b) => a.similarity - b.similarity);
	for (const item of items) {
		if (heap.size() < k) {
			heap.push(item);
		} else {
			const top = heap.peek();
			if (top && item.similarity > top.similarity) {
				heap.pop();
				heap.push(item);
			}
		}
	}
	return heap.toArrayDesc();
}
