import test from "tape";
import { MinHeap } from "./heapUtils.ts";

test("MinHeap maintains the heap property and ordering", (t) => {
	const heap = new MinHeap<number>((a, b) => a - b);
	heap.push(5);
	heap.push(3);
	heap.push(8);
	heap.push(1);

	t.equal(heap.pop(), 1, "returns the smallest item first");
	t.equal(heap.pop(), 3, "returns the next smallest item");
	t.equal(heap.pop(), 5, "returns the next smallest item");
	t.equal(heap.pop(), 8, "returns the next smallest item");
	t.equal(heap.pop(), undefined, "returns undefined when empty");

	const heap2 = new MinHeap<number>((a, b) => a - b);
	heap2.push(5);
	heap2.push(3);
	heap2.push(8);
	heap2.push(1);

	t.deepEqual(
		heap2.toArrayDesc(),
		[8, 5, 3, 1],
		"returns values in descending order",
	);
	t.end();
});
