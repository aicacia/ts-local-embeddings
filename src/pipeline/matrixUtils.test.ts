import test from "tape";
import { ensureEmbeddingMatrix } from "./matrixUtils.js";

test("ensureEmbeddingMatrix validates length and throws on mismatch", (t) => {
	const good = [
		[1, 2],
		[3, 4],
	];
	t.deepEqual(
		ensureEmbeddingMatrix(good, 2, "ctx"),
		good,
		"returns matrix when lengths match",
	);

	t.throws(
		() => ensureEmbeddingMatrix(good, 3, "ctx"),
		/produced 2 vectors for 3 documents/,
		"throws when lengths mismatch",
	);
	t.end();
});
