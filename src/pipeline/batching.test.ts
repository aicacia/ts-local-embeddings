import test from "tape";
import { estimateDocumentTokenLength, resolveBatchLimits } from "./batching.js";

test("estimateDocumentTokenLength ascii vs multibyte", (t) => {
	const maxInput = 128;
	const ascii = "abcdefgh";
	const mb = "漢字abc";

	const asciiEstimate = estimateDocumentTokenLength(ascii, maxInput);
	const mbEstimate = estimateDocumentTokenLength(mb, maxInput);

	t.ok(asciiEstimate >= 1, "ascii estimate >= 1");
	t.ok(mbEstimate >= 1, "multibyte estimate >= 1");
	t.ok(mbEstimate !== asciiEstimate, "multibyte estimate differs from ascii");
	t.end();
});

test("resolveBatchLimits returns sensible values", (t) => {
	const { targetBatchTokens, maxDocumentsPerBatch } = resolveBatchLimits(64);
	t.ok(
		targetBatchTokens >= 4096 || targetBatchTokens > 0,
		"targetBatchTokens computed",
	);
	t.ok(maxDocumentsPerBatch >= 1, "maxDocumentsPerBatch computed");
	t.end();
});
