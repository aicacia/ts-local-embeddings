import test from "tape";
import { loadEmbeddingRuntime } from "./embeddingRuntime.js";
import type {
	EmbeddingModelFallback,
	RuntimeLoaderPort,
} from "./runtimeLoaderPort.js";

function createLoader(options?: {
	delayMs?: number;
	failUntilIndex?: number;
}): RuntimeLoaderPort {
	let modelCall = 0;

	return {
		loadTokenizer: async () => {
			if (options?.delayMs) {
				await new Promise((resolve) => setTimeout(resolve, options.delayMs));
			}
			return { tokenizer: true };
		},
		loadModel: async (_modelId, fallback: EmbeddingModelFallback) => {
			modelCall += 1;
			if (options?.delayMs) {
				await new Promise((resolve) => setTimeout(resolve, options.delayMs));
			}

			const failUntilIndex = options?.failUntilIndex ?? 0;
			if (modelCall <= failUntilIndex) {
				throw new Error(`failed ${fallback.dtype}`);
			}
			return { model: fallback.dtype };
		},
	};
}

test("loadEmbeddingRuntime supports injected loader and fallback recovery", async (assert) => {
	const runtime = await loadEmbeddingRuntime(
		{
			modelId: "demo-model",
			modelFallbacks: [{ dtype: "q4" }, { dtype: "q8" }],
		},
		createLoader({ failUntilIndex: 1 }),
	);

	assert.equal(runtime.modelId, "demo-model", "keeps resolved model id");
	assert.equal(runtime.variant, "q8", "recovers to the next fallback");
	assert.deepEqual(
		runtime.model,
		{ model: "q8" },
		"uses model from injected loader",
	);
	assert.deepEqual(
		runtime.tokenizer,
		{ tokenizer: true },
		"uses tokenizer from injected loader",
	);
	assert.end();
});

test("loadEmbeddingRuntime initializes model and tokenizer in parallel", async (assert) => {
	const startedAt = Date.now();
	await loadEmbeddingRuntime(
		{
			modelFallbacks: [{ dtype: "q4" }],
		},
		createLoader({ delayMs: 25 }),
	);
	const elapsedMs = Date.now() - startedAt;

	assert.ok(
		elapsedMs < 45,
		`loads model and tokenizer concurrently (${elapsedMs}ms)`,
	);
	assert.end();
});
