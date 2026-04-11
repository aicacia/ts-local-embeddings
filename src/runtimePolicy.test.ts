import test from "tape";
import {
	loadModelWithFallbacks,
	resolveRuntimePolicy,
} from "./runtimePolicy.js";
import type {
	EmbeddingModelFallback,
	RuntimeLoaderPort,
} from "./runtimeLoaderPort.js";

test("resolveRuntimePolicy uses defaults when options are empty", (assert) => {
	const resolved = resolveRuntimePolicy();

	assert.equal(
		resolved.modelId,
		"onnx-community/embeddinggemma-300m-ONNX",
		"defaults model id",
	);
	assert.equal(resolved.loaderArgs.localFilesOnly, false, "remote models enabled by default");
	assert.ok(resolved.modelFallbacks.length > 0, "defaults model fallback list");
	assert.end();
});

test("resolveRuntimePolicy propagates local_files_only and cache_dir", (assert) => {
	const resolved = resolveRuntimePolicy({
		allowRemoteModels: false,
		modelPath: "/tmp/models",
		modelFallbacks: [{ dtype: "q8" }],
	});

	assert.equal(resolved.loaderArgs.localFilesOnly, true, "disables remote files");
	assert.equal(
		resolved.loaderArgs.pretrainedOptions.cache_dir,
		"/tmp/models",
		"propagates cache directory",
	);
	assert.deepEqual(
		resolved.modelFallbacks,
		[{ dtype: "q8" }],
		"uses explicit fallback list",
	);
	assert.end();
});

test("loadModelWithFallbacks follows fallback order and includes tried variants on failure", async (assert) => {
	const fallbackOrder: readonly EmbeddingModelFallback[] = [
		{ dtype: "q4", model_file_name: "model_no_gather" },
		{ dtype: "q8" },
	];
	const attempted: string[] = [];
	const failingLoader: RuntimeLoaderPort = {
		loadTokenizer: async () => ({}),
		loadModel: async (_modelId, fallback) => {
			attempted.push(fallback.model_file_name ?? fallback.dtype);
			throw new Error(`failed ${fallback.dtype}`);
		},
	};

	try {
		await loadModelWithFallbacks("model-x", fallbackOrder, failingLoader, {
			localFilesOnly: true,
			pretrainedOptions: {},
		});
		assert.fail("expected fallback exhaustion to throw");
	} catch (error) {
		assert.deepEqual(
			attempted,
			["model_no_gather", "q8"],
			"attempts variants in deterministic order",
		);
		assert.ok(
			error instanceof Error &&
				/Tried variants: model_no_gather_q4, q8/i.test(error.message),
			"error contains attempted variant list",
		);
	}
	assert.end();
});
