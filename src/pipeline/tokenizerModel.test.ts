import test from "tape";
import type {
	ModelInstance,
	TokenizerInstance,
} from "../runtime/embeddingRuntime.js";
import {
	invokeModel,
	invokeTokenizer,
	resolveMaxInputTokens,
} from "./tokenizerModel.js";

test("resolveMaxInputTokens uses tokenizer and model config", (t) => {
	const tokenizer = Object.assign((docs: string[]) => docs, {
		model_max_length: 128,
	}) as unknown as TokenizerInstance;
	const model = {
		config: { max_position_embeddings: 64 },
	} as unknown as ModelInstance;

	const resolved = resolveMaxInputTokens(tokenizer, model);
	t.equal(resolved, 64, "chooses the smaller of tokenizer and model limits");
	t.end();
});

test("invokeTokenizer and invokeModel support function and _call styles", async (t) => {
	const tokenizerFn = (documents: string[], options: unknown) => ({
		documents,
		options,
	});
	const tokenizerObj = {
		_call: tokenizerFn,
		model_max_length: 16,
	} as unknown as TokenizerInstance;

	const modelFn = async (_inputs: unknown) => ({
		sentence_embedding: { tolist: () => [[1, 2]] },
	});
	const modelObj = {
		_call: modelFn,
		config: { max_position_embeddings: 16 },
	} as ModelInstance;

	const tokOut = invokeTokenizer(tokenizerObj, ["a"], {
		max_length: 16,
	});
	t.deepEqual(
		tokOut,
		{ documents: ["a"], options: { max_length: 16 } },
		"tokenizer _call invoked",
	);

	const modelOut = await invokeModel(modelObj, {});
	t.ok(
		modelOut.sentence_embedding.tolist,
		"model _call invoked and returns sentence_embedding with tolist",
	);

	t.end();
});
