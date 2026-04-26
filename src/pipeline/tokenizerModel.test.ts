import test from "tape";
import {
	resolveMaxInputTokens,
	invokeTokenizer,
	invokeModel,
} from "./tokenizerModel.js";

test("resolveMaxInputTokens uses tokenizer and model config", (t) => {
	const tokenizer: any = Object.assign((docs: string[]) => docs, {
		model_max_length: 128,
	});
	const model: any = { config: { max_position_embeddings: 64 } };

	const resolved = resolveMaxInputTokens(tokenizer, model);
	t.equal(resolved, 64, "chooses the smaller of tokenizer and model limits");
	t.end();
});

test("invokeTokenizer and invokeModel support function and _call styles", async (t) => {
	const tokenizerFn = (documents: string[], options: any) => ({
		documents,
		options,
	});
	const tokenizerObj = { _call: tokenizerFn, model_max_length: 16 };

	const modelFn = async (inputs: any) => ({
		sentence_embedding: { tolist: () => [[1, 2]] },
	});
	const modelObj = { _call: modelFn, config: { max_position_embeddings: 16 } };

	const tokOut = invokeTokenizer(tokenizerObj as any, ["a"], {
		max_length: 16,
	});
	t.deepEqual(
		tokOut,
		{ documents: ["a"], options: { max_length: 16 } },
		"tokenizer _call invoked",
	);

	const modelOut = await invokeModel(modelObj as any, {});
	t.ok(
		(modelOut as any).sentence_embedding.tolist,
		"model _call invoked and returns sentence_embedding with tolist",
	);

	t.end();
});
