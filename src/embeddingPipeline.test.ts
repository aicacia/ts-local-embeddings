import test from "tape";
import {
	createEmbeddingPipeline,
	estimateDocumentTokenLength,
	type EmbeddingPipelineEvent,
} from "./embeddingPipeline.js";

type TestTokenizer = (documents: string[], options: { max_length: number }) => {
	documents: string[];
	options: { max_length: number };
};

type TestModel = (inputs: {
	documents: string[];
	options: { max_length: number };
}) => Promise<{ sentence_embedding: { tolist: () => unknown } }>;

function createRuntime(options?: {
	modelMaxLength?: number;
	tokenizerAsObjectCall?: boolean;
	modelAsObjectCall?: boolean;
	modelOutputFactory?: (inputs: {
		documents: string[];
		options: { max_length: number };
	}) => unknown;
}): {
	tokenizer: unknown;
	model: unknown;
} {
	const modelMaxLength = options?.modelMaxLength ?? 16;
	const modelOutputFactory =
		options?.modelOutputFactory ??
		((inputs: { documents: string[]; options: { max_length: number } }) =>
			inputs.documents.map((document) => [
				document.length,
				inputs.options.max_length,
			]));

	const tokenizerFn: TestTokenizer = (
		documents: string[],
		options: { max_length: number },
	) => ({ documents, options });
	const tokenizerWithConfig = Object.assign(tokenizerFn, {
		model_max_length: modelMaxLength,
	});

	const modelFn: TestModel = async (inputs: {
		documents: string[];
		options: { max_length: number };
	}) => ({
		sentence_embedding: {
			tolist: () => modelOutputFactory(inputs),
		},
	});
	const modelWithConfig = Object.assign(modelFn, {
		config: { max_position_embeddings: modelMaxLength },
	});

	const tokenizer = options?.tokenizerAsObjectCall
		? {
				model_max_length: modelMaxLength,
				_call: tokenizerFn,
		  }
		: tokenizerWithConfig;
	const model = options?.modelAsObjectCall
		? {
				config: { max_position_embeddings: modelMaxLength },
				_call: modelFn,
		  }
		: modelWithConfig;

	return { tokenizer, model };
}

test("embeddingPipeline returns empty output for empty input", async (assert) => {
	const runtime = createRuntime();
	const pipeline = createEmbeddingPipeline(runtime);

	const result = await pipeline.embedDocuments([]);

	assert.deepEqual(result, [], "returns [] without touching tokenizer/model");
	assert.end();
});

test("embeddingPipeline splits documents across batches at limits", async (assert) => {
	const runtime = createRuntime({ modelMaxLength: 5000 });
	const events: EmbeddingPipelineEvent[] = [];
	const pipeline = createEmbeddingPipeline(runtime, {
		onEvent: (event) => {
			events.push(event);
		},
	});
	const longDocument = "x".repeat(20_000);

	const result = await pipeline.embedDocuments([
		longDocument,
		longDocument,
		longDocument,
		longDocument,
		longDocument,
	]);

	assert.equal(result.length, 5, "returns one embedding per input document");
	assert.deepEqual(
		events.map((event) => event.batchDocuments),
		[3, 2],
		"flushes batch when token budget would be exceeded",
	);
	assert.end();
});

test("embeddingPipeline handles callable and _call runtimes", async (assert) => {
	const functionRuntime = createRuntime();
	const objectRuntime = createRuntime({
		tokenizerAsObjectCall: true,
		modelAsObjectCall: true,
	});

	const functionPipeline = createEmbeddingPipeline(functionRuntime);
	const objectPipeline = createEmbeddingPipeline(objectRuntime);

	const functionResult = await functionPipeline.embedDocuments(["one", "two"]);
	const objectResult = await objectPipeline.embedDocuments(["one", "two"]);

	assert.deepEqual(functionResult, objectResult, "supports both invocation styles");
	assert.end();
});

test("embeddingPipeline rejects malformed embedding output", async (assert) => {
	const runtime = createRuntime({
		modelOutputFactory: (inputs) =>
			inputs.documents.map(() => [1, Number.NaN] as unknown as number[]),
	});
	const pipeline = createEmbeddingPipeline(runtime);

	try {
		await pipeline.embedDocuments(["one"]);
		assert.fail("expected embedDocuments to throw");
	} catch (error) {
		assert.ok(
			error instanceof Error &&
				/non-numeric embedding vector/i.test(error.message),
			"throws for non-finite vectors",
		);
	}
	assert.end();
});

test("embeddingPipeline query path requires exactly one embedding", async (assert) => {
	const runtime = createRuntime({
		modelOutputFactory: () => [
			[1, 2],
			[3, 4],
		],
	});
	const pipeline = createEmbeddingPipeline(runtime);

	try {
		await pipeline.embedQuery("query");
		assert.fail("expected embedQuery to throw");
	} catch (error) {
		assert.ok(
			error instanceof Error && /produced 2 vectors for 1 documents/i.test(error.message),
			"throws when query path receives more than one vector",
		);
	}
	assert.end();
});

test("embeddingPipeline multibyte estimation is stricter than ascii", (assert) => {
	const maxInputTokens = 128;
	const asciiEstimate = estimateDocumentTokenLength("abcdefgh", maxInputTokens);
	const multibyteEstimate = estimateDocumentTokenLength("你好世界", maxInputTokens);

	assert.equal(asciiEstimate, 2, "ascii estimate uses ~4 chars/token");
	assert.equal(multibyteEstimate, 2, "multibyte estimate uses ~2 chars/token");
	assert.end();
});
