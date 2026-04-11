import test from "tape";
import { LocalEmbeddings } from "./LocalEmbeddings.js";

type TestTokenizer = ((
	documents: string[],
	options: { max_length: number },
) => { documents: string[]; options: { max_length: number } }) & {
	model_max_length: number;
};

type TestModel = ((inputs: {
	documents: string[];
	options: { max_length: number };
}) => Promise<{
	sentence_embedding: { tolist: () => number[][] };
}>) & {
	config: { max_position_embeddings: number };
};

function createRuntime(): { tokenizer: TestTokenizer; model: TestModel } {
	const tokenizer = Object.assign(
		(
			documents: string[],
			options: { max_length: number },
		): { documents: string[]; options: { max_length: number } } => ({
			documents,
			options,
		}),
		{ model_max_length: 16 },
	) as TestTokenizer;

	const model = Object.assign(
		async (inputs: {
			documents: string[];
			options: { max_length: number };
		}): Promise<{ sentence_embedding: { tolist: () => number[][] } }> => ({
			sentence_embedding: {
				tolist: () =>
					inputs.documents.map((document) => [
						document.length,
						inputs.options.max_length,
					]),
			},
		}),
		{ config: { max_position_embeddings: 16 } },
	) as TestModel;

	return { tokenizer, model };
}

test("LocalEmbeddings embeds document arrays", async (assert) => {
	const runtime = createRuntime();
	const embeddings = new LocalEmbeddings(runtime);

	const result = await embeddings.embedDocuments(["one", "three"]);

	assert.deepEqual(result, [[3, 16], [5, 16]], "returns one vector per document");
	assert.end();
});

test("LocalEmbeddings embeds a single query", async (assert) => {
	const runtime = createRuntime();
	const embeddings = new LocalEmbeddings(runtime);

	const result = await embeddings.embedQuery("query");

	assert.deepEqual(result, [5, 16], "returns exactly one vector for query");
	assert.end();
});

test("LocalEmbeddings rejects malformed numeric vectors", async (assert) => {
	const runtime = createRuntime();
	const malformedModel = Object.assign(
		async (inputs: {
			documents: string[];
			options: { max_length: number };
		}): Promise<{ sentence_embedding: { tolist: () => unknown } }> => ({
			sentence_embedding: {
				tolist: () =>
					inputs.documents.map(() => [1, Number.NaN] as unknown as number[]),
			},
		}),
		{ config: { max_position_embeddings: 16 } },
	) as TestModel;

	const embeddings = new LocalEmbeddings({
		tokenizer: runtime.tokenizer,
		model: malformedModel,
	});

	try {
		await embeddings.embedDocuments(["one"]);
		assert.fail("expected embedDocuments to throw for malformed vectors");
	} catch (error) {
		assert.ok(
			error instanceof Error && /non-numeric embedding vector/i.test(error.message),
			"throws when any embedding contains non-finite numeric values",
		);
	}
	assert.end();
});

test("LocalEmbeddings rejects wrong query embedding count", async (assert) => {
	const runtime = createRuntime();
	const wrongCountModel = Object.assign(
		async (): Promise<{ sentence_embedding: { tolist: () => number[][] } }> => ({
			sentence_embedding: {
				tolist: () => [[1, 2], [3, 4]],
			},
		}),
		{ config: { max_position_embeddings: 16 } },
	) as TestModel;

	const embeddings = new LocalEmbeddings({
		tokenizer: runtime.tokenizer,
		model: wrongCountModel,
	});

	try {
		await embeddings.embedQuery("query");
		assert.fail("expected embedQuery to throw for invalid output length");
	} catch (error) {
		assert.ok(
			error instanceof Error && /produced 2 vectors for 1 documents/i.test(error.message),
			"throws when query embedding response length is not exactly one",
		);
	}
	assert.end();
});
