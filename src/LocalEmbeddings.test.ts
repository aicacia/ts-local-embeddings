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
