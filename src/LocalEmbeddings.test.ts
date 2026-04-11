import test from "tape";
import { LocalEmbeddings } from "./LocalEmbeddings.js";

test("LocalEmbeddings delegates to runtime pipeline", async (assert) => {
	let tokenizerCalls = 0;
	let modelCalls = 0;
	const tokenizer = Object.assign(
		(documents: string[], options: { max_length: number }) => {
			tokenizerCalls += 1;
			return { documents, options };
		},
		{ model_max_length: 16 },
	);
	const model = Object.assign(
		async (inputs: {
			documents: string[];
			options: { max_length: number };
		}) => {
			modelCalls += 1;
			return {
				sentence_embedding: {
					tolist: () =>
						inputs.documents.map((document) => [
							document.length,
							inputs.options.max_length,
						]),
				},
			};
		},
		{ config: { max_position_embeddings: 16 } },
	);
	const embeddings = new LocalEmbeddings({ tokenizer, model });

	const docs = await embeddings.embedDocuments(["one", "three"]);
	const query = await embeddings.embedQuery("query");

	assert.deepEqual(
		docs,
		[
			[3, 16],
			[5, 16],
		],
		"embedDocuments delegates correctly",
	);
	assert.deepEqual(query, [5, 16], "embedQuery delegates correctly");
	assert.equal(tokenizerCalls, 2, "tokenizer invoked once per API call");
	assert.equal(modelCalls, 2, "model invoked once per API call");
	assert.end();
});
