import type { Suite, Deferred } from "benchmark";
import type { LocalEmbeddingsRuntime } from "./embeddingPipeline.js";
import {
	createEmbeddingPipeline,
	estimateDocumentTokenLength,
	invokeModel,
	invokeTokenizer,
	resolveBatchLimits,
	resolveMaxInputTokens,
} from "./embeddingPipeline.js";
import type { Constructor } from "../types.js";

type TestRuntime = LocalEmbeddingsRuntime;
type TokenizerOptions = {
	max_length: number;
	padding?: boolean;
	truncation?: boolean;
};

function createRuntime(options?: {
	modelMaxLength?: number;
	tokenizerAsObjectCall?: boolean;
	modelAsObjectCall?: boolean;
}): TestRuntime {
	const modelMaxLength = options?.modelMaxLength ?? 64;

	const tokenizerFn = (
		documents: string[],
		options: { max_length: number },
	) => ({
		documents,
		options,
	});

	const modelFn = async (inputs: {
		documents: string[];
		options: { max_length: number };
	}) => ({
		sentence_embedding: {
			tolist: () =>
				inputs.documents.map((document) => [
					document.length,
					inputs.options.max_length,
				]),
		},
	});

	const tokenizerWithConfig = Object.assign(tokenizerFn, {
		model_max_length: modelMaxLength,
	}) as unknown as LocalEmbeddingsRuntime["tokenizer"];

	const modelWithConfig = Object.assign(modelFn, {
		config: { max_position_embeddings: modelMaxLength },
	}) as LocalEmbeddingsRuntime["model"];

	const tokenizer = options?.tokenizerAsObjectCall
		? ({
				model_max_length: modelMaxLength,
				_call: tokenizerFn,
			} as unknown as LocalEmbeddingsRuntime["tokenizer"])
		: tokenizerWithConfig;

	const model = options?.modelAsObjectCall
		? ({
				config: { max_position_embeddings: modelMaxLength },
				_call: modelFn,
			} as unknown as LocalEmbeddingsRuntime["model"])
		: modelWithConfig;

	return { tokenizer, model };
}

export default function register(Suite: Constructor<Suite>) {
	const runtime = createRuntime({ modelMaxLength: 64 });
	const pipeline = createEmbeddingPipeline(runtime);
	const pipelineWithoutValidation = createEmbeddingPipeline(runtime);
	const documents = Array.from(
		{ length: 256 },
		(_, index) => `document ${index} ${"x".repeat(32)}`,
	);
	const documents512 = Array.from(
		{ length: 512 },
		(_, index) => `document ${index} ${"x".repeat(32)}`,
	);
	const asciiDocument = "hello world";
	const multibyteDocument = "你好世界";
	const longAsciiDocument = "a".repeat(2048);
	const tokenizerOptions: TokenizerOptions = {
		max_length: 64,
		padding: false,
		truncation: true,
	};

	return new Promise<void>((resolve) => {
		new Suite()
			.add("resolveMaxInputTokens", () => {
				resolveMaxInputTokens(runtime.tokenizer, runtime.model);
			})
			.add("resolveBatchLimits/64", () => {
				resolveBatchLimits(64);
			})
			.add("resolveBatchLimits/128", () => {
				resolveBatchLimits(128);
			})
			.add("estimateDocumentTokenLength/ascii", () => {
				estimateDocumentTokenLength(asciiDocument, 64);
			})
			.add("estimateDocumentTokenLength/multibyte", () => {
				estimateDocumentTokenLength(multibyteDocument, 64);
			})
			.add("estimateDocumentTokenLength/long-ascii", () => {
				estimateDocumentTokenLength(longAsciiDocument, 64);
			})
			.add("invokeTokenizer/function-call", () => {
				invokeTokenizer(runtime.tokenizer, documents, tokenizerOptions);
			})
			.add("invokeTokenizer/object-_call", () => {
				invokeTokenizer(
					{
						model_max_length: 64,
						_call: (documents: string[], options: { max_length: number }) => ({
							documents,
							options,
						}),
					} as unknown as LocalEmbeddingsRuntime["tokenizer"],
					documents,
					tokenizerOptions,
				);
			})
			.add("invokeModel/function-call", {
				defer: true,
				fn: async (deferred: { resolve: () => void }) => {
					await invokeModel(runtime.model, {
						documents,
						options: { max_length: 64 },
					});
					deferred.resolve();
				},
			})
			.add("pipeline/embedDocuments", {
				defer: true,
				fn: async (deferred: { resolve: () => void }) => {
					await pipeline.embedDocuments(documents);
					deferred.resolve();
				},
			})
			.add("pipeline/embedDocuments/no-validation", {
				defer: true,
				fn: async (deferred: { resolve: () => void }) => {
					await pipelineWithoutValidation.embedDocuments(documents);
					deferred.resolve();
				},
			})
			.add("pipeline/embedDocuments/512", {
				defer: true,
				fn: async (deferred: { resolve: () => void }) => {
					await pipeline.embedDocuments(documents512);
					deferred.resolve();
				},
			})
			.on("cycle", (event: Event) => {
				console.log(String(event.target));
			})
			.on("complete", () => {
				resolve();
			})
			.run({ async: true });
	});
}
