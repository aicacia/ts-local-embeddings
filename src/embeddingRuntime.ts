import type { AutoModel, AutoTokenizer } from "@huggingface/transformers";
import { HuggingFaceRuntimeLoader } from "./runtimeLoader.js";
import type {
	EmbeddingModelFallback,
	LoadEmbeddingRuntimeOptions,
	RuntimeLoaderPort,
} from "./runtimeLoaderPort.js";
import {
	loadModelWithFallbacks,
	resolveRuntimePolicy,
} from "./runtimePolicy.js";

export type TokenizerInstance = Awaited<
	ReturnType<typeof AutoTokenizer.from_pretrained>
>;
export type ModelInstance = Awaited<
	ReturnType<typeof AutoModel.from_pretrained>
>;

export type { EmbeddingModelFallback } from "./runtimeLoaderPort.js";

export type EmbeddingRuntime = {
	modelId: string;
	variant: string;
	tokenizer: TokenizerInstance;
	model: ModelInstance;
};

export type { LoadEmbeddingRuntimeOptions } from "./runtimeLoaderPort.js";

export async function loadEmbeddingRuntime(
	options: LoadEmbeddingRuntimeOptions = {},
	loader: RuntimeLoaderPort = new HuggingFaceRuntimeLoader(),
): Promise<EmbeddingRuntime> {
	const policy = resolveRuntimePolicy(options);

	const tokenizerPromise = loader.loadTokenizer(
		policy.modelId,
		policy.loaderArgs,
	);
	const [{ model, variant }, tokenizer] = await Promise.all([
		loadModelWithFallbacks(
			policy.modelId,
			policy.modelFallbacks,
			loader,
			policy.loaderArgs,
		),
		tokenizerPromise,
	]);

	return {
		modelId: policy.modelId,
		variant,
		tokenizer: tokenizer as TokenizerInstance,
		model: model as ModelInstance,
	};
}
