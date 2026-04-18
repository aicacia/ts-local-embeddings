import type {
	EmbeddingModelFallback,
	LoadEmbeddingRuntimeOptions,
	RuntimeLoaderArgs,
	RuntimeLoaderPort,
} from "./runtimeLoaderPort.js";
import { isDebugLoggingEnabled } from "../debug.js";

const DEFAULT_MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const DEFAULT_MODEL_FALLBACKS: readonly EmbeddingModelFallback[] = [
	{ dtype: "q4", model_file_name: "model_no_gather" },
	{ dtype: "q4" },
	{ dtype: "q8" },
	{ dtype: "fp16" },
	{ dtype: "fp32" },
];

export function describeVariant(fallback: EmbeddingModelFallback): string {
	return fallback.model_file_name
		? `${fallback.model_file_name}_${fallback.dtype}`
		: fallback.dtype;
}

export type ResolvedRuntimePolicy = {
	modelId: string;
	modelFallbacks: readonly EmbeddingModelFallback[];
	loaderArgs: RuntimeLoaderArgs;
};

export function resolveRuntimePolicy(
	options: LoadEmbeddingRuntimeOptions = {},
): ResolvedRuntimePolicy {
	const modelId = options.modelId ?? DEFAULT_MODEL_ID;
	const allowRemoteModels = options.allowRemoteModels ?? true;
	const localFilesOnly = !allowRemoteModels;
	const modelFallbacks =
		options.modelFallbacks && options.modelFallbacks.length > 0
			? options.modelFallbacks
			: DEFAULT_MODEL_FALLBACKS;
	const pretrainedOptions: Record<string, unknown> =
		typeof options.modelPath === "string" && options.modelPath.length > 0
			? { cache_dir: options.modelPath }
			: {};

	return {
		modelId,
		modelFallbacks,
		loaderArgs: {
			localFilesOnly,
			pretrainedOptions,
		},
	};
}

export async function loadModelWithFallbacks(
	modelId: string,
	modelFallbacks: readonly EmbeddingModelFallback[],
	loader: RuntimeLoaderPort,
	args: RuntimeLoaderArgs,
): Promise<{ model: unknown; variant: string; attemptedVariants: string[] }> {
	let lastError: unknown;
	const attemptedVariants: string[] = [];

	for (const fallback of modelFallbacks) {
		const variant = describeVariant(fallback);
		attemptedVariants.push(variant);

		if (isDebugLoggingEnabled()) {
			console.debug(
				"[EmbeddingRuntime] Attempting to initialize embedding model variant.",
				{
					modelId,
					variant,
				},
			);
		}

		try {
			const model = await loader.loadModel(modelId, fallback, args);
			if (isDebugLoggingEnabled()) {
				console.debug(
					"[EmbeddingRuntime] Embedding model variant initialized successfully.",
					{
						modelId,
						variant,
					},
				);
			}
			return { model, variant, attemptedVariants };
		} catch (error) {
			lastError = error;
			if (isDebugLoggingEnabled()) {
				console.warn(
					`Failed to initialize embedding model variant=${variant}, trying next fallback.`,
					error,
				);
			}
		}
	}

	throw new Error(
		`Failed to initialize embedding model for ${modelId}. Tried variants: ${attemptedVariants.join(", ")}`,
		{ cause: lastError },
	);
}
