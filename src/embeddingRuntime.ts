import { AutoModel, AutoTokenizer } from "@huggingface/transformers";

export type TokenizerInstance = Awaited<
	ReturnType<typeof AutoTokenizer.from_pretrained>
>;
export type ModelInstance = Awaited<
	ReturnType<typeof AutoModel.from_pretrained>
>;

export type EmbeddingModelFallback = {
	dtype: "q4" | "q8" | "fp16" | "fp32";
	model_file_name?: string;
};

export type EmbeddingRuntime = {
	modelId: string;
	variant: string;
	tokenizer: TokenizerInstance;
	model: ModelInstance;
};

export type LoadEmbeddingRuntimeOptions = {
	modelId?: string;
	modelFallbacks?: readonly EmbeddingModelFallback[];
	allowRemoteModels?: boolean;
	modelPath?: string;
};

const DEFAULT_MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const DEFAULT_MODEL_FALLBACKS: readonly EmbeddingModelFallback[] = [
	{ dtype: "q4", model_file_name: "model_no_gather" },
	{ dtype: "q4" },
	{ dtype: "q8" },
	{ dtype: "fp16" },
	{ dtype: "fp32" },
];

type DebugGlobal = typeof globalThis & {
	__LOCAL_EMBEDDINGS_DEBUG__?: boolean;
};

function isDebugLoggingEnabled(): boolean {
	return Boolean(
		(globalThis as DebugGlobal).__LOCAL_EMBEDDINGS_DEBUG__ === true,
	);
}

function describeVariant(fallback: EmbeddingModelFallback): string {
	return fallback.model_file_name
		? `${fallback.model_file_name}_${fallback.dtype}`
		: fallback.dtype;
}

async function loadModelWithFallbacks(
	modelId: string,
	modelFallbacks: readonly EmbeddingModelFallback[],
	localFilesOnly: boolean,
	pretrainedOptions: Record<string, unknown>,
): Promise<{ model: ModelInstance; variant: string }> {
	let lastError: unknown;

	for (const fallback of modelFallbacks) {
		const variant = describeVariant(fallback);

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
			const model = await AutoModel.from_pretrained(modelId, {
				...fallback,
				local_files_only: localFilesOnly,
				...pretrainedOptions,
			});
			if (isDebugLoggingEnabled()) {
				console.debug(
					"[EmbeddingRuntime] Embedding model variant initialized successfully.",
					{
						modelId,
						variant,
					},
				);
			}
			return { model, variant };
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
		`Failed to initialize embedding model for ${modelId}. Tried variants: ${modelFallbacks.map(describeVariant).join(", ")}`,
		{ cause: lastError },
	);
}

export async function loadEmbeddingRuntime(
	options: LoadEmbeddingRuntimeOptions = {},
): Promise<EmbeddingRuntime> {
	const modelId = options.modelId ?? DEFAULT_MODEL_ID;
	// Keep remote enabled by default: local snapshot may omit tokenizer/config metadata.
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

	const tokenizerPromise = AutoTokenizer.from_pretrained(modelId, {
		local_files_only: localFilesOnly,
		...pretrainedOptions,
	});
	const [{ model, variant }, tokenizer] = await Promise.all([
		loadModelWithFallbacks(
			modelId,
			modelFallbacks,
			localFilesOnly,
			pretrainedOptions,
		),
		tokenizerPromise,
	]);

	return {
		modelId,
		variant,
		tokenizer,
		model,
	};
}
