export type EmbeddingModelFallback = {
	dtype: "q4" | "q8" | "fp16" | "fp32";
	model_file_name?: string;
};

export type LoadEmbeddingRuntimeOptions = {
	modelId?: string;
	modelFallbacks?: readonly EmbeddingModelFallback[];
	allowRemoteModels?: boolean;
	modelPath?: string;
	debugLogging?: boolean;
};

export type RuntimeLoaderArgs = {
	localFilesOnly: boolean;
	pretrainedOptions: Record<string, unknown>;
};

export interface RuntimeLoaderPort {
	loadTokenizer(modelId: string, args: RuntimeLoaderArgs): Promise<unknown>;
	loadModel(
		modelId: string,
		fallback: EmbeddingModelFallback,
		args: RuntimeLoaderArgs,
	): Promise<unknown>;
}
