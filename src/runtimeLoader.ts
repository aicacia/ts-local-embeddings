import { AutoModel, AutoTokenizer } from "@huggingface/transformers";
import type {
	EmbeddingModelFallback,
	RuntimeLoaderArgs,
	RuntimeLoaderPort,
} from "./runtimeLoaderPort.js";

export class HuggingFaceRuntimeLoader implements RuntimeLoaderPort {
	loadTokenizer(modelId: string, args: RuntimeLoaderArgs): Promise<unknown> {
		return AutoTokenizer.from_pretrained(modelId, {
			local_files_only: args.localFilesOnly,
			...args.pretrainedOptions,
		});
	}

	loadModel(
		modelId: string,
		fallback: EmbeddingModelFallback,
		args: RuntimeLoaderArgs,
	): Promise<unknown> {
		return AutoModel.from_pretrained(modelId, {
			...fallback,
			local_files_only: args.localFilesOnly,
			...args.pretrainedOptions,
		});
	}
}