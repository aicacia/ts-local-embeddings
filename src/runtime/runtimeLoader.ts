import type {
	EmbeddingModelFallback,
	RuntimeLoaderArgs,
	RuntimeLoaderPort,
} from "./runtimeLoaderPort.js";
import type { AutoModel, AutoTokenizer } from "@huggingface/transformers";

// Lazy-load heavy `@huggingface/transformers` at runtime to avoid bundling
// large libraries into the main UI bundle. Methods perform dynamic imports
// so the heavy code is only loaded when the runtime is actually used.
export class HuggingFaceRuntimeLoader
	implements
		RuntimeLoaderPort<
			Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
			Awaited<ReturnType<typeof AutoModel.from_pretrained>>
		>
{
	async loadTokenizer(
		modelId: string,
		args: RuntimeLoaderArgs,
	): Promise<Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>> {
		const mod = (await import("@huggingface/transformers")) as unknown;
		const modRec = mod as Record<string, unknown>;

		type PretrainedLoader = {
			from_pretrained: (
				modelId: string,
				options?: Record<string, unknown>,
			) => Promise<unknown>;
		};

		const candidate =
			(modRec.AutoTokenizer as unknown) ??
			((modRec.default as Record<string, unknown> | undefined)
				?.AutoTokenizer as unknown);

		if (!candidate) {
			throw new Error("AutoTokenizer not available from transformers module");
		}

		const loader = candidate as unknown as PretrainedLoader;
		return (await loader.from_pretrained(modelId, {
			local_files_only: args.localFilesOnly,
			...args.pretrainedOptions,
		})) as Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
	}

	async loadModel(
		modelId: string,
		fallback: EmbeddingModelFallback,
		args: RuntimeLoaderArgs,
	): Promise<Awaited<ReturnType<typeof AutoModel.from_pretrained>>> {
		const mod = (await import("@huggingface/transformers")) as unknown;
		const modRec = mod as Record<string, unknown>;

		type PretrainedLoader = {
			from_pretrained: (
				modelId: string,
				options?: Record<string, unknown>,
			) => Promise<unknown>;
		};

		const candidate =
			(modRec.AutoModel as unknown) ??
			((modRec.default as Record<string, unknown> | undefined)
				?.AutoModel as unknown);

		if (!candidate) {
			throw new Error("AutoModel not available from transformers module");
		}

		const loader = candidate as unknown as PretrainedLoader;
		return (await loader.from_pretrained(modelId, {
			...fallback,
			local_files_only: args.localFilesOnly,
			...args.pretrainedOptions,
		})) as Awaited<ReturnType<typeof AutoModel.from_pretrained>>;
	}
}
