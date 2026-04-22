import type { LoadEmbeddingRuntimeOptions } from "../runtime/embeddingRuntime.js";
import {
	DEFAULT_MODEL_FALLBACKS,
	DEFAULT_MODEL_ID,
} from "../runtime/runtimePolicy.js";

export function normalizeLoadEmbeddingRuntimeOptions(
	options: LoadEmbeddingRuntimeOptions | undefined,
): LoadEmbeddingRuntimeOptions {
	const modelPath =
		typeof options?.modelPath === "string" && options.modelPath.length > 0
			? options.modelPath
			: undefined;

	return {
		modelId: options?.modelId ?? DEFAULT_MODEL_ID,
		allowRemoteModels: options?.allowRemoteModels ?? true,
		modelFallbacks:
			options?.modelFallbacks && options.modelFallbacks.length > 0
				? options.modelFallbacks
				: DEFAULT_MODEL_FALLBACKS,
		modelPath,
		debugLogging: options?.debugLogging === true,
	};
}

export function areLoadEmbeddingRuntimeOptionsEqual(
	a: LoadEmbeddingRuntimeOptions | undefined,
	b: LoadEmbeddingRuntimeOptions | undefined,
): boolean {
	const left = normalizeLoadEmbeddingRuntimeOptions(a);
	const right = normalizeLoadEmbeddingRuntimeOptions(b);

	if (left.modelId !== right.modelId) {
		return false;
	}

	if (left.allowRemoteModels !== right.allowRemoteModels) {
		return false;
	}

	if (left.modelPath !== right.modelPath) {
		return false;
	}

	if (left.debugLogging !== right.debugLogging) {
		return false;
	}

	const leftFallbacks = left.modelFallbacks ?? [];
	const rightFallbacks = right.modelFallbacks ?? [];

	if (leftFallbacks.length !== rightFallbacks.length) {
		return false;
	}

	for (let index = 0; index < leftFallbacks.length; index += 1) {
		const leftFallback = leftFallbacks[index];
		const rightFallback = rightFallbacks[index];
		if (leftFallback.dtype !== rightFallback.dtype) {
			return false;
		}
		if (leftFallback.model_file_name !== rightFallback.model_file_name) {
			return false;
		}
	}

	return true;
}
