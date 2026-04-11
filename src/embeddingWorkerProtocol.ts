import type {
	EmbeddingRuntime,
	LoadEmbeddingRuntimeOptions,
} from "./embeddingRuntime.js";

export type WorkerInitPayload = {
	options?: LoadEmbeddingRuntimeOptions;
};

export type WorkerRequestMap = {
	init: WorkerInitPayload;
	embedDocuments: {
		documents: string[];
	};
	embedQuery: {
		document: string;
	};
};

export type WorkerRequestType = keyof WorkerRequestMap;

export type WorkerRequest = {
	[K in WorkerRequestType]: {
		type: K;
		requestId: number;
		payload: WorkerRequestMap[K];
	};
}[WorkerRequestType];

export type SerializedError = {
	message: string;
	name?: string;
	stack?: string;
};

export type WorkerResponseMap = {
	ready: {
		runtime: Pick<EmbeddingRuntime, "modelId" | "variant">;
	};
	documentsEmbedded: {
		embeddings: number[][];
	};
	queryEmbedded: {
		embedding: number[];
	};
	error: SerializedError;
};

export type WorkerResponseType = keyof WorkerResponseMap;

export type WorkerResponse = {
	[K in WorkerResponseType]: {
		type: K;
		requestId: number;
		payload: WorkerResponseMap[K];
	};
}[WorkerResponseType];
