import type {
	EmbeddingRuntime,
	LoadEmbeddingRuntimeOptions,
} from "../runtime/embeddingRuntime.js";
import type { EmbeddingPipelineEvent } from "../pipeline/embeddingPipeline.js";

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
	code?: string;
	cause?: Array<{
		message: string;
		name?: string;
		code?: string;
	}>;
};

export type WorkerResponseMap = {
	ready: {
		runtime: Pick<EmbeddingRuntime, "modelId" | "variant">;
	};
	progress: {
		requestType: "embedDocuments";
		event: EmbeddingPipelineEvent;
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

export type WorkerSuccessResponse = Exclude<WorkerResponse, { type: "error" }>;

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function isFiniteRequestId(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}

function isWorkerInitPayload(
	value: unknown,
): value is WorkerRequestMap["init"] {
	return isObject(value);
}

function isEmbedDocumentsPayload(
	value: unknown,
): value is WorkerRequestMap["embedDocuments"] {
	return (
		isObject(value) &&
		Array.isArray(value.documents) &&
		value.documents.every((document) => typeof document === "string")
	);
}

function isEmbedQueryPayload(
	value: unknown,
): value is WorkerRequestMap["embedQuery"] {
	return isObject(value) && typeof value.document === "string";
}

function isSerializedErrorCause(value: unknown): boolean {
	if (!isObject(value) || typeof value.message !== "string") {
		return false;
	}

	if ("name" in value && typeof value.name !== "string") {
		return false;
	}

	if ("code" in value && typeof value.code !== "string") {
		return false;
	}

	return true;
}

export function isSerializedError(value: unknown): value is SerializedError {
	if (!isObject(value) || typeof value.message !== "string") {
		return false;
	}

	if ("name" in value && typeof value.name !== "string") {
		return false;
	}

	if ("stack" in value && typeof value.stack !== "string") {
		return false;
	}

	if ("code" in value && typeof value.code !== "string") {
		return false;
	}

	if (
		"cause" in value &&
		(!Array.isArray(value.cause) ||
			!value.cause.every((cause) => isSerializedErrorCause(cause)))
	) {
		return false;
	}

	return true;
}

export function isWorkerRequest(value: unknown): value is WorkerRequest {
	if (!isObject(value)) {
		return false;
	}

	if (!isFiniteRequestId(value.requestId) || typeof value.type !== "string") {
		return false;
	}

	switch (value.type) {
		case "init":
			return isWorkerInitPayload(value.payload);
		case "embedDocuments":
			return isEmbedDocumentsPayload(value.payload);
		case "embedQuery":
			return isEmbedQueryPayload(value.payload);
		default:
			return false;
	}
}

function isReadyPayload(value: unknown): value is WorkerResponseMap["ready"] {
	return (
		isObject(value) &&
		isObject(value.runtime) &&
		typeof value.runtime.modelId === "string" &&
		typeof value.runtime.variant === "string"
	);
}

function isProgressPayload(
	value: unknown,
): value is WorkerResponseMap["progress"] {
	if (!isObject(value) || value.requestType !== "embedDocuments") {
		return false;
	}

	if (!isObject(value.event) || value.event.type !== "batch") {
		return false;
	}

	const event = value.event as Record<string, unknown>;
	return (
		typeof event.batchNumber === "number" &&
		Number.isFinite(event.batchNumber) &&
		typeof event.batchDocuments === "number" &&
		Number.isFinite(event.batchDocuments) &&
		typeof event.batchTokens === "number" &&
		Number.isFinite(event.batchTokens) &&
		typeof event.processedAfterBatch === "number" &&
		Number.isFinite(event.processedAfterBatch) &&
		typeof event.totalDocuments === "number" &&
		Number.isFinite(event.totalDocuments)
	);
}

function isDocumentsEmbeddedPayload(
	value: unknown,
): value is WorkerResponseMap["documentsEmbedded"] {
	return (
		isObject(value) &&
		Array.isArray(value.embeddings) &&
		value.embeddings.every(
			(vector) =>
				Array.isArray(vector) &&
				vector.every(
					(entry) => typeof entry === "number" && Number.isFinite(entry),
				),
		)
	);
}

function isQueryEmbeddedPayload(
	value: unknown,
): value is WorkerResponseMap["queryEmbedded"] {
	return (
		isObject(value) &&
		Array.isArray(value.embedding) &&
		value.embedding.every(
			(entry) => typeof entry === "number" && Number.isFinite(entry),
		)
	);
}

export function isWorkerResponse(value: unknown): value is WorkerResponse {
	if (!isObject(value)) {
		return false;
	}

	if (!isFiniteRequestId(value.requestId) || typeof value.type !== "string") {
		return false;
	}

	switch (value.type) {
		case "ready":
			return isReadyPayload(value.payload);
		case "progress":
			return isProgressPayload(value.payload);
		case "documentsEmbedded":
			return isDocumentsEmbeddedPayload(value.payload);
		case "queryEmbedded":
			return isQueryEmbeddedPayload(value.payload);
		case "error":
			return isSerializedError(value.payload);
		default:
			return false;
	}
}
