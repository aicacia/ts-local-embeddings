/* eslint-disable @typescript-eslint/no-explicit-any */
import type {
	EmbeddingRuntime,
	LoadEmbeddingRuntimeOptions,
} from "../runtime/embeddingRuntime.js";
import type { EmbeddingPipelineEvent } from "../pipeline/createEmbeddingPipeline.js";

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
		// Either a nested number array (legacy) or a transferable Float32Array
		// buffer with shape metadata for efficient transfer from worker -> main.
		embeddings?: number[][];
		embeddingsBuffer?: {
			buffer: ArrayBuffer;
			rows: number;
			dims: number;
		};
	};
	queryEmbedded: {
		embedding?: number[];
		embeddingBuffer?: {
			buffer: ArrayBuffer;
			dims: number;
		};
	};
	error: SerializedError;
};

export type WorkerResponseType = keyof WorkerResponseMap;

export type WorkerResponse = {
	[K in WorkerResponseType]: {
		type: K;
		requestId: number;
		payload: WorkerResponseMap[K];
		/** @internal Helper for worker to provide transferable ArrayBuffers */
		_transfer?: ArrayBuffer[];
	};
}[WorkerResponseType];

export type WorkerSuccessResponse = Exclude<WorkerResponse, { type: "error" }>;

/**
 * WorkerResponse with an optional internal `_transfer` helper for
 * attaching transferable ArrayBuffer(s) when posting from the worker.
 *
 * This is intended for worker-internal use; the extra field is optional
 * and does not change the existing `WorkerResponse` shape.
 *
 * @internal
 */
export type WorkerResponseWithTransfer = WorkerResponse & {
	_transfer?: ArrayBuffer[];
};

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function isString(value: unknown): value is string {
	return typeof value === "string";
}

function isArrayOf<T>(
	value: unknown,
	predicate: (item: unknown) => item is T,
): value is T[] {
	return Array.isArray(value) && value.every(predicate);
}

function isFiniteNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}

function hasOptionalStringProperty(
	value: Record<string, unknown>,
	property: string,
): boolean {
	return !(property in value) || isString(value[property]);
}

function isPlainBufferLike(value: unknown): boolean {
	return (
		isObject(value) &&
		typeof (value as Record<string, unknown>).byteLength === "number" &&
		Number.isFinite((value as Record<string, unknown>).byteLength)
	);
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
		isArrayOf((value as Record<string, unknown>).documents, isString)
	);
}

function isEmbedQueryPayload(
	value: unknown,
): value is WorkerRequestMap["embedQuery"] {
	return (
		isObject(value) && isString((value as Record<string, unknown>).document)
	);
}

function isSerializedErrorCause(value: unknown): boolean {
	return (
		isObject(value) &&
		isString(value.message) &&
		hasOptionalStringProperty(value, "name") &&
		hasOptionalStringProperty(value, "code")
	);
}

export function isSerializedError(value: unknown): value is SerializedError {
	return (
		isObject(value) &&
		isString(value.message) &&
		hasOptionalStringProperty(value, "name") &&
		hasOptionalStringProperty(value, "stack") &&
		hasOptionalStringProperty(value, "code") &&
		(!("cause" in value) ||
			(Array.isArray(value.cause) &&
				value.cause.every((cause) => isSerializedErrorCause(cause))))
	);
}

export function isWorkerRequest(value: unknown): value is WorkerRequest {
	if (!isObject(value)) {
		return false;
	}

	if (!isFiniteNumber(value.requestId) || typeof value.type !== "string") {
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

function isBatchDecisionEvent(
	value: unknown,
): value is Extract<EmbeddingPipelineEvent, { type: "batch" }> {
	if (!isObject(value) || value.type !== "batch") {
		return false;
	}

	return (
		isFiniteNumber((value as Record<string, unknown>).batchNumber) &&
		isFiniteNumber((value as Record<string, unknown>).batchDocuments) &&
		isFiniteNumber((value as Record<string, unknown>).batchTokens) &&
		isFiniteNumber((value as Record<string, unknown>).processedAfterBatch) &&
		isFiniteNumber((value as Record<string, unknown>).totalDocuments)
	);
}

function isEmbeddingPipelineEvent(
	value: unknown,
): value is EmbeddingPipelineEvent {
	return isBatchDecisionEvent(value);
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
	return (
		isObject(value) &&
		value.requestType === "embedDocuments" &&
		isEmbeddingPipelineEvent(value.event)
	);
}

function isDocumentsEmbeddedPayload(
	value: unknown,
): value is WorkerResponseMap["documentsEmbedded"] {
	if (!isObject(value)) return false;

	// Legacy: nested number[][]
	if (
		Array.isArray((value as any).embeddings) &&
		(value as any).embeddings.every(
			(vector: unknown) =>
				Array.isArray(vector) &&
				vector.every((entry) => typeof entry === "number"),
		)
	) {
		return true;
	}

	// Transferable buffer form: { embeddingsBuffer: { buffer, rows, dims } }
	if (isObject((value as any).embeddingsBuffer)) {
		const b = (value as any).embeddingsBuffer;
		if (
			typeof b.rows === "number" &&
			Number.isFinite(b.rows) &&
			b.rows >= 0 &&
			typeof b.dims === "number" &&
			Number.isFinite(b.dims) &&
			b.dims > 0 &&
			isPlainBufferLike(b.buffer)
		) {
			return true;
		}
	}

	return false;
}

function isQueryEmbeddedPayload(
	value: unknown,
): value is WorkerResponseMap["queryEmbedded"] {
	if (!isObject(value)) return false;

	if (
		Array.isArray((value as any).embedding) &&
		(value as any).embedding.every(
			(entry: unknown) => typeof entry === "number",
		)
	) {
		return true;
	}

	if (isObject((value as any).embeddingBuffer)) {
		const b = (value as any).embeddingBuffer;
		if (
			typeof b.dims === "number" &&
			Number.isFinite(b.dims) &&
			b.dims > 0 &&
			isPlainBufferLike(b.buffer)
		) {
			return true;
		}
	}

	return false;
}

export function isWorkerResponse(value: unknown): value is WorkerResponse {
	if (!isObject(value)) {
		return false;
	}

	if (!isFiniteNumber(value.requestId) || typeof value.type !== "string") {
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
