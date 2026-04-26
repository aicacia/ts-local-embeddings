import type {
	WorkerResponseMap,
	WorkerSuccessResponse,
} from "./embeddingWorkerProtocol.js";

export function isMessageEvent(value: unknown): value is MessageEvent<unknown> {
	return typeof value === "object" && value !== null && "data" in value;
}

export function getWorkerMessageData(event: unknown): unknown {
	if (isMessageEvent(event)) {
		return event.data;
	}

	return event;
}

export function mapDocumentsEmbeddedResponse(
	response: WorkerSuccessResponse,
): Array<number[] | Float32Array> {
	if (response.type !== "documentsEmbedded") {
		throw new Error(
			`Embedding worker returned ${response.type} for embedDocuments request.`,
		);
	}

	const payload = response.payload as WorkerResponseMap["documentsEmbedded"];

	if (Array.isArray(payload.embeddings)) {
		return payload.embeddings;
	}

	if (payload.embeddingsBuffer?.buffer) {
		const { buffer, rows, dims } = payload.embeddingsBuffer;
		const float32 = new Float32Array(buffer as ArrayBuffer);
		const out: Array<number[] | Float32Array> = new Array(rows);
		for (let r = 0; r < rows; r += 1) {
			const start = r * dims;
			out[r] = float32.subarray(start, start + dims);
		}
		return out;
	}

	return [];
}

export function mapQueryEmbeddedResponse(
	response: WorkerSuccessResponse,
): number[] | Float32Array {
	if (response.type !== "queryEmbedded") {
		throw new Error(
			`Embedding worker returned ${response.type} for embedQuery request.`,
		);
	}

	const payload = response.payload as WorkerResponseMap["queryEmbedded"];

	if (Array.isArray(payload.embedding)) {
		return payload.embedding;
	}

	if (payload.embeddingBuffer?.buffer) {
		const { buffer, dims } = payload.embeddingBuffer;
		const float32 = new Float32Array(buffer as ArrayBuffer);
		return float32.subarray(0, dims);
	}

	return [];
}
