/// <reference lib="webworker" />

import { LocalEmbeddings } from "./LocalEmbeddings.js";
import { loadEmbeddingRuntime } from "./embeddingRuntime.js";
import type {
	SerializedError,
	WorkerRequest,
	WorkerRequestMap,
	WorkerResponse,
} from "./embeddingWorkerProtocol.js";

let embeddings: LocalEmbeddings | null = null;
let initializing: Promise<void> | null = null;
let runtimeInfo: { modelId: string; variant: string } | null = null;

function serializeError(error: unknown): SerializedError {
	if (error instanceof Error) {
		return {
			message: error.message,
			name: error.name,
			stack: error.stack,
		};
	}

	return {
		message: typeof error === "string" ? error : "Unknown worker error.",
	};
}

function post(response: WorkerResponse): void {
	self.postMessage(response);
}

async function ensureInitialized(
	options: WorkerRequestMap["init"]["options"],
): Promise<void> {
	if (embeddings !== null) {
		return;
	}

	if (initializing === null) {
		initializing = (async () => {
			const runtime = await loadEmbeddingRuntime(options ?? {});
			runtimeInfo = { modelId: runtime.modelId, variant: runtime.variant };
			embeddings = new LocalEmbeddings(runtime);
		})();
	}

	try {
		await initializing;
	} catch (error) {
		// Allow retrying initialization on subsequent requests after transient failures.
		initializing = null;
		runtimeInfo = null;
		embeddings = null;
		throw error;
	}
}

async function getInitializedEmbeddings(): Promise<LocalEmbeddings> {
	await ensureInitialized(undefined);
	if (embeddings === null) {
		throw new Error("Worker embeddings are not initialized.");
	}

	return embeddings;
}

async function handleRequest(request: WorkerRequest): Promise<void> {
	try {
		switch (request.type) {
			case "init": {
				await ensureInitialized(request.payload.options);
				if (runtimeInfo === null) {
					throw new Error("Worker initialized without runtime metadata.");
				}
				post({
					type: "ready",
					requestId: request.requestId,
					payload: {
						runtime: runtimeInfo,
					},
				});
				return;
			}
			case "embedDocuments": {
				const runtimeEmbeddings = await getInitializedEmbeddings();
				const result = await runtimeEmbeddings.embedDocuments(
					request.payload.documents,
				);
				post({
					type: "documentsEmbedded",
					requestId: request.requestId,
					payload: {
						embeddings: result,
					},
				});
				return;
			}
			case "embedQuery": {
				const runtimeEmbeddings = await getInitializedEmbeddings();
				const result = await runtimeEmbeddings.embedQuery(
					request.payload.document,
				);
				post({
					type: "queryEmbedded",
					requestId: request.requestId,
					payload: {
						embedding: result,
					},
				});
				return;
			}
		}
	} catch (error) {
		post({
			type: "error",
			requestId: request.requestId,
			payload: serializeError(error),
		});
	}
}

self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
	void handleRequest(event.data);
};
