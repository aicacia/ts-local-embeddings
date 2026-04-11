/// <reference lib="webworker" />

import { LocalEmbeddings } from "./LocalEmbeddings.js";
import { setDebugLogging } from "./debug.js";
import { loadEmbeddingRuntime } from "./embeddingRuntime.js";
import type {
	SerializedError,
	WorkerRequest,
	WorkerRequestMap,
	WorkerResponse,
} from "./embeddingWorkerProtocol.js";
import { isWorkerRequest } from "./embeddingWorkerProtocol.js";

let embeddings: LocalEmbeddings | null = null;
let initializing: Promise<void> | null = null;
let runtimeInfo: { modelId: string; variant: string } | null = null;
let activeProgressRequestId: number | null = null;

function serializeError(error: unknown): SerializedError {
	if (error instanceof Error) {
		const code = (error as Error & { code?: unknown }).code;
		const causes: SerializedError["cause"] = [];
		let currentCause = (error as Error & { cause?: unknown }).cause;
		let depth = 0;

		while (depth < 5 && currentCause !== undefined) {
			if (currentCause instanceof Error) {
				const causeCode = (currentCause as Error & { code?: unknown }).code;
				causes.push({
					message: currentCause.message,
					name: currentCause.name,
					code:
						typeof causeCode === "string" || typeof causeCode === "number"
							? String(causeCode)
							: undefined,
				});
				currentCause = (currentCause as Error & { cause?: unknown }).cause;
				depth += 1;
				continue;
			}

			if (typeof currentCause === "string") {
				causes.push({ message: currentCause });
			}
			break;
		}

		return {
			message: error.message,
			name: error.name,
			stack: error.stack,
			code:
				typeof code === "string" || typeof code === "number"
					? String(code)
					: undefined,
			cause: causes.length > 0 ? causes : undefined,
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
			setDebugLogging(options?.debugLogging === true);
			const runtime = await loadEmbeddingRuntime(options ?? {});
			runtimeInfo = { modelId: runtime.modelId, variant: runtime.variant };
			embeddings = new LocalEmbeddings(runtime, {
				onEvent: (event) => {
					if (activeProgressRequestId === null) {
						return;
					}

					post({
						type: "progress",
						requestId: activeProgressRequestId,
						payload: {
							requestType: "embedDocuments",
							event,
						},
					});
				},
			});
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
				activeProgressRequestId = request.requestId;
				try {
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
				} finally {
					activeProgressRequestId = null;
				}
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

self.onmessage = (event: MessageEvent<unknown>): void => {
	if (!isWorkerRequest(event.data)) {
		post({
			type: "error",
			requestId: -1,
			payload: {
				message: "Worker received an invalid request payload.",
				name: "ProtocolValidationError",
			},
		});
		return;
	}

	void handleRequest(event.data);
};
