/// <reference lib="webworker" />

/* eslint-disable @typescript-eslint/no-explicit-any */

import { LocalEmbeddings } from "../pipeline/LocalEmbeddings.js";
import { setDebugLogging } from "../debug.js";
import { loadEmbeddingRuntime } from "../runtime/embeddingRuntime.js";
import type { LoadEmbeddingRuntimeOptions } from "../runtime/embeddingRuntime.js";
import {
	areLoadEmbeddingRuntimeOptionsEqual,
	normalizeLoadEmbeddingRuntimeOptions,
} from "./workerRuntimeOptions.js";
import {
	packRowsToFloat32,
	arrayLikeToFloat32,
} from "../utils/typedArrayUtils.js";
import type {
	SerializedError,
	WorkerRequest,
	WorkerRequestMap,
	WorkerResponse,
	WorkerResponseWithTransfer,
} from "./embeddingWorkerProtocol.js";
import { isWorkerRequest } from "./embeddingWorkerProtocol.js";

let embeddings: LocalEmbeddings | null = null;
let initializing: Promise<void> | null = null;
let runtimeInfo: { modelId: string; variant: string } | null = null;
let initializationOptions: LoadEmbeddingRuntimeOptions | undefined = undefined;
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

function post(response: WorkerResponseWithTransfer): void {
	// Allow optional transfer of ArrayBuffer(s) when provided on the response
	// object via a special `_transfer` property to avoid changing the public
	// WorkerResponse shape used elsewhere.
	const maybeTransfer = response._transfer;
	if (Array.isArray(maybeTransfer) && maybeTransfer.length > 0) {
		// Remove the helper property before posting so it doesn't leak to the
		// receiver payload type checks.
		// Assign `undefined` instead of using `delete` for better performance.
		response._transfer = undefined;
		try {
			self.postMessage(response, maybeTransfer as Transferable[]);
			return;
		} catch (err) {
			// Fall back to non-transferred post if transfer fails.
			// eslint-disable-next-line no-console
			console.error("[worker] postMessage with transfer failed", err);
		}
	}

	self.postMessage(response);
}

async function ensureInitialized(
	options: WorkerRequestMap["init"]["options"],
): Promise<void> {
	if (embeddings !== null) {
		if (options === undefined) {
			return;
		}

		if (!areLoadEmbeddingRuntimeOptionsEqual(initializationOptions, options)) {
			const stored = normalizeLoadEmbeddingRuntimeOptions(
				initializationOptions,
			);
			const attempted = normalizeLoadEmbeddingRuntimeOptions(options);
			throw new Error(
				`Embedding worker already initialized with different runtime options. stored=${JSON.stringify(
					stored,
				)}, attempted=${JSON.stringify(attempted)}`,
			);
		}

		return;
	}

	if (initializing !== null) {
		if (options === undefined) {
			try {
				await initializing;
			} catch (error) {
				// Allow retrying initialization on subsequent requests after transient failures.
				initializing = null;
				initializationOptions = undefined;
				runtimeInfo = null;
				embeddings = null;
				throw error;
			}

			return;
		}

		if (!areLoadEmbeddingRuntimeOptionsEqual(initializationOptions, options)) {
			const stored = normalizeLoadEmbeddingRuntimeOptions(
				initializationOptions,
			);
			const attempted = normalizeLoadEmbeddingRuntimeOptions(options);
			throw new Error(
				`Embedding worker is already initializing with different runtime options. stored=${JSON.stringify(
					stored,
				)}, attempted=${JSON.stringify(attempted)}`,
			);
		}
	}

	if (initializing === null) {
		const normalizedOptions = normalizeLoadEmbeddingRuntimeOptions(options);
		initializationOptions = normalizedOptions;
		initializing = (async () => {
			setDebugLogging(normalizedOptions.debugLogging === true);
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
		initializationOptions = undefined;
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
					// Prefer a raw packed-buffer API if the runtime exposes it. This
					// allows the runtime to produce a transferable ArrayBuffer without
					// an extra copy on the worker side.
					const embedRaw = (runtimeEmbeddings as any).embedDocumentsRaw;
					if (typeof embedRaw === "function") {
						try {
							const raw = await embedRaw.call(
								runtimeEmbeddings,
								request.payload.documents,
							);
							if (
								raw &&
								(raw.buffer instanceof ArrayBuffer ||
									(raw.buffer && raw.buffer.buffer instanceof ArrayBuffer)) &&
								typeof raw.rows === "number" &&
								typeof raw.dims === "number"
							) {
								const buffer =
									raw.buffer instanceof ArrayBuffer
										? raw.buffer
										: raw.buffer.buffer;
								const response: WorkerResponseWithTransfer = {
									type: "documentsEmbedded",
									requestId: request.requestId,
									payload: {
										embeddingsBuffer: {
											buffer,
											rows: raw.rows,
											dims: raw.dims,
										},
									},
									_transfer: [buffer],
								};
								post(response);
								return;
							}
						} catch (_err) {
							// Fall back to legacy path on any error from raw method.
						}
					}

					const result = await runtimeEmbeddings.embedDocuments(
						request.payload.documents,
					);

					// If we have non-empty embeddings, attempt to pack into a
					// Float32Array and transfer the underlying ArrayBuffer to the
					// main thread to avoid structured-clone overhead for large
					// matrices. Fall back to nested arrays for empty results.
					if (Array.isArray(result) && result.length > 0) {
						const packed = packRowsToFloat32(result as ArrayLike<number>[]);
						const { buffer, rows, dims } = packed;

						const response: WorkerResponseWithTransfer = {
							type: "documentsEmbedded",
							requestId: request.requestId,
							payload: {
								embeddingsBuffer: {
									buffer,
									rows,
									dims,
								},
							},
							// include transfer list via helper property removed in `post`
							_transfer: [buffer],
						};
						post(response);
					} else {
						post({
							type: "documentsEmbedded",
							requestId: request.requestId,
							payload: {
								embeddings: result as any,
							},
						});
					}
				} finally {
					activeProgressRequestId = null;
				}
				return;
			}
			case "embedQuery": {
				const runtimeEmbeddings = await getInitializedEmbeddings();
				// Prefer a raw single-vector buffer API when available.
				const embedQueryRaw = (runtimeEmbeddings as any).embedQueryRaw;
				if (typeof embedQueryRaw === "function") {
					try {
						const raw = await embedQueryRaw.call(
							runtimeEmbeddings,
							request.payload.document,
						);
						if (
							raw &&
							(raw.buffer instanceof ArrayBuffer ||
								(raw.buffer && raw.buffer.buffer instanceof ArrayBuffer)) &&
							typeof raw.dims === "number"
						) {
							const buffer =
								raw.buffer instanceof ArrayBuffer
									? raw.buffer
									: raw.buffer.buffer;
							const response: WorkerResponseWithTransfer = {
								type: "queryEmbedded",
								requestId: request.requestId,
								payload: { embeddingBuffer: { buffer, dims: raw.dims } },
								_transfer: [buffer],
							};
							post(response);
							return;
						}
					} catch (_err) {
						// fall through to legacy path
					}
				}

				const result = await runtimeEmbeddings.embedQuery(
					request.payload.document,
				);

				if (Array.isArray(result) && result.length > 0) {
					const view = arrayLikeToFloat32(result as ArrayLike<number>);
					const dims = view.length;
					let transferBuffer: ArrayBuffer;
					if (
						(view as any).byteOffset === 0 &&
						view.buffer.byteLength === dims * 4
					) {
						transferBuffer = view.buffer;
					} else {
						// ensure we transfer a compact buffer matching dims
						const copy = new Float32Array(view);
						transferBuffer = copy.buffer;
					}

					const response: WorkerResponseWithTransfer = {
						type: "queryEmbedded",
						requestId: request.requestId,
						payload: {
							embeddingBuffer: {
								buffer: transferBuffer,
								dims,
							},
						},
						_transfer: [transferBuffer],
					};
					post(response);
				} else {
					post({
						type: "queryEmbedded",
						requestId: request.requestId,
						payload: {
							embedding: result as any,
						},
					});
				}
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
