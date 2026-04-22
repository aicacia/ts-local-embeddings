/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { computeVectorNorm } from "./utils.js";

export type StoredVectorRecord = {
	id: string;
	content: string;
	embeddingSpace?: string;
	contentHash?: string;
	cacheKey?: string;
	// Allow typed arrays to avoid materializing large nested `number[]` copies
	// when possible. IndexedDB will structured-clone typed arrays efficiently.
	embedding: number[] | Float32Array | ArrayBufferView;
	embeddingNorm?: number;
	metadata: Record<string, unknown>;
};

type PendingEmbeddingGroup = {
	representativeDocument: Document;
	indices: number[];
};

export type VectorWriteDedupStrategy =
	| "contentHashOnly"
	| "contentAndText"
	| "none";

export type VectorWriteResult = {
	insertedCount: number;
	reusedEmbeddingCount: number;
	dedupGroupCount: number; // number of unique dedup groups processed
};

export type VectorWritePipeline = {
	addDocuments(documents: Document[]): Promise<VectorWriteResult>;
	addVectors(
		vectors: number[][],
		documents: Document[],
	): Promise<VectorWriteResult>;
};

export type VectorWritePipelineOptions = {
	embeddings: EmbeddingsInterface<number[] | Float32Array>;
	resolveEmbeddingSpace: () => Promise<string>;
	getCachedRecords: (
		embeddingSpace: string,
		contentHashes: string[],
		contents: string[],
	) => Promise<Array<StoredVectorRecord | null>>;
	putRecords: (records: StoredVectorRecord[]) => Promise<void>;
	dedupStrategy?: VectorWriteDedupStrategy;
	strictDeterminism?: boolean;
	// Optional custom content hashing function (sync or async). If provided,
	// this will be used instead of the built-in `sha256` / fallback.
	contentHasher?: (content: string) => Promise<string> | string;
	// Max entries for the in-memory content hash cache. Default: 4096.
	contentHashCacheMax?: number;
};

function fallbackHash(input: string): string {
	let hash = 5381;
	for (let index = 0; index < input.length; index += 1) {
		hash = (hash * 33) ^ input.charCodeAt(index);
	}

	return `fallback-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

let __nodeCryptoModule: any | null = null;

async function sha256(input: string): Promise<string> {
	// Node.js fast path: use synchronous `createHash` via dynamic import when
	// running under Node to avoid the async WebCrypto overhead.
	try {
		if (typeof process !== "undefined" && (process as any).versions?.node) {
			if (__nodeCryptoModule === null) {
				__nodeCryptoModule = await import("node:crypto");
			}
			return __nodeCryptoModule
				.createHash("sha256")
				.update(input)
				.digest("hex");
		}
	} catch (_) {
		// fall through to fallback hashing in browser environments.
	}

	// For browser-embedded workloads, use a deterministic JS hash instead of
	// async WebCrypto digest to reduce latency and avoid extra promise/encoder cost.
	return fallbackHash(input);
}

function stableStringify(value: unknown): string {
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value);
	}

	if (Array.isArray(value)) {
		return `[${value.map((item) => stableStringify(item)).join(",")}]`;
	}

	const objectValue = value as Record<string, unknown>;
	const sortedKeys = Object.keys(objectValue).sort();
	return `{
${sortedKeys
	.map((key) => `${JSON.stringify(key)}:${stableStringify(objectValue[key])}`)
	.join(",")}}`;
}

export function resolveRecordId(
	document: Document,
	fallbackIndex: number,
): string {
	const metadataId = (document.metadata as { id?: unknown } | undefined)?.id;
	if (typeof metadataId === "string" && metadataId.trim().length > 0) {
		return metadataId;
	}

	if (typeof metadataId === "number" && Number.isFinite(metadataId)) {
	}

	if (
		typeof crypto !== "undefined" &&
		typeof crypto.randomUUID === "function"
	) {
		return crypto.randomUUID();
	}

	const metadataString = stableStringify(
		(document.metadata as Record<string, unknown> | undefined) ?? {},
	);
	return `doc-${fallbackHash(
		`${document.pageContent}:${metadataString}:${fallbackIndex}`,
	)}`;
}

export function mapStoredVectorRecord(args: {
	id: string;
	document: Document;
	embeddingSpace: string;
	contentHash: string;
	embedding: number[] | Float32Array | ArrayBufferView;
}): StoredVectorRecord {
	const metadata =
		(args.document.metadata as Record<string, unknown> | undefined) ?? {};

	// Prefer typed arrays for storage to reduce structured-clone overhead when
	// writing large numeric vectors to IndexedDB.
	const emb = args.embedding;
	// Avoid converting small numeric arrays to `Float32Array` as the
	// allocation/copy cost can outweigh structured-clone benefits for tiny
	// vectors used in many browser workloads. Only coerce to a typed array
	// when the vector length exceeds a conservative threshold.
	const TYPED_ARRAY_CONVERSION_THRESHOLD = 16;
	const embeddingForStorage: number[] | Float32Array | ArrayBufferView =
		Array.isArray(emb) && emb.length >= TYPED_ARRAY_CONVERSION_THRESHOLD
			? new Float32Array(emb)
			: Array.isArray(emb)
				? emb
				: emb;

	let embeddingNorm: number | undefined;
	try {
		if (ArrayBuffer.isView(embeddingForStorage)) {
			embeddingNorm = computeVectorNorm(
				embeddingForStorage as unknown as ArrayLike<number>,
			);
		} else if (Array.isArray(embeddingForStorage)) {
			embeddingNorm = computeVectorNorm(
				embeddingForStorage as ArrayLike<number>,
			);
		}
	} catch (_e) {
		embeddingNorm = undefined;
	}

	return {
		id: args.id,
		content: args.document.pageContent,
		embeddingSpace: args.embeddingSpace,
		contentHash: args.contentHash,
		cacheKey: `${args.embeddingSpace}:${args.contentHash}`,
		embedding: embeddingForStorage,
		embeddingNorm,
		metadata,
	};
}

function resolveGroupKey(
	strategy: VectorWriteDedupStrategy,
	contentHash: string,
	document: Document,
	index: number,
): string {
	switch (strategy) {
		case "contentHashOnly":
			return contentHash;
		case "none":
			return `${index}:${contentHash}`;
		default:
			return `${contentHash}:${document.pageContent}`;
	}
}

function equalVectors(
	left: ArrayLike<number>,
	right: ArrayLike<number>,
): boolean {
	if (left.length !== right.length) {
		return false;
	}

	for (let index = 0; index < left.length; index += 1) {
		if (left[index] !== right[index]) {
			return false;
		}
	}

	return true;
}

export function createVectorWritePipeline(
	options: VectorWritePipelineOptions,
): VectorWritePipeline {
	const dedupStrategy = options.dedupStrategy ?? "contentAndText";
	const strictDeterminism = options.strictDeterminism ?? false;
	// Recommend embeddings runtimes return typed arrays for large vectors to
	// avoid repeated numeric array allocations and structured-clone copies.
	const EMBEDDING_TYPED_ARRAY_RECOMMENDATION_THRESHOLD = 16;
	let warnedAboutArrayReturn = false;
	// LRU-ish bounded cache for computed content hashes to avoid unbounded
	// Map growth and repeated expensive hashing for identical documents.
	const CONTENT_HASH_CACHE_MAX =
		typeof options.contentHashCacheMax === "number" &&
		Number.isFinite(options.contentHashCacheMax) &&
		options.contentHashCacheMax > 0
			? Math.max(1, Math.floor(options.contentHashCacheMax))
			: 4096;
	const contentHashCache = new Map<string, Promise<string>>();

	function setContentHashCache(key: string, value: Promise<string>): void {
		contentHashCache.set(key, value);
		if (contentHashCache.size > CONTENT_HASH_CACHE_MAX) {
			// evict oldest entry (Map preserves insertion order)
			const firstKey = contentHashCache.keys().next().value as
				| string
				| undefined;
			if (firstKey !== undefined) {
				contentHashCache.delete(firstKey);
			}
		}
	}

	async function getContentHash(content: string): Promise<string> {
		const cachedHash = contentHashCache.get(content);
		if (cachedHash) {
			return cachedHash;
		}

		const hashPromise = (async () => {
			if (typeof options.contentHasher === "function") {
				const maybe = options.contentHasher(content);
				return typeof maybe === "string" ? maybe : await maybe;
			}
			return sha256(content);
		})();

		setContentHashCache(content, hashPromise);
		return hashPromise;
	}

	return {
		async addDocuments(documents: Document[]): Promise<VectorWriteResult> {
			if (documents.length === 0) {
				return {
					insertedCount: 0,
					reusedEmbeddingCount: 0,
					dedupGroupCount: 0,
				};
			}

			const embeddingSpace = await options.resolveEmbeddingSpace();
			const resolvedIds = documents.map((document, index) =>
				resolveRecordId(document, index),
			);
			const contents = documents.map((document) => document.pageContent);
			const contentHashes = await Promise.all(
				contents.map((content) => getContentHash(content)),
			);
			const cachedRecords = await options.getCachedRecords(
				embeddingSpace,
				contentHashes,
				contents,
			);

			const recordsToWrite: StoredVectorRecord[] = [];
			const pendingEmbeddingGroups = new Map<string, PendingEmbeddingGroup>();
			let reusedEmbeddingCount = 0;

			for (let index = 0; index < documents.length; index += 1) {
				const document = documents[index];
				const id = resolvedIds[index];
				const contentHash = contentHashes[index];
				const cachedRecord = cachedRecords[index];

				if (cachedRecord) {
					reusedEmbeddingCount += 1;
					recordsToWrite.push(
						mapStoredVectorRecord({
							id,
							document,
							embeddingSpace,
							contentHash,
							embedding: cachedRecord.embedding,
						}),
					);
					continue;
				}

				const groupKey = resolveGroupKey(
					dedupStrategy,
					contentHash,
					document,
					index,
				);
				const existingGroup = pendingEmbeddingGroups.get(groupKey);
				if (existingGroup) {
					existingGroup.indices.push(index);
					continue;
				}

				pendingEmbeddingGroups.set(groupKey, {
					representativeDocument: document,
					indices: [index],
				});
			}

			const uniquePendingGroups = Array.from(pendingEmbeddingGroups.values());
			if (uniquePendingGroups.length > 0) {
				const inputs = uniquePendingGroups.map(
					(group) => group.representativeDocument.pageContent,
				);

				// Prefer embedding runtimes that expose `embedDocumentsRaw` which
				// returns a packed ArrayBuffer { buffer, rows, dims } so we can
				// avoid per-row allocations and create zero-copy Float32Array
				// subarray views into the transferred buffer.
				const embedRaw = (options.embeddings as any).embedDocumentsRaw;
				if (typeof embedRaw === "function") {
					try {
						const raw = await embedRaw.call(
							options.embeddings,
							inputs as string[],
						);
						if (
							raw &&
							raw.buffer instanceof ArrayBuffer &&
							typeof raw.rows === "number" &&
							typeof raw.dims === "number"
						) {
							const { buffer, rows, dims } = raw as {
								buffer: ArrayBuffer;
								rows: number;
								dims: number;
							};

							if (rows !== uniquePendingGroups.length) {
								throw new Error(
									`Embedding runtime produced ${rows} vectors for ${uniquePendingGroups.length} documents.`,
								);
							}

							const float32 = new Float32Array(buffer);
							for (let groupIndex = 0; groupIndex < rows; groupIndex += 1) {
								const start = groupIndex * dims;
								const view = float32.subarray(start, start + dims);
								const group = uniquePendingGroups[groupIndex];
								for (const originalIndex of group.indices) {
									const document = documents[originalIndex];
									recordsToWrite.push(
										mapStoredVectorRecord({
											id: resolvedIds[originalIndex],
											document,
											embeddingSpace,
											contentHash: contentHashes[originalIndex],
											embedding: view,
										}),
									);
								}
							}

							// Skip fallback embedDocuments path.
							await options.putRecords(recordsToWrite);
							return {
								insertedCount: recordsToWrite.length,
								reusedEmbeddingCount,
								dedupGroupCount: uniquePendingGroups.length,
							} as VectorWriteResult;
						}
					} catch (err) {
						// If raw embedding extraction fails, fall back to standard path.
					}
				}

				const embeddedVectors = await options.embeddings.embedDocuments(inputs);

				// If the embedding runtime returns plain `number[]` arrays for large
				// vectors, log a one-time recommendation to return `Float32Array` to
				// avoid allocation + structured-clone overhead on the main thread.
				if (!warnedAboutArrayReturn && Array.isArray(embeddedVectors)) {
					for (let vi = 0; vi < embeddedVectors.length; vi++) {
						const ev = embeddedVectors[vi] as unknown;
						if (
							Array.isArray(ev) &&
							(ev as any).length >=
								EMBEDDING_TYPED_ARRAY_RECOMMENDATION_THRESHOLD
						) {
							// eslint-disable-next-line no-console
							console.warn(
								"[local-embeddings] Performance: consider returning Float32Array from embeddings.embedDocuments to avoid copying large vectors.",
							);
							warnedAboutArrayReturn = true;
							break;
						}
					}
				}

				if (embeddedVectors.length !== uniquePendingGroups.length) {
					throw new Error(
						`Embedding runtime produced ${embeddedVectors.length} vectors for ${uniquePendingGroups.length} documents.`,
					);
				}

				for (
					let groupIndex = 0;
					groupIndex < uniquePendingGroups.length;
					groupIndex += 1
				) {
					const group = uniquePendingGroups[groupIndex];
					const embedding = embeddedVectors[groupIndex] ?? [];

					for (const originalIndex of group.indices) {
						const document = documents[originalIndex];
						recordsToWrite.push(
							mapStoredVectorRecord({
								id: resolvedIds[originalIndex],
								document,
								embeddingSpace,
								contentHash: contentHashes[originalIndex],
								embedding,
							}),
						);
					}
				}
			}

			await options.putRecords(recordsToWrite);
			return {
				insertedCount: recordsToWrite.length,
				reusedEmbeddingCount,
				dedupGroupCount: uniquePendingGroups.length,
			};
		},

		async addVectors(
			vectors: number[][],
			documents: Document[],
		): Promise<VectorWriteResult> {
			if (vectors.length !== documents.length) {
				throw new Error(
					`Expected vectors/documents lengths to match, got vectors=${vectors.length}, documents=${documents.length}.`,
				);
			}

			if (vectors.length === 0) {
				return {
					insertedCount: 0,
					reusedEmbeddingCount: 0,
					dedupGroupCount: 0,
				};
			}

			const embeddingSpace = await options.resolveEmbeddingSpace();
			const resolvedIds = documents.map((document, index) =>
				resolveRecordId(document, index),
			);
			const contents = documents.map((document) => document.pageContent);
			const contentHashes = await Promise.all(
				contents.map((content) => getContentHash(content)),
			);
			// If callers pass plain nested arrays, encourage typed arrays for
			// larger vectors (one-time warning).
			if (
				!warnedAboutArrayReturn &&
				Array.isArray(vectors) &&
				vectors.length > 0
			) {
				const first = vectors[0] as any;
				if (
					Array.isArray(first) &&
					first.length >= EMBEDDING_TYPED_ARRAY_RECOMMENDATION_THRESHOLD
				) {
					// eslint-disable-next-line no-console
					console.warn(
						"[local-embeddings] Performance: consider passing Float32Array vectors to addVectors to avoid copies.",
					);
					warnedAboutArrayReturn = true;
				}
			}

			const recordsToWrite = vectors.map((embedding, index) =>
				mapStoredVectorRecord({
					id: resolvedIds[index],
					document: documents[index],
					embeddingSpace,
					contentHash: contentHashes[index],
					embedding,
				}),
			);

			if (strictDeterminism) {
				const vectorByKey = new Map<string, number[]>();
				for (let index = 0; index < recordsToWrite.length; index += 1) {
					const record = recordsToWrite[index];
					const key = resolveGroupKey(
						dedupStrategy,
						record.contentHash ?? "",
						documents[index],
						index,
					);
					const prior = vectorByKey.get(key);
					if (!prior) {
						vectorByKey.set(
							key,
							Array.isArray(record.embedding)
								? record.embedding
								: Array.from(record.embedding as ArrayLike<number>),
						);
						continue;
					}

					if (!equalVectors(prior, record.embedding as ArrayLike<number>)) {
						throw new Error(
							"Deterministic embedding guard failed: repeated content produced mismatched vectors.",
						);
					}
				}
			}

			const uniqueGroupKeys = new Set(
				recordsToWrite.map((record, index) =>
					resolveGroupKey(
						dedupStrategy,
						record.contentHash ?? "",
						documents[index],
						index,
					),
				),
			);

			await options.putRecords(recordsToWrite);
			return {
				insertedCount: recordsToWrite.length,
				reusedEmbeddingCount: 0,
				dedupGroupCount: uniqueGroupKeys.size,
			};
		},
	};
}
