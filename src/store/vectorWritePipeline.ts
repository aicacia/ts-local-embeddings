/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import {
	equalVectors,
	mapStoredVectorRecord,
	resolveGroupKey,
	resolveRecordId,
	createContentHashGetter,
} from "./vectorWritePipelineUtils.js";

export { resolveRecordId };

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

export function createVectorWritePipeline(
	options: VectorWritePipelineOptions,
): VectorWritePipeline {
	const dedupStrategy = options.dedupStrategy ?? "contentAndText";
	const strictDeterminism = options.strictDeterminism ?? false;
	// Recommend embeddings runtimes return typed arrays for large vectors to
	// avoid repeated numeric array allocations and structured-clone copies.
	const EMBEDDING_TYPED_ARRAY_RECOMMENDATION_THRESHOLD = 16;
	let warnedAboutArrayReturn = false;

	const getContentHash = createContentHashGetter({
		contentHasher: options.contentHasher,
		contentHashCacheMax: options.contentHashCacheMax,
	});

	function groupDocumentsForEmbedding(
		documents: Document[],
		contentHashes: string[],
		cachedRecords: Array<StoredVectorRecord | null>,
		resolvedIds: string[],
		dedupStrategy: VectorWriteDedupStrategy,
		embeddingSpace: string,
	) {
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

		return {
			recordsToWrite,
			pendingGroups: Array.from(pendingEmbeddingGroups.values()),
			reusedEmbeddingCount,
		};
	}

	function addPendingGroupRecords(
		recordsToWrite: StoredVectorRecord[],
		pendingGroups: PendingEmbeddingGroup[],
		embeddedVectors: Array<number[] | Float32Array>,
		resolvedIds: string[],
		contentHashes: string[],
		documents: Document[],
		embeddingSpace: string,
	) {
		for (
			let groupIndex = 0;
			groupIndex < pendingGroups.length;
			groupIndex += 1
		) {
			const group = pendingGroups[groupIndex];
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

	async function tryWriteRawEmbeddedRecords(
		recordsToWrite: StoredVectorRecord[],
		pendingGroups: PendingEmbeddingGroup[],
		resolvedIds: string[],
		contentHashes: string[],
		documents: Document[],
		embeddingSpace: string,
		embedRaw: (documents: string[]) => Promise<unknown>,
	): Promise<boolean> {
		const raw = await embedRaw(
			pendingGroups.map((group) => group.representativeDocument.pageContent),
		);
		if (
			!raw ||
			!(raw as { buffer?: unknown }).buffer ||
			typeof (raw as any).rows !== "number" ||
			typeof (raw as any).dims !== "number"
		) {
			return false;
		}

		const { buffer, rows, dims } = raw as {
			buffer: ArrayBuffer;
			rows: number;
			dims: number;
		};

		if (rows !== pendingGroups.length) {
			throw new Error(
				`Embedding runtime produced ${rows} vectors for ${pendingGroups.length} documents.`,
			);
		}

		const float32 = new Float32Array(buffer);
		for (let groupIndex = 0; groupIndex < rows; groupIndex += 1) {
			const start = groupIndex * dims;
			const view = float32.subarray(start, start + dims);
			const group = pendingGroups[groupIndex];
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

		return true;
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

			const { recordsToWrite, pendingGroups, reusedEmbeddingCount } =
				groupDocumentsForEmbedding(
					documents,
					contentHashes,
					cachedRecords,
					resolvedIds,
					dedupStrategy,
					embeddingSpace,
				);

			if (pendingGroups.length > 0) {
				const inputs = pendingGroups.map(
					(group) => group.representativeDocument.pageContent,
				);

				const embedRaw = (options.embeddings as any).embedDocumentsRaw;
				if (typeof embedRaw === "function") {
					try {
						const wroteRaw = await tryWriteRawEmbeddedRecords(
							recordsToWrite,
							pendingGroups,
							resolvedIds,
							contentHashes,
							documents,
							embeddingSpace,
							embedRaw.bind(options.embeddings),
						);

						if (wroteRaw) {
							await options.putRecords(recordsToWrite);
							return {
								insertedCount: recordsToWrite.length,
								reusedEmbeddingCount,
								dedupGroupCount: pendingGroups.length,
							};
						}
					} catch {
						// If raw embedding extraction fails, fall back to standard path.
					}
				}

				const embeddedVectors = await options.embeddings.embedDocuments(inputs);

				if (embeddedVectors.length !== pendingGroups.length) {
					throw new Error(
						`Embedding runtime produced ${embeddedVectors.length} vectors for ${pendingGroups.length} documents.`,
					);
				}

				addPendingGroupRecords(
					recordsToWrite,
					pendingGroups,
					embeddedVectors,
					resolvedIds,
					contentHashes,
					documents,
					embeddingSpace,
				);
			}

			await options.putRecords(recordsToWrite);
			return {
				insertedCount: recordsToWrite.length,
				reusedEmbeddingCount,
				dedupGroupCount: pendingGroups.length,
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
