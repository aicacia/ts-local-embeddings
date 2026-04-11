import type { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";

export type StoredVectorRecord = {
	id: string;
	content: string;
	embeddingSpace?: string;
	contentHash?: string;
	cacheKey?: string;
	embedding: number[];
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
	dedupGroupCount: number;
};

export type VectorWritePipeline = {
	addDocuments(documents: Document[]): Promise<VectorWriteResult>;
	addVectors(
		vectors: number[][],
		documents: Document[],
	): Promise<VectorWriteResult>;
};

export type VectorWritePipelineOptions = {
	embeddings: EmbeddingsInterface<number[]>;
	resolveEmbeddingSpace: () => Promise<string>;
	getCachedRecords: (
		embeddingSpace: string,
		contentHashes: string[],
		contents: string[],
	) => Promise<Array<StoredVectorRecord | null>>;
	putRecords: (records: StoredVectorRecord[]) => Promise<void>;
	dedupStrategy?: VectorWriteDedupStrategy;
	strictDeterminism?: boolean;
};

function fallbackHash(input: string): string {
	let hash = 5381;
	for (let index = 0; index < input.length; index += 1) {
		hash = (hash * 33) ^ input.charCodeAt(index);
	}

	return `fallback-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

async function sha256(input: string): Promise<string> {
	if (typeof crypto === "undefined" || !crypto.subtle) {
		return fallbackHash(input);
	}

	const data = new TextEncoder().encode(input);
	const digest = await crypto.subtle.digest("SHA-256", data);
	return Array.from(new Uint8Array(digest), (byte) =>
		byte.toString(16).padStart(2, "0"),
	).join("");
}

export function resolveRecordId(document: Document, fallbackIndex: number): string {
	const metadataId = (document.metadata as { id?: unknown } | undefined)?.id;
	if (typeof metadataId === "string" && metadataId.trim().length > 0) {
		return metadataId;
	}

	if (typeof metadataId === "number" && Number.isFinite(metadataId)) {
		return String(metadataId);
	}

	if (
		typeof crypto !== "undefined" &&
		typeof crypto.randomUUID === "function"
	) {
		return crypto.randomUUID();
	}

	return `doc-${Date.now()}-${fallbackIndex}`;
}

export function mapStoredVectorRecord(args: {
	id: string;
	document: Document;
	embeddingSpace: string;
	contentHash: string;
	embedding: number[];
}): StoredVectorRecord {
	const metadata =
		(args.document.metadata as Record<string, unknown> | undefined) ?? {};

	return {
		id: args.id,
		content: args.document.pageContent,
		embeddingSpace: args.embeddingSpace,
		contentHash: args.contentHash,
		cacheKey: `${args.embeddingSpace}:${args.contentHash}`,
		embedding: args.embedding,
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

function equalVectors(left: number[], right: number[]): boolean {
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
			const contentHashes = await Promise.all(
				documents.map((document) => sha256(document.pageContent)),
			);
			const cachedRecords = await options.getCachedRecords(
				embeddingSpace,
				contentHashes,
				documents.map((document) => document.pageContent),
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
				const embeddedVectors = await options.embeddings.embedDocuments(
					uniquePendingGroups.map(
						(group) => group.representativeDocument.pageContent,
					),
				);

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
			const contentHashes = await Promise.all(
				documents.map((document) => sha256(document.pageContent)),
			);
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
						vectorByKey.set(key, record.embedding);
						continue;
					}

					if (!equalVectors(prior, record.embedding)) {
						throw new Error(
							"Deterministic embedding guard failed: repeated content produced mismatched vectors.",
						);
					}
				}
			}

			await options.putRecords(recordsToWrite);
			return {
				insertedCount: recordsToWrite.length,
				reusedEmbeddingCount: 0,
				dedupGroupCount: recordsToWrite.length,
			};
		},
	};
}