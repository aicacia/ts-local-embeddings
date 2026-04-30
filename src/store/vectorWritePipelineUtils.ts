/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Document } from "@langchain/core/documents";
import type { StoredVectorRecord } from "./vectorWritePipeline.js";
import { computeVectorNorm } from "./vectorMathUtils.js";
import { isTypedArray } from "../utils/typedArrayUtils.js";
import {
	fallbackHash,
	createContentHashGetter,
	sha256,
	type ContentHasher,
} from "../utils/contentHashUtils.js";
import { stableStringify } from "../utils/stableStringify.js";

export { fallbackHash, createContentHashGetter, sha256, stableStringify };
export type { ContentHasher };

const TYPED_ARRAY_CONVERSION_THRESHOLD = 16;

export function normalizeEmbeddingForStorage(
	embedding: number[] | Float32Array | ArrayBufferView,
): number[] | Float32Array | ArrayBufferView {
	return Array.isArray(embedding) &&
		embedding.length >= TYPED_ARRAY_CONVERSION_THRESHOLD
		? new Float32Array(embedding)
		: embedding;
}

export function computeEmbeddingNorm(
	embedding: number[] | Float32Array | ArrayBufferView,
): number | undefined {
	try {
		if (isTypedArray(embedding) || Array.isArray(embedding)) {
			return computeVectorNorm(embedding as ArrayLike<number>);
		}
	} catch (_e) {
		return undefined;
	}

	return undefined;
}

export function warnIfLargeArrayEmbedding(
	embeddedVectors: unknown,
	threshold: number,
	warningMessage: string,
): boolean {
	if (!Array.isArray(embeddedVectors)) {
		return false;
	}

	for (let index = 0; index < embeddedVectors.length; index += 1) {
		const ev = embeddedVectors[index] as unknown;
		if (Array.isArray(ev) && (ev as any).length >= threshold) {
			// eslint-disable-next-line no-console
			console.warn(warningMessage);
			return true;
		}
	}

	return false;
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
		return String(metadataId);
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

	const embeddingForStorage = normalizeEmbeddingForStorage(args.embedding);
	const embeddingNorm = computeEmbeddingNorm(embeddingForStorage);

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

export function resolveGroupKey(
	strategy: "contentHashOnly" | "contentAndText" | "none",
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

export function equalVectors(
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
