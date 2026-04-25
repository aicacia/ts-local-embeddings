/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Document } from "@langchain/core/documents";
import type { StoredVectorRecord } from "./vectorWritePipeline.js";
import { computeVectorNorm } from "./utils.js";

export function fallbackHash(input: string): string {
  let hash = 5381;
  for (let index = 0; index < input.length; index += 1) {
    hash = (hash * 33) ^ input.charCodeAt(index);
  }

  return `fallback-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

let __nodeCryptoModule: any | null = null;

export async function sha256(input: string): Promise<string> {
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
  } catch (_error) {
    // fall through to fallback hashing in browser environments.
  }

  return fallbackHash(input);
}

export function stableStringify(value: unknown): string {
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

  const emb = args.embedding;
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
