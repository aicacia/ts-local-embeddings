/**
 * Create an array of Document instances with a content prefix and optional metadata.
 */
export function createDocuments(
	count: number,
	contentPrefix = "document",
	metadataFactory?: (index: number) => Record<string, unknown>,
): Document[] {
	return Array.from(
		{ length: count },
		(_, index) =>
			new Document({
				pageContent: `${contentPrefix}-${index}-${"x".repeat(32)}`,
				metadata: metadataFactory ? metadataFactory(index) : { id: index },
			}),
	);
}
// Document creation and metadata normalization utilities
import { Document } from "@langchain/core/documents";

/**
 * Create a Document instance from content and metadata.
 */
export function createDocument(
	content: string,
	metadata: Record<string, unknown> = {},
	id?: string,
): Document {
	return new Document({
		pageContent: content,
		metadata,
		id,
	});
}

/**
 * Normalize metadata to ensure consistent structure.
 */
export function normalizeMetadata(
	metadata: unknown,
	length: number,
): Record<string, unknown>[] {
	if (Array.isArray(metadata)) return metadata as Record<string, unknown>[];
	return Array.from(
		{ length },
		() => (metadata as Record<string, unknown>) ?? {},
	);
}
