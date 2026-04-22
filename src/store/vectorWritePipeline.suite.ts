/* eslint-disable @typescript-eslint/no-explicit-any */
import type { Suite, Deferred } from "benchmark";
import { Document } from "@langchain/core/documents";
import { createVectorWritePipeline } from "./vectorWritePipeline.js";
import type { Constructor } from "../types.js";

function createDocuments(
	count: number,
	contentPrefix = "document",
): Document[] {
	return Array.from(
		{ length: count },
		(_, index) =>
			new Document({
				pageContent: `${contentPrefix}-${index}-${"x".repeat(16)}`,
				metadata: { id: index },
			}),
	);
}

const embeddings = {
	embedDocuments: async (documents: string[]) =>
		documents.map((content) => [content.length]),
	embedQuery: async (document: string) => [document.length],
};

function createPipeline() {
	const storedRecords: unknown[] = [];

	return createVectorWritePipeline({
		embeddings,
		resolveEmbeddingSpace: async () => "default",
		getCachedRecords: async () => storedRecords.map(() => null),
		putRecords: async (records) => {
			storedRecords.push(...records);
		},
	});
}

export default function register(Suite: Constructor<Suite>) {
	const uniqueDocs = createDocuments(100, "unique-document");
	const duplicateDocs = createDocuments(100, "duplicate-document");
	const uniqueVectors = uniqueDocs.map((doc) => [doc.pageContent.length]);

	return new Promise<void>((resolve) => {
		new Suite()
			.add("pipeline/addDocuments/100-unique", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const pipeline = createPipeline();
					await pipeline.addDocuments(uniqueDocs);
					deferred.resolve();
				},
			})
			.add("pipeline/addDocuments/100-duplicate", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const pipeline = createPipeline();
					await pipeline.addDocuments(duplicateDocs);
					deferred.resolve();
				},
			})
			.add("pipeline/addVectors/100", {
				defer: true,
				fn: async (deferred: Deferred) => {
					const pipeline = createPipeline();
					await pipeline.addVectors(uniqueVectors, uniqueDocs);
					deferred.resolve();
				},
			})
			.on("cycle", (event: any) => {
				console.log(String(event.target));
			})
			.on("complete", () => {
				resolve();
			})
			.run({ async: true });
	});
}
