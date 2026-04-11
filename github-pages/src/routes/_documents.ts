import type { Document } from "@langchain/core/documents";

type CorpusMetadata = {
	id: number;
	source: string;
	topic: string;
};

const topics = [
	"biology",
	"architecture",
	"machine-learning",
	"databases",
	"astronomy",
	"oceanography",
	"history",
	"cooking",
	"literature",
	"physics",
	"chemistry",
	"geography",
] as const;

const templates = [
	"The core idea in {topic} is explained with practical examples and common terminology.",
	"An introductory guide to {topic} often starts with definitions, then moves to real-world applications.",
	"In {topic}, experts compare foundational concepts, trade-offs, and implementation details.",
	"Research notes on {topic} include observations, evidence, and concise conclusions.",
	"A short handbook on {topic} summarizes the most important principles for beginners.",
	"This article on {topic} discusses methods, constraints, and frequent misconceptions.",
	"Advanced discussions of {topic} connect theory with measurable outcomes.",
	"Case studies in {topic} highlight failures, lessons learned, and best practices.",
] as const;

const generatedDocuments: Document<CorpusMetadata>[] = Array.from(
	{ length: 32 },
	(_, index) => {
		const id = index + 1;
		const topic = topics[index % topics.length];
		const template = templates[index % templates.length];
		const sentence = template.replace("{topic}", topic);

		return {
			pageContent: `Document ${id}: ${sentence} Example note ${index + 1} discusses ${topic} in context.`,
			metadata: {
				id,
				source: `generated://${topic}/${id}`,
				topic,
			},
		};
	},
);

export const documents: Document<CorpusMetadata>[] =
	generatedDocuments;
