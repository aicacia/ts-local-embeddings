import test from "tape";
import { createDocument } from "../utils/documentUtils.js";
import {
	createContentHashGetter,
	fallbackHash,
	// normalizeEmbeddingForStorage,
	// computeEmbeddingNorm,
	// intentionally test through the public mapper for alignment with runtime behavior
	mapStoredVectorRecord,
	stableStringify,
} from "./vectorWritePipelineUtils.js";

test("createContentHashGetter returns stable cached results and supports custom hashers", async (assert) => {
	const getContentHash = createContentHashGetter({ contentHashCacheMax: 2 });

	const firstHash = await getContentHash("hello");
	const secondHash = await getContentHash("hello");

	assert.equal(
		firstHash,
		secondHash,
		"returns the same hash for repeated content",
	);

	const custom = createContentHashGetter({
		contentHasher: (content) => `${content}-custom`,
		contentHashCacheMax: 1,
	});

	assert.equal(
		await custom("a"),
		"a-custom",
		"uses the provided content hasher",
	);
	assert.equal(await custom("a"), "a-custom", "caches custom hash results");
	assert.end();
});

test("stableStringify produces deterministic output for objects and nested arrays", (assert) => {
	const value = { b: 2, a: [1, { c: 3 }] };
	assert.equal(
		stableStringify(value),
		`{\n"a":[1,{\n"c":3}],"b":2}`,
		"sorts object keys and stringifies nested values deterministically",
	);
	assert.end();
});

test("mapStoredVectorRecord converts large number arrays to Float32Array and computes norm", (assert) => {
	const embedding = Array.from({ length: 16 }, (_, index) => index + 1);
	const document = createDocument("test-content", { id: "1" });
	const record = mapStoredVectorRecord({
		id: "1",
		document,
		embeddingSpace: "space-1",
		contentHash: "hash-1",
		embedding,
	});

	assert.ok(
		record.embedding instanceof Float32Array,
		"large arrays are converted to Float32Array for storage",
	);
	assert.equal(
		record.embeddingNorm,
		Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0)),
		"computes embedding norm for stored vectors",
	);
	assert.equal(
		record.cacheKey,
		"space-1:hash-1",
		"builds the expected cache key",
	);
	assert.equal(
		fallbackHash("abc").startsWith("fallback-"),
		true,
		"fallbackHash returns a fallback identifier",
	);
	assert.end();
});
