import * as BenchmarkBarrel from "benchmark";
import test from "tape";

type BenchmarkModule = typeof Benchmark;

const Benchmark = BenchmarkBarrel.default ?? BenchmarkBarrel;
let RealBenchmark: BenchmarkModule | undefined;
try {
	if (typeof Benchmark?.runInContext === "function") {
		// biome-ignore lint/suspicious/noExplicitAny: benchmark is a nightmare
		RealBenchmark = (Benchmark as any).runInContext();
	}
} catch {
	// ignore
}
RealBenchmark =
	RealBenchmark ??
	(Benchmark &&
		// biome-ignore lint/suspicious/noExplicitAny: benchmark is a nightmare
		((Benchmark as any).Benchmark ?? (Benchmark as any).default ?? Benchmark));
const Suite = (RealBenchmark?.Suite ?? Benchmark.Suite) as unknown as new (
	...args: unknown[]
) => BenchmarkBarrel.Suite;

import embedding from "../../src/pipeline/embeddingPipeline.suite.js";
import vectorStore from "../../src/store/IndexedDBVectorStore.suite.js";
import indexedDb from "../../src/store/indexedDbStoreGateway.suite.js";
import vectorWrite from "../../src/store/vectorWritePipeline.suite.js";
import worker from "../../src/worker/WorkerEmbeddings.suite.js";

test("bench/suites", async (t) => {
	try {
		await embedding(Suite);
		t.pass("embeddingPipeline suite complete");
		await worker(Suite);
		t.pass("WorkerEmbeddings suite complete");
		await indexedDb(Suite);
		t.pass("indexedDbStoreGateway suite complete");
		await vectorStore(Suite);
		t.pass("IndexedDBVectorStore suite complete");
		await vectorWrite(Suite);
		t.pass("vectorWritePipeline suite complete");
	} catch (err) {
		t.fail(String(err));
	} finally {
		t.end();
	}
});
