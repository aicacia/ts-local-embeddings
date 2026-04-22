import test from "tape";
import * as BenchmarkModule from "benchmark";

const Benchmark: any = (BenchmarkModule as any).default ?? BenchmarkModule;
let RealBenchmark: any | undefined;
try {
	if (typeof (Benchmark as any)?.runInContext === "function") {
		RealBenchmark = (Benchmark as any).runInContext();
	}
} catch (e) {
	// ignore
}
RealBenchmark =
	RealBenchmark ??
	(Benchmark &&
		(Benchmark.Benchmark ?? (Benchmark as any).default ?? Benchmark));
const Suite: any = RealBenchmark?.Suite ?? (Benchmark as any).Suite;

import embedding from "../../src/pipeline/embeddingPipeline.suite.js";
import worker from "../../src/worker/WorkerEmbeddings.suite.js";
import indexedDb from "../../src/store/indexedDbStoreGateway.suite.js";
import vectorStore from "../../src/store/IndexedDBVectorStore.suite.js";
import vectorWrite from "../../src/store/vectorWritePipeline.suite.js";

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
