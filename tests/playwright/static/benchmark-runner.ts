import * as BenchmarkModule from "benchmark";
import _ from "lodash";
const Benchmark: any = (BenchmarkModule as any).default ?? BenchmarkModule;
console.log(
	"[benchmark-runner] Benchmark export keys:",
	Object.keys(Benchmark || {}),
);

declare global {
	interface Window {
		__benchmarkStatus: "pending" | "complete" | "error" | null;
		__benchmarkResults: Array<{
			name: string;
			iterations: number;
			duration: number;
			hz: number;
		}> | null;
		__benchmarkError: string | null;
	}
}

// initial global status
window.__benchmarkStatus = "pending";
window.__benchmarkResults = null;
window.__benchmarkError = null;

// Polyfill `process` for browser environment so Node-oriented deps (tape, etc.)
// that expect `process` don't immediately throw. Keep minimal to avoid
// changing behavior elsewhere.
if (typeof (window as any).process === "undefined") {
	(window as any).process = {
		env: {},
		nextTick: (cb: Function) => Promise.resolve().then(() => cb()),
		stdout: { write: () => {} },
		stderr: { write: () => {} },
	};
}

// Provide a minimal `test` shim for benchmark modules that expect `test(name, fn)`
if (typeof (window as any).test === "undefined") {
	(window as any).test = (name: string, fn: any) => {
		try {
			return fn({ end: () => {} });
		} catch (err) {
			console.error(err);
		}
	};
}

// Expose the Benchmark object to imported modules that may reference it.
try {
	(window as any).Benchmark = Benchmark;
	(globalThis as any).Benchmark = Benchmark;
} catch (e) {
	// ignore
}

// Ensure prebundled lodash is available on the global root for Benchmark.runInContext
try {
	(globalThis as any)._ = _;
	(window as any)._ = _;
} catch (e) {
	// ignore
}

const outputEl =
	document.getElementById("output") ||
	(() => {
		const el = document.createElement("pre");
		el.id = "output";
		el.textContent = "Starting browser benchmark...";
		document.body.appendChild(el);
		return el;
	})();

function appendOutput(msg: string) {
	try {
		outputEl.textContent += `\n${msg}`;
	} catch (e) {
		// ignore
	}
}

const originalConsoleLog = console.log.bind(console);
console.log = (...args: unknown[]) => {
	try {
		originalConsoleLog(...args);
	} catch (e) {
		// ignore
	}
	try {
		appendOutput(args.map((a) => String(a)).join(" "));
	} catch (e) {
		// ignore
	}
};

const originalConsoleError = console.error.bind(console);
console.error = (...args: unknown[]) => {
	try {
		originalConsoleError(...args);
	} catch (e) {
		// ignore
	}
	try {
		appendOutput(`[error] ${args.map((a) => String(a)).join(" ")}`);
	} catch (e) {
		// ignore
	}
};

type BenchResult = {
	name: string;
	iterations: number;
	duration: number; // ms
	hz: number;
};

const results: BenchResult[] = [];
let activeSuites = 0;
let finalized = false;

function finalize() {
	if (finalized) return;
	finalized = true;
	window.__benchmarkResults = results;
	window.__benchmarkStatus = "complete";
	appendOutput("Benchmark complete");
}

function findSuite(obj: any, depth = 0, seen = new Set<any>()): any {
	if (!obj || depth > 6 || seen.has(obj)) return undefined;
	seen.add(obj);
	try {
		if (obj.Suite) return obj.Suite;
	} catch (e) {
		// ignore
	}
	for (const k of Object.keys(obj || {})) {
		try {
			const v = (obj as any)[k];
			if (v && (typeof v === "object" || typeof v === "function")) {
				const s = findSuite(v, depth + 1, seen);
				if (s) return s;
			}
		} catch (e) {
			// ignore
		}
	}
	return undefined;
}

console.log(
	"[benchmark-runner] attempting to resolve Suite constructor from Benchmark export",
);
// Wrap `Benchmark.runInContext` so any calls (by this runner or by imported
// benchmark modules) return a `Benchmark` object whose `Suite.prototype.run`
// has been patched to notify this runner about completion and collect results.
function patchSuitePrototype(SuiteCtor: any) {
	try {
		if (!SuiteCtor || SuiteCtor.prototype.__patchedForBenchmarkRunner) return;
		SuiteCtor.prototype.__patchedForBenchmarkRunner = true;
		const originalRun = SuiteCtor.prototype.run;
		SuiteCtor.prototype.run = function (...args: unknown[]) {
			activeSuites++;
			console.log(
				"[benchmark-runner] Suite.run invoked, activeSuites ->",
				activeSuites,
			);
			this.on("complete", () => {
				try {
					console.log(
						"[benchmark-runner] Suite.complete event fired, activeSuites (before) ->",
						activeSuites,
					);
					this.forEach((bench: any) => {
						try {
							const iterations =
								bench?.stats?.sample?.length ?? bench?.count ?? 0;
							const duration = bench?.times?.elapsed
								? bench.times.elapsed * 1000
								: 0;
							const hz = bench?.hz ?? 0;
							results.push({
								name: bench?.name ?? "",
								iterations,
								duration,
								hz,
							});
						} catch (err) {
							console.error(err);
						}
					});
				} catch (err) {
					console.error(err);
					window.__benchmarkError = String(err);
					window.__benchmarkStatus = "error";
				} finally {
					activeSuites--;
					console.log(
						"[benchmark-runner] Suite.complete handler finished, activeSuites ->",
						activeSuites,
					);
					if (activeSuites === 0) finalize();
				}
			});
			// @ts-expect-error
			return originalRun.apply(this, args);
		};
	} catch (e) {
		// ignore
	}
	// Attach per-benchmark start/complete listeners so we can emit precise
	// profiling markers for Playwright to pick up. This avoids embedding
	// markers inside suites (which may run the benchmark function many times
	// during sampling) and gives a single start/stop pair per benchmark run.
	// No per-benchmark instrumentation here. Suite-level profiling markers
	// are emitted around each suite import in `loadSuites()` instead.
}

// We intentionally do not wrap `Benchmark.runInContext` here.
// Suites are expected to be invoked with the `Suite` constructor passed
// in by the runner (`register(Suite)`), so wrapping runInContext is
// unnecessary and can cause surprising behavior when suites rely only on
// the provided `Suite`.

// Resolve the Suite constructor once and patch it. Use the resolved
// constructor when invoking each suite's `register(Suite)` so the same
// constructor is passed everywhere (avoids `Suite is not a constructor`).
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
const ResolvedSuite =
	RealBenchmark?.Suite ??
	findSuite(Benchmark) ??
	findSuite((Benchmark as any).default);
patchSuitePrototype(ResolvedSuite);
console.log("[benchmark-runner] Suite patching installed");
console.log(
	"[benchmark-runner] ResolvedSuite:",
	typeof ResolvedSuite === "function"
		? ResolvedSuite.name || "<anonymous>"
		: ResolvedSuite,
);

// dynamically import benchmark suites after we've patched Benchmark
async function loadSuites() {
	try {
		console.log("[benchmark-runner] loading suites...");
		// expected number of suites we will import (adjust if you change the list)
		window.__expectedSuites = 5;
		window.__completedSuites = 0;
		window.__benchmarkResults = window.__benchmarkResults || [];
		const params = new URL(window.location.href).searchParams;
		const selectedSuite = params.get("suite");
		// import suites (relative to this file) sequentially so we can see progress
		if (!selectedSuite || selectedSuite === "embeddingPipeline") {
			try {
				console.log(
					"[benchmark-runner] importing embeddingPipeline -> ../../../src/pipeline/embeddingPipeline.suite.ts",
				);
				const mod = await import(
					"../../../src/pipeline/embeddingPipeline.suite.ts"
				);
				if (mod && typeof mod.default === "function") {
					try {
						console.log("[cpuprofile:start] suite:embeddingPipeline");
					} catch (_e) {}
					await Promise.resolve(mod.default(ResolvedSuite));
					try {
						console.log("[cpuprofile:stop] suite:embeddingPipeline");
					} catch (_e) {}
				}
				console.log("[benchmark-runner] imported embeddingPipeline");
			} catch (err) {
				console.error(
					"[benchmark-runner] import failed: embeddingPipeline",
					err as Error,
				);
				window.__benchmarkError = String(err);
				window.__benchmarkStatus = "error";
				return;
			}
		}

		if (!selectedSuite || selectedSuite === "WorkerEmbeddings") {
			try {
				console.log(
					"[benchmark-runner] importing WorkerEmbeddings -> ../../../src/worker/WorkerEmbeddings.suite.ts",
				);
				const mod = await import(
					"../../../src/worker/WorkerEmbeddings.suite.ts"
				);
				if (mod && typeof mod.default === "function") {
					try {
						console.log("[cpuprofile:start] suite:WorkerEmbeddings");
					} catch (_e) {}
					await Promise.resolve(mod.default(ResolvedSuite));
					try {
						console.log("[cpuprofile:stop] suite:WorkerEmbeddings");
					} catch (_e) {}
				}
				console.log("[benchmark-runner] imported WorkerEmbeddings");
			} catch (err) {
				console.error(
					"[benchmark-runner] import failed: WorkerEmbeddings",
					err as Error,
				);
				window.__benchmarkError = String(err);
				window.__benchmarkStatus = "error";
				return;
			}
		}

		if (!selectedSuite || selectedSuite === "indexedDbStoreGateway") {
			try {
				console.log(
					"[benchmark-runner] importing indexedDbStoreGateway -> ../../../src/store/indexedDbStoreGateway.suite.ts",
				);
				const mod = await import(
					"../../../src/store/indexedDbStoreGateway.suite.ts"
				);
				if (mod && typeof mod.default === "function") {
					try {
						console.log("[cpuprofile:start] suite:indexedDbStoreGateway");
					} catch (_e) {}
					await Promise.resolve(mod.default(ResolvedSuite));
					try {
						console.log("[cpuprofile:stop] suite:indexedDbStoreGateway");
					} catch (_e) {}
				}
				console.log("[benchmark-runner] imported indexedDbStoreGateway");
			} catch (err) {
				console.error(
					"[benchmark-runner] import failed: indexedDbStoreGateway",
					err as Error,
				);
				window.__benchmarkError = String(err);
				window.__benchmarkStatus = "error";
				return;
			}
		}

		if (!selectedSuite || selectedSuite === "IndexedDBVectorStore") {
			try {
				console.log(
					"[benchmark-runner] importing IndexedDBVectorStore -> ../../../src/store/IndexedDBVectorStore.suite.ts",
				);
				const mod = await import(
					"../../../src/store/IndexedDBVectorStore.suite.ts"
				);
				if (mod && typeof mod.default === "function") {
					try {
						console.log("[cpuprofile:start] suite:IndexedDBVectorStore");
					} catch (_e) {}
					await Promise.resolve(mod.default(ResolvedSuite));
					try {
						console.log("[cpuprofile:stop] suite:IndexedDBVectorStore");
					} catch (_e) {}
				}
				console.log("[benchmark-runner] imported IndexedDBVectorStore");
			} catch (err) {
				console.error(
					"[benchmark-runner] import failed: IndexedDBVectorStore",
					err as Error,
				);
				window.__benchmarkError = String(err);
				window.__benchmarkStatus = "error";
				return;
			}
		}

		if (!selectedSuite || selectedSuite === "vectorWritePipeline") {
			try {
				console.log(
					"[benchmark-runner] importing vectorWritePipeline -> ../../../src/store/vectorWritePipeline.suite.ts",
				);
				const mod = await import(
					"../../../src/store/vectorWritePipeline.suite.ts"
				);
				if (mod && typeof mod.default === "function") {
					try {
						console.log("[cpuprofile:start] suite:vectorWritePipeline");
					} catch (_e) {}
					await Promise.resolve(mod.default(ResolvedSuite));
					try {
						console.log("[cpuprofile:stop] suite:vectorWritePipeline");
					} catch (_e) {}
				}
				console.log("[benchmark-runner] imported vectorWritePipeline");
			} catch (err) {
				console.error(
					"[benchmark-runner] import failed: vectorWritePipeline",
					err as Error,
				);
				window.__benchmarkError = String(err);
				window.__benchmarkStatus = "error";
				return;
			}
		}

		console.log("[benchmark-runner] suites loaded");
	} catch (err) {
		console.error(err as Error);
		window.__benchmarkError = String(err);
		window.__benchmarkStatus = "error";
	}
}

// start
loadSuites();

// safety timeout -- mark error if not completed in time
setTimeout(() => {
	if (
		window.__benchmarkStatus !== "complete" &&
		window.__benchmarkStatus !== "error"
	) {
		window.__benchmarkStatus = "error";
		window.__benchmarkError = window.__benchmarkError ?? "benchmark timeout";
		appendOutput("Benchmark timed out");
	}
}, 110000);
