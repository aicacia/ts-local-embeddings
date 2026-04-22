/// <reference path="./benchmark-globals.d.ts" />
import { test, expect, type TestInfo } from "@playwright/test";
import { promises as fs } from "node:fs";
import path from "node:path";
import type { BenchmarkStatus, BenchmarkResult } from "./benchmark-globals";

test("imports benchmark harness and runs browser benchmark", async ({
	page,
}, testInfo: TestInfo) => {
	// Create a Chromium CDP session for CPU profiling (optional).
	let cdpClient: any = null;
	try {
		// newCDPSession is Chromium-only; if unavailable, profiling will be skipped.
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		cdpClient = await page.context().newCDPSession(page as any);
	} catch (e) {
		cdpClient = null;
	}

	// Forward browser console messages to Node's stdout so benchmark logs appear
	// in CI/terminal output as well. Also listen for suite-level profiling
	// markers emitted by the benchmark runner and capture a CPU profile for
	// each suite via the Chromium CDP Profiler.
	let profilerActive = false;
	let currentSuiteName: string | null = null;
	let profLock: Promise<void> = Promise.resolve();
	page.on("console", (msg) => {
		// queue handling to ensure serial CDP operations
		profLock = profLock
			.then(async () => {
				const text = String(msg.text());
				try {
					console.log(`[browser:${msg.type()}] ${text}`);
				} catch (e) {
					console.log(`[browser] ${String(msg)}`);
				}

				// Ensure CDP client is available; create or recreate if necessary.
				if (!cdpClient) {
					try {
						// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
						cdpClient = await page.context().newCDPSession(page as any);
					} catch (err) {
						console.error("[cpuprofile] cannot create CDP session", err);
						return;
					}
				}

				try {
					const match = text.match(
						/^\[cpuprofile:(start|stop)\]\s*suite:(.+)$/,
					);
					if (!match) return;
					const action = match[1] as "start" | "stop";
					const rawName = match[2].trim();
					const suiteName = rawName.replace(/[^a-zA-Z0-9_.-]/g, "_");

					if (action === "start") {
						if (!profilerActive) {
							try {
								await cdpClient.send("Profiler.enable");
								await cdpClient.send("Profiler.start");
								profilerActive = true;
								currentSuiteName = suiteName;
								console.log(`[cpuprofile] started ${suiteName}`);
							} catch (err) {
								// If the CDP session was closed mid-run, try to recreate once
								console.error(
									"[cpuprofile] start failed, attempting recreate",
									err,
								);
								try {
									// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
									cdpClient = await page.context().newCDPSession(page as any);
									await cdpClient.send("Profiler.enable");
									await cdpClient.send("Profiler.start");
									profilerActive = true;
									currentSuiteName = suiteName;
									console.log(`[cpuprofile] started ${suiteName}`);
								} catch (err2) {
									console.error("[cpuprofile] start retry failed", err2);
								}
							}
							return;
						}

						// If a different suite starts while profiler is active, stop the
						// current profile and save it, then start a new profiler for the
						// incoming suite.
						if (profilerActive && currentSuiteName !== suiteName) {
							try {
								const res = await cdpClient.send("Profiler.stop");
								try {
									await cdpClient.send("Profiler.disable");
								} catch (_) {}
								// discard session after stopping
								try {
									cdpClient = null;
								} catch (_) {}
								const profile = res.profile as unknown;
								const prevName = currentSuiteName || "profile";
								const outDir = path.resolve(process.cwd(), "test-results");
								await fs.mkdir(outDir, { recursive: true });
								const filePath = path.join(
									outDir,
									`cpu-profile-${prevName}-${Date.now()}.cpuprofile`,
								);
								await fs.writeFile(filePath, JSON.stringify(profile, null, 2));
								await testInfo.attach(`cpu-profile-${prevName}.cpuprofile`, {
									body: JSON.stringify(profile, null, 2),
									contentType: "application/json",
								});
								console.log(`[cpuprofile] saved ${filePath}`);
							} catch (err) {
								console.error("[cpuprofile] rollover stop failed", err);
							}

							// start new profiler for new suite
							try {
								await cdpClient.send("Profiler.enable");
								await cdpClient.send("Profiler.start");
								profilerActive = true;
								currentSuiteName = suiteName;
								console.log(`[cpuprofile] started ${suiteName}`);
							} catch (err) {
								console.error("[cpuprofile] restart failed", err);
							}
						}
					}

					if (action === "stop") {
						if (!profilerActive) {
							console.log("[cpuprofile] stop ignored, profiler not active");
							return;
						}

						try {
							const res = await cdpClient.send("Profiler.stop");
							try {
								await cdpClient.send("Profiler.disable");
							} catch (_) {}
							// discard session after stopping so we recreate cleanly next time
							try {
								cdpClient = null;
							} catch (_) {}
							profilerActive = false;
							const profile = res.profile as unknown;
							const nameToUse = suiteName || currentSuiteName || "profile";

							// write to test-results/
							const outDir = path.resolve(process.cwd(), "test-results");
							await fs.mkdir(outDir, { recursive: true });
							const filePath = path.join(
								outDir,
								`cpu-profile-${nameToUse}-${Date.now()}.cpuprofile`,
							);
							await fs.writeFile(filePath, JSON.stringify(profile, null, 2));

							// Attach to Playwright test artifacts
							await testInfo.attach(`cpu-profile-${nameToUse}.cpuprofile`, {
								body: JSON.stringify(profile, null, 2),
								contentType: "application/json",
							});

							console.log(`[cpuprofile] saved ${filePath}`);
						} catch (err) {
							console.error("[cpuprofile] stop failed", err);
						}
					}
				} catch (err) {
					// swallow errors from profiling handler so benchmark-runner continues
				}
			})
			.catch((err) => {
				// log lock-chain errors but don't disrupt the watcher
				console.error("[cpuprofile] handler error", err);
			});
	});

	const suites = [
		"embeddingPipeline",
		"WorkerEmbeddings",
		"indexedDbStoreGateway",
		"IndexedDBVectorStore",
		"vectorWritePipeline",
	];

	const outDir = path.resolve(process.cwd(), "test-results");
	await fs.mkdir(outDir, { recursive: true });

	for (const suite of suites) {
		await page.goto(
			`http://127.0.0.1:4173/benchmark.html?suite=${encodeURIComponent(suite)}`,
		);
		await page.waitForFunction(
			"window.__benchmarkStatus === 'complete' || window.__benchmarkStatus === 'error'",
			null,
			{ timeout: 300000 },
		);

		const status = await page.evaluate<BenchmarkStatus | null>(
			() => window.__benchmarkStatus ?? null,
		);
		const results = await page.evaluate<BenchmarkResult[] | null>(
			() => window.__benchmarkResults,
		);
		const error = await page.evaluate<string | null>(
			() => window.__benchmarkError ?? null,
		);

		if (status !== "complete") {
			throw new Error(
				`Benchmark failed in browser for suite=${suite}: ${String(error)} - results: ${String(results)}`,
			);
		}

		// Attach JSON to Playwright test artifacts per-suite
		await testInfo.attach(`benchmark-results-${suite}.json`, {
			body: JSON.stringify({ status, results, error }, null, 2),
			contentType: "application/json",
		});

		// Also write a CI-friendly artifact to `test-results/`
		await fs.writeFile(
			path.join(outDir, `browser-benchmark-${suite}-${Date.now()}.json`),
			JSON.stringify({ status, results, error }, null, 2),
		);
	}

	// basic assert to keep existing test expectation
	expect(true).toBeTruthy();
});
