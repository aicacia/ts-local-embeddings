<script lang="ts">
import { base } from "$app/paths";
import {
	WorkerEmbeddings,
	IndexedDBVectorStore,
} from "@aicacia/local-embeddings";
import { documents } from "./_documents";
import { onMount } from "svelte";
import type { Document } from "@langchain/core/documents";
import type { WorkerResponseMap } from "@aicacia/local-embeddings";

let vectorStore: IndexedDBVectorStore | null = null;
let workerEmbeddings: WorkerEmbeddings | null = null;

let query = $state("");
let topK = $state(5);
let modelLoading = $state(true);
let indexing = $state(false);
let searching = $state(false);
let error = $state("");
let matches = $state<Array<{ document: Document; score: number }>>([]);
let filteredDocuments = $state<
	Array<{ document: Document; score: number | null }>
>(documents.map((document) => ({ document, score: null })));
let lastSearchDurationMs = $state<number | null>(null);
let indexDurationMs = $state<number | null>(null);
let indexedDocumentCount = $state(0);
let indexedDocumentTotal = $state(documents.length);
let indexProgressPercent = $state(0);

const modelPath = `${base === "/" ? "" : base}/models/`;

function getNowMs(): number {
	if (
		typeof performance !== "undefined" &&
		typeof performance.now === "function"
	) {
		return performance.now();
	}

	return Date.now();
}

function formatDuration(durationMs: number): string {
	if (!Number.isFinite(durationMs) || durationMs < 0) {
		return "0ms";
	}

	if (durationMs < 1000) {
		return `${durationMs.toFixed(2)}ms`;
	}

	const totalSeconds = durationMs / 1000;
	if (totalSeconds < 60) {
		return `${totalSeconds.toFixed(2)}s`;
	}

	const totalMinutes = Math.floor(totalSeconds / 60);
	const remainingSeconds = totalSeconds % 60;
	if (totalMinutes < 60) {
		return `${totalMinutes}m ${remainingSeconds.toFixed(1)}s`;
	}

	const totalHours = Math.floor(totalMinutes / 60);
	const remainingMinutes = totalMinutes % 60;
	return `${totalHours}h ${remainingMinutes}m ${remainingSeconds.toFixed(0)}s`;
}

onMount(() => {
	let disposed = false;

	void (async () => {
		try {
			workerEmbeddings = new WorkerEmbeddings({
				runtime: {
					modelPath,
				},
				onProgress: (progress: WorkerResponseMap["progress"]) => {
					if (progress.requestType !== "embedDocuments") {
						return;
					}

					const totalDocuments =
						progress.event.totalDocuments > 0
							? progress.event.totalDocuments
							: documents.length;

					indexedDocumentCount = progress.event.processedAfterBatch;
					indexedDocumentTotal = totalDocuments;
					if (totalDocuments > 0) {
						indexProgressPercent =
							(progress.event.processedAfterBatch / totalDocuments) * 100;
					} else {
						indexProgressPercent = null;
					}
				},
			});

			if (disposed) {
				workerEmbeddings.terminate();
				return;
			}

			vectorStore = new IndexedDBVectorStore(workerEmbeddings);

			modelLoading = false;
			indexing = true;
			indexedDocumentCount = 0;
			indexedDocumentTotal = documents.length;
			indexProgressPercent = 0;
			const indexingStartedAt = getNowMs();
			await vectorStore.addDocuments(documents);
			indexDurationMs = getNowMs() - indexingStartedAt;
			indexedDocumentCount = documents.length;
			indexedDocumentTotal = documents.length;
			indexProgressPercent = 100;
		} catch (err) {
			error =
				err instanceof Error
					? err.message
					: "Failed to initialize local embeddings.";
		} finally {
			modelLoading = false;
			indexing = false;
		}
	})();

	return () => {
		disposed = true;
		vectorStore = null;
		workerEmbeddings?.terminate();
		workerEmbeddings = null;
	};
});

async function runSearch() {
	if (!vectorStore) {
		error = "Vector store is not ready yet.";
		return;
	}

	if (!query.trim()) {
		matches = [];
		filteredDocuments = documents.map((document) => ({
			document,
			score: null,
		}));
		lastSearchDurationMs = null;
		return;
	}

	topK = Math.min(20, Math.max(1, Math.round(topK || 1)));

	searching = true;
	error = "";
	lastSearchDurationMs = null;

	const startedAt = getNowMs();
	try {
		const results = await vectorStore.similaritySearchWithScore(query, topK);
		matches = results.map(([document, score]) => ({ document, score }));
		filteredDocuments = matches;
		lastSearchDurationMs = getNowMs() - startedAt;
	} catch (err) {
		error = err instanceof Error ? err.message : "Search failed.";
	} finally {
		searching = false;
	}
}
</script>

<div class="mx-auto flex h-full min-h-0 max-w-5xl flex-1 flex-col px-4 py-8">
	<h1 class="text-3xl font-bold tracking-tight text-slate-900">Local Embeddings Search</h1>
	<p class="mt-1 text-slate-600">
		{#if modelLoading}
			Preparing {documents.length} documents for indexing
		{:else if indexing}
			Indexing {documents.length} documents in memory
			({indexedDocumentCount}/{indexedDocumentTotal}, {indexProgressPercent.toFixed(1)}%)
		{:else}
			{documents.length} documents indexed in memory
			{#if indexDurationMs !== null}
				in {formatDuration(indexDurationMs)}
			{/if}
		{/if}
	</p>

	<div class="mt-5 grid items-end gap-3 md:grid-cols-[1fr_auto_auto]">
		<input
			type="text"
			class="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 transition outline-none focus:border-orange-500 focus:ring-2 focus:ring-orange-200 disabled:cursor-not-allowed disabled:bg-slate-100"
			bind:value={query}
			placeholder="Search for related content..."
			onkeydown={(event) => event.key === 'Enter' && runSearch()}
			disabled={modelLoading || indexing}
		/>
		<label class="grid gap-1 text-sm text-slate-700">
			Top K
			<input
				type="number"
				min="1"
				max="20"
				class="w-20 rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 transition outline-none focus:border-orange-500 focus:ring-2 focus:ring-orange-200 disabled:cursor-not-allowed disabled:bg-slate-100"
				bind:value={topK}
				disabled={modelLoading || indexing}
			/>
		</label>
		<button
			onclick={runSearch}
			disabled={modelLoading || indexing || searching}
			class="rounded-lg bg-orange-600 px-4 py-2 font-medium text-white transition hover:bg-orange-500 disabled:cursor-not-allowed disabled:bg-slate-400"
		>
			Search
		</button>
	</div>

	{#if modelLoading}
		<p class="mt-4 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-sky-800">
			Loading embedding model. This can take a bit on first load.
		</p>
	{:else if indexing}
		<p class="mt-4 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-sky-800">
			Indexing documents. This can take a bit on first load.
			<span class="mt-1 block text-sky-900">
				Indexed {indexedDocumentCount} of {indexedDocumentTotal} ({indexProgressPercent.toFixed(1)}%)
			</span>
		</p>
	{:else if error}
		<p class="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700">{error}</p>
	{/if}

	{#if lastSearchDurationMs !== null}
		<p class="mt-6 text-sm text-slate-600">Results in {formatDuration(lastSearchDurationMs)}.</p>
	{/if}

	<section class="mt-8 flex min-h-0 flex-1 flex-col">
		<h2 class="text-xl font-semibold text-slate-900">Documents</h2>
		{#if !modelLoading && !indexing && query.trim() && filteredDocuments.length === 0}
			<p class="mt-3 text-slate-600">No matches for this search query.</p>
		{/if}
		<ul
			class="mt-3 grid min-h-0 flex-1 gap-2 overflow-y-auto rounded-xl border border-slate-200 p-2"
		>
			{#each filteredDocuments as item (String(item.document.metadata?.id ?? item.document.pageContent))}
				<li class="grid grid-cols-[auto_1fr] gap-2 rounded-lg border border-slate-200 bg-white p-3">
					<strong class="text-slate-500">#{String(item.document.metadata?.id ?? '-')}</strong>
					<div class="min-w-0">
						<span class="text-slate-900">{item.document.pageContent}</span>
						{#if item.score !== null}
							<small class="mt-1 block text-slate-500">
								source: {String(item.document.metadata?.source ?? 'unknown')} | score:
								{item.score.toFixed(4)}
							</small>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	</section>
</div>
