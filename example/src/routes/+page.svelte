<script lang="ts">
import { base } from "$app/paths";
import {
	WorkerEmbeddings,
	IndexedDBVectorStore,
} from "@aicacia/local-embeddings";
import { documents } from "./_documents";
import { onMount } from "svelte";
import type { Document } from "@langchain/core/documents";

let vectorStore: IndexedDBVectorStore | null = null;
let workerEmbeddings: WorkerEmbeddings | null = null;

let query = $state("");
let topK = $state(5);
let modelLoading = $state(true);
let indexing = $state(false);
let searching = $state(false);
let error = $state("");
let matches = $state<Array<{ document: Document; score: number }>>([]);

const modelPath = `${base === "/" ? "" : base}/models/`;

onMount(() => {
	let disposed = false;

	void (async () => {
		try {
			workerEmbeddings = new WorkerEmbeddings({
				runtime: {
					modelPath,
				},
			});

			if (disposed) {
				workerEmbeddings.terminate();
				return;
			}

			vectorStore = new IndexedDBVectorStore(workerEmbeddings);

			modelLoading = false;
			indexing = true;
			await vectorStore.addDocuments(documents);
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
		return;
	}

	topK = Math.min(20, Math.max(1, Math.round(topK || 1)));

	searching = true;
	error = "";

	try {
		const results = await vectorStore.similaritySearchWithScore(query, topK);
		matches = results.map(([document, score]) => ({ document, score }));
	} catch (err) {
		error = err instanceof Error ? err.message : "Search failed.";
	} finally {
		searching = false;
	}
}
</script>

<div class="mx-auto flex h-full min-h-0 max-w-5xl flex-1 flex-col px-4 py-8">
	<h1 class="text-3xl font-bold tracking-tight text-slate-900">Local Embeddings Search</h1>
	<p class="mt-1 text-slate-600">{documents.length} documents indexed in memory</p>

	<div class="mt-5 grid items-end gap-3 md:grid-cols-[1fr_auto_auto]">
		<input
			type="text"
			class="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 outline-none transition focus:border-orange-500 focus:ring-2 focus:ring-orange-200 disabled:cursor-not-allowed disabled:bg-slate-100"
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
				class="w-20 rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 outline-none transition focus:border-orange-500 focus:ring-2 focus:ring-orange-200 disabled:cursor-not-allowed disabled:bg-slate-100"
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
		</p>
	{:else if error}
		<p class="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700">{error}</p>
	{/if}

	<section class="mt-8">
		<h2 class="text-xl font-semibold text-slate-900">Best matches</h2>
		{#if !modelLoading && !indexing && matches.length === 0}
			<p class="mt-3 text-slate-600">No matches yet. Try a search query.</p>
		{:else}
			<ul class="mt-3 grid gap-2">
				{#each matches as item (String(item.document.metadata?.id ?? item.document.pageContent))}
					<li class="rounded-lg border border-slate-200 bg-white p-3">
						<p class="text-slate-900">{item.document.pageContent}</p>
						<small class="mt-2 block text-slate-500">
							source: {String(item.document.metadata?.source ?? 'unknown')} | score:
							{item.score.toFixed(4)}
						</small>
					</li>
				{/each}
			</ul>
		{/if}
	</section>

	<section class="mt-8 flex min-h-0 flex-1 flex-col">
		<h2 class="text-xl font-semibold text-slate-900">All documents</h2>
		<ul class="mt-3 grid min-h-0 flex-1 gap-2 overflow-y-auto rounded-xl border border-slate-200 p-2">
			{#each documents as document (String(document.metadata?.id ?? document.pageContent))}
				<li class="grid grid-cols-[auto_1fr] gap-2 rounded-lg border border-slate-200 bg-white p-3">
					<strong class="text-slate-500">#{String(document.metadata?.id ?? '-')}</strong>
					<span class="text-slate-900">{document.pageContent}</span>
				</li>
			{/each}
		</ul>
	</section>
</div>