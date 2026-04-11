<script lang="ts">
import "../app.css";
import favicon from "$lib/assets/favicon.svg";

const { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<main class="flex h-svh max-w-svw overflow-hidden">
	{@render children()}
</main>
