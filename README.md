# ts-local-embeddings

[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![docs](https://img.shields.io/badge/docs-typescript-blue.svg)](https://aicacia.github.io/ts-local-embeddings/docs/)
[![npm (scoped)](https://img.shields.io/npm/v/@aicacia/local-embeddings)](https://www.npmjs.com/package/@aicacia/local-embeddings)
[![build](https://github.com/aicacia/ts-local-embeddings/workflows/Test/badge.svg)](https://github.com/aicacia/ts-local-embeddings/actions?query=workflow%3ATest)

Browser-first local embedding utilities for:

- Loading a local ONNX embedding model with fallbacks
- Embedding documents in-process or via Web Worker
- Storing vectors in IndexedDB with similarity search and MMR

## Documentation

Hosted API docs: https://aicacia.github.io/ts-local-embeddings/docs/

## Install

```bash
pnpm add @aicacia/local-embeddings @huggingface/transformers @langchain/core
```

## Quick start (Web Worker + IndexedDB)

```ts
import {
  IndexedDBVectorStore,
  WorkerEmbeddings,
} from "@aicacia/local-embeddings";
import { Document } from "@langchain/core/documents";

const embeddings = new WorkerEmbeddings({
  runtime: {
    modelPath: "/models/",
  },
});

const store = new IndexedDBVectorStore(embeddings);

await store.addDocuments([
  new Document({ pageContent: "TypeScript is strongly typed JavaScript" }),
  new Document({
    pageContent: "Transformers can run in the browser with ONNX",
  }),
]);

const matches = await store.similaritySearchWithScore("browser embeddings", 3);
console.log(matches);
```

When finished, terminate worker resources:

```ts
embeddings.terminate();
```

## Runtime options

`loadEmbeddingRuntime` and `WorkerEmbeddings` runtime options support:

- `modelId`: model repo id (default: `onnx-community/embeddinggemma-300m-ONNX`)
- `modelFallbacks`: preferred `dtype` / file fallback order
- `allowRemoteModels`: when `false`, require local files only
- `modelPath`: path passed to transformers as `cache_dir`; in browser apps this is typically a served model base path like `'/models/'` or `'/my-base/models/'`

`WorkerEmbeddings` also supports:

- `onProgress`: callback fired during `embedDocuments` batching with `processedAfterBatch` and `totalDocuments` values for UI progress indicators.

To enable internal debug logs while diagnosing runtime/model loading issues, set:

```ts
(
  globalThis as { __LOCAL_EMBEDDINGS_DEBUG__?: boolean }
).__LOCAL_EMBEDDINGS_DEBUG__ = true;
```

Example `modelFallbacks`:

```ts
const embeddings = new WorkerEmbeddings({
  runtime: {
    modelFallbacks: [
      { dtype: "q4", model_file_name: "model_no_gather" },
      { dtype: "q4" },
      { dtype: "fp16" },
    ],
  },
});
```

## Architecture

The library now uses deep internal boundary modules while keeping the public API unchanged:

- `embeddingPipeline`: owns token limit resolution, adaptive batching, tokenizer/model invocation compatibility, and embedding output validation.
- `workerChannel`: owns worker request ids, pending promise lifecycle, timeout handling, and failure fan-out semantics.
- `vectorWritePipeline`: owns write-path dedup policy, cache reuse, deterministic guard behavior, and record mapping.
- `indexedDbStoreGateway`: owns IndexedDB lifecycle/open-upgrade flow, schema checks, and low-level read/write/query operations.
- `runtimePolicy` + `runtimeLoader`: separates fallback-selection policy from the Hugging Face runtime adapter.

## Local model files

If you run fully local, host model assets under your configured `modelPath`, typically:

```text
<public-or-static-root>/models/onnx-community/<model-id>/onnx/*
```

For SvelteKit, files in `static/` are served at the app base path.

## Development

```bash
pnpm install
pnpm build
pnpm lint
pnpm coverage
pnpm github-pages:dev
```

For running the browser benchmark locally, build the browser assets first:

```bash
pnpm run build:browser
pnpm run benchmark:browser
```
