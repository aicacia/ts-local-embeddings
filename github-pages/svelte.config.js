import path from 'path';
import { fileURLToPath } from 'url';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const dev = process.argv.includes('dev');
const base = dev ? '' : (process.env.BASE_PATH ?? '');
const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter(),
		paths: {
			base
		},
		alias: {
			'@aicacia/local-embeddings': path.resolve(__dirname, '../src/index.ts')
		}
	}
};

export default config;
