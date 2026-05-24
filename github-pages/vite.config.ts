import path from "node:path";
import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import devtoolsJson from "vite-plugin-devtools-json";

const libraryRoot = new URL("..", import.meta.url).pathname;

const alias = [
	{
		find: /^@aicacia\/local-embeddings$/,
		replacement: path.resolve(libraryRoot, "src/index.ts"),
	},
];

export default defineConfig({
	plugins: [tailwindcss(), sveltekit(), devtoolsJson()],
	resolve: {
		alias,
	},
	server: {
		fs: {
			allow: [libraryRoot],
		},
	},
});
