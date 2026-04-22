import { defineConfig } from "vite";
import path from "node:path";

export default defineConfig({
	root: path.resolve(__dirname, "tests/playwright/static"),
	optimizeDeps: {
		include: ["benchmark", "lodash", "platform"],
	},
	server: {
		port: 4173,
		strictPort: true,
		fs: {
			// allow serving files from project root so imports from ../../src resolve
			allow: [path.resolve(__dirname)],
		},
	},
	resolve: {
		alias: {
			// optional alias if code imports from `src/*`
			src: path.resolve(__dirname, "src"),
		},
	},
});
