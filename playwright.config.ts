import { defineConfig } from "@playwright/test";

export default defineConfig({
	testDir: "tests/playwright",
	timeout: 600000,
	expect: {
		timeout: 30000,
	},
	webServer: {
		// Use Vite dev server for benchmarks so workers resolve correctly
		command: "pnpm run benchmark:dev",
		port: 4173,
		reuseExistingServer: true,
	},
	use: {
		baseURL: "http://127.0.0.1:4173",
		headless: true,
		ignoreHTTPSErrors: true,
	},
});
