import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import terser from "@rollup/plugin-terser";
import esmImportToUrl from "rollup-plugin-esm-import-to-url";
import typescript2 from "rollup-plugin-typescript2";

export default [
	{
		input: "src/index.ts",
		onwarn(warning, warn) {
			if (
				warning.code === "CIRCULAR_DEPENDENCY" &&
				typeof warning.message === "string" &&
				warning.message.includes("node_modules")
			) {
				return;
			}
			warn(warning);
		},
		output: [
			{
				dir: "dist/browser",
				format: "es",
				sourcemap: true,
				entryFileNames: "[name].js",
				plugins: [terser()],
			},
		],
		plugins: [
			typescript2({
				tsconfigOverride: {
					compilerOptions: {
						module: "ES2020",
						moduleResolution: "bundler",
					},
				},
			}),
			esmImportToUrl({
				imports: {
					tslib: "https://unpkg.com/tslib@2/tslib.es6.js",
				},
			}),
			resolve({ browser: true }),
			commonjs({
				transformMixedEsModules: true,
			}),
		],
	},
];
