import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import typescript from "@rollup/plugin-typescript";
import terser from "@rollup/plugin-terser";
import esmImportToUrl from "rollup-plugin-esm-import-to-url";

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
				dir: "browser",
				format: "es",
				sourcemap: true,
				plugins: [terser()],
			},
		],
		plugins: [
			esmImportToUrl({
				imports: {
					tslib: "https://unpkg.com/tslib@2/tslib.es6.js",
				},
			}),
			resolve({ browser: true }),
			commonjs({
				transformMixedEsModules: true,
			}),
			typescript({
				tsconfig: "./tsconfig.json",
				compilerOptions: {
					outDir: "browser",
					declaration: false,
					declarationMap: false,
					declarationDir: undefined,
				},
			}),
		],
	},
];
