export type BenchmarkStatus = "pending" | "complete" | "error";

export interface BenchmarkResult {
	name: string;
	iterations: number;
	duration: number;
	hz: number;
}

declare global {
	interface Window {
		__benchmarkStatus: BenchmarkStatus | null;
		__benchmarkResults: BenchmarkResult[] | null;
		__benchmarkError: string | null;
	}
}
