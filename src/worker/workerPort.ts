export type WorkerPort = {
	postMessage(message: unknown): void;
	terminate(): void;
	onmessage: ((event: MessageEvent<unknown>) => void) | null;
	onerror: ((event: ErrorEvent) => void) | null;
	onmessageerror: ((event: MessageEvent<unknown>) => void) | null;
};
