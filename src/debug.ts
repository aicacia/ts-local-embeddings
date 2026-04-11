type DebugGlobal = typeof globalThis & {
	__LOCAL_EMBEDDINGS_DEBUG__?: boolean;
};

export function isDebugLoggingEnabled(): boolean {
	return Boolean(
		(globalThis as DebugGlobal).__LOCAL_EMBEDDINGS_DEBUG__ === true,
	);
}

export function setDebugLogging(enabled: boolean) {
	(globalThis as DebugGlobal).__LOCAL_EMBEDDINGS_DEBUG__ = enabled;
}
