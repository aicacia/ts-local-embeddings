/* eslint-disable @typescript-eslint/no-explicit-any */
export function fallbackHash(input: string): string {
	let hash = 5381;
	for (let index = 0; index < input.length; index += 1) {
		hash = (hash * 33) ^ input.charCodeAt(index);
	}

	return `fallback-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

let __nodeCryptoModule: any | null = null;

export async function sha256(input: string): Promise<string> {
	try {
		if (typeof process !== "undefined" && (process as any).versions?.node) {
			if (__nodeCryptoModule === null) {
				__nodeCryptoModule = await import("node:crypto");
			}
			return __nodeCryptoModule
				.createHash("sha256")
				.update(input)
				.digest("hex");
		}
	} catch (_error) {
		// fall through to fallback hashing in browser environments.
	}

	return fallbackHash(input);
}

export type ContentHasher = (content: string) => Promise<string> | string;

export function createContentHashGetter(options: {
	contentHasher?: ContentHasher;
	contentHashCacheMax?: number;
}) {
	const contentHashCacheMax =
		typeof options.contentHashCacheMax === "number" &&
		Number.isFinite(options.contentHashCacheMax) &&
		options.contentHashCacheMax > 0
			? Math.max(1, Math.floor(options.contentHashCacheMax))
			: 4096;
	const cache = new Map<string, Promise<string>>();

	function setContentHashCache(key: string, value: Promise<string>): void {
		cache.set(key, value);
		if (cache.size > contentHashCacheMax) {
			const firstKey = cache.keys().next().value as string | undefined;
			if (firstKey !== undefined) {
				cache.delete(firstKey);
			}
		}
	}

	return async function getContentHash(content: string): Promise<string> {
		const cachedHash = cache.get(content);
		if (cachedHash) {
			return cachedHash;
		}

		const hashPromise = (async () => {
			if (typeof options.contentHasher === "function") {
				const maybe = options.contentHasher(content);
				return typeof maybe === "string" ? maybe : await maybe;
			}
			return sha256(content);
		})();

		setContentHashCache(content, hashPromise);
		return hashPromise;
	};
}
