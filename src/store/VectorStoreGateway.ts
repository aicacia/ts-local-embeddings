import type { StoredVectorRecord } from "./vectorWritePipeline.js";

export interface VectorStoreGateway {
	open(): Promise<IDBDatabase>;
	close(): Promise<void>;
	getAll(): Promise<StoredVectorRecord[]>;
	iterateAll<T>(
		callback: (record: T) => Promise<boolean | undefined> | boolean | undefined,
	): Promise<void>;
	queryByContentHash(
		contentHashes: string[],
		contents: string[],
		predicate: (record: StoredVectorRecord, index: number) => boolean,
	): Promise<Array<StoredVectorRecord | null>>;
	count(): Promise<number>;
	put(records: StoredVectorRecord[]): Promise<void>;
	clear(): Promise<void>;
}
