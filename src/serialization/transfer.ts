import {
	arrayLikeToFloat32,
	packRowsToFloat32,
} from "../utils/typedArrayUtils.js";

export type TransferPolicy = {
	transferThreshold?: number;
	transferOwnership?: boolean;
	// `auto` leaves the storage decision to the gateway; `typedarray` prefers
	// keeping numeric embeddings as typed arrays when possible.
	// This field is currently advisory for serialization helpers.
	persistEmbeddingAs?: "auto" | "arraybuffer" | "typedarray";
};

export type PackedEmbeddings = {
	buffer: ArrayBuffer;
	rows: number;
	dims: number;
};

export type PackedEmbeddingsWithTransfer = PackedEmbeddings & {
	transferList: ArrayBuffer[];
};

export type SerializedEmbedding =
	| { type: "buffer"; buffer: ArrayBuffer; length: number }
	| { type: "array"; array: number[] }
	| unknown;

export function packEmbeddings(
	embeddings: ArrayLike<number>[],
): PackedEmbeddings {
	return packRowsToFloat32(embeddings);
}

export function shouldTransfer(length: number, threshold = 16): boolean {
	return Number.isFinite(length) && length >= threshold;
}

export function packEmbeddingsForTransfer(
	embeddings: ArrayLike<number>[],
	policy: TransferPolicy = {},
): PackedEmbeddingsWithTransfer {
	const packed = packEmbeddings(embeddings);
	const threshold = policy.transferThreshold ?? 16;
	const transferList = shouldTransfer(packed.rows * packed.dims, threshold)
		? [packed.buffer]
		: [];
	return { ...packed, transferList };
}

function resolveTransferOwnership(
	view: Float32Array,
	ownership: boolean,
): { buffer: ArrayBuffer; transferList: ArrayBuffer[] } {
	if (
		ownership &&
		view.byteOffset === 0 &&
		view.buffer.byteLength === view.byteLength &&
		view.buffer instanceof ArrayBuffer
	) {
		return { buffer: view.buffer, transferList: [view.buffer] };
	}

	const copy = new Float32Array(view);
	return { buffer: copy.buffer, transferList: [copy.buffer] };
}

export function serializeEmbeddingForTransfer(
	embedding: number[] | ArrayBufferView | ArrayBuffer | unknown,
	policy: TransferPolicy = {},
): {
	serializedEmbedding: SerializedEmbedding;
	transferList: ArrayBuffer[];
} {
	const threshold = policy.transferThreshold ?? 16;
	const transferOwnership = policy.transferOwnership ?? true;

	if (ArrayBuffer.isView(embedding)) {
		const view = arrayLikeToFloat32(embedding);
		const length = view.length;
		if (shouldTransfer(length, threshold)) {
			const { buffer, transferList } = resolveTransferOwnership(
				view,
				transferOwnership,
			);
			return {
				serializedEmbedding: { type: "buffer", buffer, length },
				transferList,
			};
		}

		return {
			serializedEmbedding: { type: "array", array: Array.from(view) },
			transferList: [],
		};
	}

	if (embedding instanceof ArrayBuffer) {
		const view = new Float32Array(embedding);
		const length = view.length;
		if (shouldTransfer(length, threshold)) {
			if (transferOwnership) {
				return {
					serializedEmbedding: { type: "buffer", buffer: embedding, length },
					transferList: [embedding],
				};
			}

			const copy = Float32Array.from(view);
			return {
				serializedEmbedding: {
					type: "buffer",
					buffer: copy.buffer,
					length: copy.length,
				},
				transferList: [copy.buffer],
			};
		}

		return {
			serializedEmbedding: { type: "array", array: Array.from(view) },
			transferList: [],
		};
	}

	if (Array.isArray(embedding)) {
		const length = embedding.length;
		if (shouldTransfer(length, threshold)) {
			const copy = Float32Array.from(embedding);
			return {
				serializedEmbedding: {
					type: "buffer",
					buffer: copy.buffer,
					length: copy.length,
				},
				transferList: [copy.buffer],
			};
		}

		return {
			serializedEmbedding: { type: "array", array: [...embedding] },
			transferList: [],
		};
	}

	return { serializedEmbedding: embedding, transferList: [] };
}
