export function isTypedArray(value: unknown): value is ArrayBufferView {
	return (
		typeof value === "object" &&
		value !== null &&
		ArrayBuffer.isView(value) &&
		!(value instanceof DataView)
	);
}

export function arrayLikeToFloat32(
	value: ArrayLike<number> | ArrayBufferView,
): Float32Array {
	if (ArrayBuffer.isView(value)) {
		// Prefer returning the original Float32Array view when possible.
		if (value instanceof Float32Array) return value;
		const view = value as ArrayBufferView & {
			length?: number;
			byteOffset?: number;
			byteLength?: number;
			buffer?: ArrayBuffer;
		};
		const byteOffset = (view as { byteOffset?: number }).byteOffset ?? 0;
		const length =
			(view as { length?: number; byteLength?: number }).length ??
			Math.floor(((view as { byteLength?: number }).byteLength ?? 0) / 4);
		try {
			return new Float32Array(
				(view as { buffer: ArrayBuffer }).buffer,
				byteOffset,
				length,
			);
		} catch (_err) {
			// Fall back to copying into a new Float32Array
			const out = new Float32Array(length);
			for (let i = 0; i < length; i++) {
				const asArrayLike = view as unknown as ArrayLike<number>;
				out[i] = Number(asArrayLike[i]) || 0;
			}
			return out;
		}
	}

	// Array-like (e.g. number[]). Coerce via unknown to satisfy TSIterable typing.
	return Float32Array.from(value as unknown as Iterable<number>);
}

export function packRowsToFloat32(
	rows: ArrayLike<number>[],
	dims?: number,
): { buffer: ArrayBuffer; rows: number; dims: number } {
	const rowCount = rows.length;
	if (rowCount === 0) return { buffer: new ArrayBuffer(0), rows: 0, dims: 0 };

	const first = rows[0] as ArrayLike<number> | ArrayBufferView;
	const inferredDims =
		dims ??
		(ArrayBuffer.isView(first) ||
		typeof (first as { length?: unknown }).length === "number"
			? ((first as { length?: number }).length ?? 0)
			: 0);
	const outDims = Number(inferredDims) || 0;
	if (outDims === 0) return { buffer: new ArrayBuffer(0), rows: 0, dims: 0 };

	// Heuristic: if rows are homogeneous (all plain arrays or all typed views),
	// use optimized paths to minimize per-row branching overhead. Sample a
	// subset to avoid expensive checks on huge inputs.
	const SAMPLE_LIMIT = 128;
	const sampleCount = Math.min(SAMPLE_LIMIT, rowCount);
	let arrayCount = 0;
	let viewCount = 0;
	for (let i = 0; i < sampleCount; i++) {
		const r = rows[i];
		if (Array.isArray(r)) arrayCount += 1;
		else if (ArrayBuffer.isView(r)) viewCount += 1;
	}

	const allArrays = arrayCount === sampleCount;
	const allViews = viewCount === sampleCount;

	if (allArrays) {
		// Fast path for many small JS arrays: avoid ArrayBuffer.isView checks.
		const flat = new Float32Array(rowCount * outDims);
		let outIndex = 0;
		for (let r = 0; r < rowCount; r++) {
			const al = rows[r] as ArrayLike<number>;
			for (let c = 0; c < outDims; c++) {
				flat[outIndex++] = Number(al[c]) || 0;
			}
		}
		return { buffer: flat.buffer, rows: rowCount, dims: outDims };
	}

	if (allViews) {
		// Fast path for typed-array views: use set() where possible.
		const flat = new Float32Array(rowCount * outDims);
		for (let r = 0; r < rowCount; r++) {
			const view = arrayLikeToFloat32(
				rows[r] as ArrayLike<number> | ArrayBufferView,
			);
			if (view.length === outDims) {
				flat.set(view, r * outDims);
				continue;
			}
			const base = r * outDims;
			const limit = Math.min(view.length, outDims);
			for (let c = 0; c < limit; c++) flat[base + c] = view[c];
			for (let c = limit; c < outDims; c++) flat[base + c] = 0;
		}
		return { buffer: flat.buffer, rows: rowCount, dims: outDims };
	}

	// Mixed case: fall back to flexible per-row handling.
	const flat = new Float32Array(rowCount * outDims);
	for (let r = 0; r < rowCount; r++) {
		const row = rows[r] as ArrayLike<number> | ArrayBufferView;
		if (ArrayBuffer.isView(row)) {
			const view = arrayLikeToFloat32(row as ArrayBufferView);
			if (view.length === outDims) {
				flat.set(view, r * outDims);
				continue;
			}
			const limit = Math.min(view.length, outDims);
			for (let c = 0; c < limit; c++) flat[r * outDims + c] = view[c];
			for (let c = limit; c < outDims; c++) flat[r * outDims + c] = 0;
			continue;
		}

		const al = row as ArrayLike<number>;
		for (let c = 0; c < outDims; c++) {
			flat[r * outDims + c] = Number(al[c]) || 0;
		}
	}

	return { buffer: flat.buffer, rows: rowCount, dims: outDims };
}
