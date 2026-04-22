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
		const anyView = value as any;
		const byteOffset = anyView.byteOffset ?? 0;
		const length = anyView.length ?? Math.floor((anyView.byteLength ?? 0) / 4);
		try {
			return new Float32Array(anyView.buffer, byteOffset, length);
		} catch (_err) {
			// Fall back to copying into a new Float32Array
			const out = new Float32Array(length);
			for (let i = 0; i < length; i++) {
				out[i] = Number((anyView as any)[i]) || 0;
			}
			return out;
		}
	}

	// Array-like (e.g. number[]). Float32Array.from handles array-like inputs.
	return Float32Array.from(value as Iterable<number>);
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
		(ArrayBuffer.isView(first) || (first as any).length
			? ((first as any).length ?? 0)
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
				flat[outIndex++] = Number((al as any)[c]) || 0;
			}
		}
		return { buffer: flat.buffer, rows: rowCount, dims: outDims };
	}

	if (allViews) {
		// Fast path for typed-array views: use set() where possible.
		const flat = new Float32Array(rowCount * outDims);
		for (let r = 0; r < rowCount; r++) {
			const view = arrayLikeToFloat32(rows[r] as ArrayBufferView);
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
			flat[r * outDims + c] = Number((al as any)[c]) || 0;
		}
	}

	return { buffer: flat.buffer, rows: rowCount, dims: outDims };
}
