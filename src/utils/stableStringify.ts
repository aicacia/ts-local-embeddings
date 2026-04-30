export function stableStringify(value: unknown): string {
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value);
	}

	if (Array.isArray(value)) {
		return `[${value.map((item) => stableStringify(item)).join(",")}]`;
	}

	const objectValue = value as Record<string, unknown>;
	const sortedKeys = Object.keys(objectValue).sort();
	return `{
${sortedKeys
	.map((key) => `${JSON.stringify(key)}:${stableStringify(objectValue[key])}`)
	.join(",")}}`;
}
