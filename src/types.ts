export type Constructor<T> = {
	new (...args: unknown[]): T;
	readonly prototype: T;
};
