export class MinHeap<T> {
	private data: T[] = [];
	private cmp: (a: T, b: T) => number;

	constructor(cmp: (a: T, b: T) => number) {
		this.cmp = cmp;
	}

	push(value: T): void {
		this.data.push(value);
		let i = this.data.length - 1;
		while (i > 0) {
			const parent = Math.floor((i - 1) / 2);
			if (this.cmp(this.data[i], this.data[parent]) >= 0) break;
			[this.data[i], this.data[parent]] = [this.data[parent], this.data[i]];
			i = parent;
		}
	}

	pop(): T | undefined {
		if (this.data.length === 0) return undefined;
		const top = this.data[0];
		const last = this.data.pop();
		if (this.data.length > 0 && last !== undefined) {
			this.data[0] = last;
			let i = 0;
			while (true) {
				const left = 2 * i + 1;
				const right = 2 * i + 2;
				let smallest = i;
				if (
					left < this.data.length &&
					this.cmp(this.data[left], this.data[smallest]) < 0
				) {
					smallest = left;
				}
				if (
					right < this.data.length &&
					this.cmp(this.data[right], this.data[smallest]) < 0
				) {
					smallest = right;
				}
				if (smallest === i) break;
				[this.data[i], this.data[smallest]] = [
					this.data[smallest],
					this.data[i],
				];
				i = smallest;
			}
		}
		return top;
	}

	size(): number {
		return this.data.length;
	}

	peek(): T | undefined {
		return this.data[0];
	}

	toArrayDesc(): T[] {
		return [...this.data].sort((a, b) => this.cmp(b, a));
	}
}
