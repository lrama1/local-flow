export type DiffOp = { type: "same" | "add" | "del"; text: string };

/** Line-based LCS diff (inputs here are a few hundred lines at most). */
export function diffLines(before: string, after: string): DiffOp[] {
  const a = before.split(/\r?\n/);
  const b = after.split(/\r?\n/);
  const n = a.length;
  const m = b.length;
  const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i]![j] = a[i] === b[j] ? lcs[i + 1]![j + 1]! + 1 : Math.max(lcs[i + 1]![j]!, lcs[i]![j + 1]!);
    }
  }
  const ops: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      ops.push({ type: "same", text: a[i]! });
      i++;
      j++;
    } else if (lcs[i + 1]![j]! >= lcs[i]![j + 1]!) {
      ops.push({ type: "del", text: a[i++]! });
    } else {
      ops.push({ type: "add", text: b[j++]! });
    }
  }
  while (i < n) ops.push({ type: "del", text: a[i++]! });
  while (j < m) ops.push({ type: "add", text: b[j++]! });
  return ops;
}

export function hasChanges(ops: DiffOp[]): boolean {
  return ops.some((o) => o.type !== "same");
}

/** Stable, human-readable text form of a structured value for diffing. */
export function toDiffText(value: unknown): string {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}
