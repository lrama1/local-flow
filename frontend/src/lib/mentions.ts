/** `@key` mentions, parsed like the backend (`services/characters/mentions.py`): `@@` is a literal `@`,
 * and an `@` right after a word character (an email address) is not a mention. */
const MENTION = /@@|(?<![\w@])@([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/g;

export interface MentionOption {
  key: string;
  name: string;
  kind: string;
  thumb?: string;
}

export interface ParsedMention {
  key: string;
  start: number;
  end: number;
}

export function parseMentions(text: string): ParsedMention[] {
  const out: ParsedMention[] = [];
  for (const m of text.matchAll(MENTION)) {
    if (m[1]) out.push({ key: m[1].toLowerCase(), start: m.index, end: m.index + m[0].length });
  }
  return out;
}

export function unknownMentions(text: string, keys: Iterable<string>, ignoreAt?: number): string[] {
  const known = new Set([...keys].map((k) => k.toLowerCase()));
  const unknown = parseMentions(text)
    .filter((m) => !known.has(m.key) && m.end !== ignoreAt)
    .map((m) => m.key);
  return [...new Set(unknown)];
}

/** The partial `@query` ending at the caret, if the user is typing a mention. */
export function mentionQuery(text: string, caret: number): { start: number; query: string } | null {
  const m = /(?:^|[^\w@])@([A-Za-z0-9-]*)$/.exec(text.slice(0, caret));
  if (!m) return null;
  return { start: caret - m[1]!.length - 1, query: m[1]!.toLowerCase() };
}

const KIND_ORDER: Record<string, number> = { character: 0, location: 1, prop: 2 };

/** Characters first, then locations, then props; key-prefix matches before name matches. */
export function matchMentions(options: MentionOption[], query: string, limit = 8): MentionOption[] {
  const q = query.toLowerCase();
  const scored = options
    .map((o) => {
      const key = o.key.toLowerCase();
      const name = o.name.toLowerCase();
      const score = key.startsWith(q)
        ? 0
        : name.startsWith(q)
          ? 1
          : key.includes(q) || name.includes(q)
            ? 2
            : -1;
      return { o, score };
    })
    .filter((x) => x.score >= 0);
  scored.sort(
    (a, b) =>
      (KIND_ORDER[a.o.kind] ?? 9) - (KIND_ORDER[b.o.kind] ?? 9) ||
      a.score - b.score ||
      a.o.key.localeCompare(b.o.key),
  );
  return scored.slice(0, limit).map((x) => x.o);
}

export function insertMention(
  text: string,
  start: number,
  caret: number,
  key: string,
): { text: string; caret: number } {
  const after = text.slice(caret);
  const insert = `@${key}${/^\s/.test(after) ? "" : " "}`;
  return { text: text.slice(0, start) + insert + after, caret: start + insert.length };
}
