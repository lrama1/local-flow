import { AtSignIcon } from "lucide-react";
import { useEffect, useId, useRef, useState, type ComponentProps, type KeyboardEvent } from "react";
import { Textarea } from "@/components/ui/primitives";
import {
  insertMention,
  matchMentions,
  mentionQuery,
  unknownMentions,
  type MentionOption,
} from "@/lib/mentions";
import { cn } from "@/lib/utils";

const KIND_LABEL: Record<string, string> = { character: "Character", location: "Location", prop: "Prop" };

/** A textarea with `@` autocomplete for characters, locations and props, and an unknown-mention warning.
 * Without `options` (still loading, or a field that doesn't resolve mentions) it's a plain textarea. */
export function MentionTextarea({
  value,
  onChange,
  options,
  className,
  ...props
}: Omit<ComponentProps<"textarea">, "value" | "onChange"> & {
  value: string;
  onChange: (value: string) => void;
  options: MentionOption[] | undefined;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const listId = useId();
  const [caret, setCaret] = useState<number | null>(null);
  const [active, setActive] = useState(0);
  const [dismissed, setDismissed] = useState<number | null>(null);

  const query = options && caret !== null ? mentionQuery(value, caret) : null;
  const matches = query && dismissed !== query.start ? matchMentions(options!, query.query) : [];
  const open = matches.length > 0;
  const unknown = options
    ? unknownMentions(
        value,
        options.map((o) => o.key),
        query ? caret! : undefined,
      )
    : [];

  // Dialogs (Radix) close on Escape from a document capture listener, before our onKeyDown runs; a
  // window capture listener runs first, and a prevented event doesn't dismiss the dialog.
  useEffect(() => {
    if (!open) return;
    const block = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape" && document.activeElement === ref.current) e.preventDefault();
    };
    window.addEventListener("keydown", block, { capture: true });
    return () => window.removeEventListener("keydown", block, { capture: true });
  }, [open]);

  const track = () => setCaret(ref.current?.selectionStart ?? null);
  const pick = (opt: MentionOption) => {
    if (!query || caret === null) return;
    const next = insertMention(value, query.start, caret, opt.key);
    onChange(next.text);
    setCaret(next.caret);
    requestAnimationFrame(() => {
      ref.current?.focus();
      ref.current?.setSelectionRange(next.caret, next.caret);
    });
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (open) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const step = e.key === "ArrowDown" ? 1 : -1;
        setActive((a) => (a + step + matches.length) % matches.length);
        return;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        pick(matches[Math.min(active, matches.length - 1)]!);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setDismissed(query!.start);
        return;
      }
    }
    props.onKeyDown?.(e);
  };

  return (
    <div className="relative">
      <Textarea
        ref={ref}
        {...props}
        className={className}
        value={value}
        aria-autocomplete={options ? "list" : undefined}
        aria-expanded={options ? open : undefined}
        aria-controls={open ? listId : undefined}
        onChange={(e) => {
          onChange(e.target.value);
          setCaret(e.target.selectionStart);
          setActive(0);
        }}
        onSelect={track}
        onClick={track}
        onKeyDown={onKeyDown}
        onBlur={(e) => {
          setCaret(null);
          props.onBlur?.(e);
        }}
      />
      {open && (
        <ul
          id={listId}
          role="listbox"
          aria-label="Mention suggestions"
          className="bg-popover absolute inset-x-0 top-full z-50 mt-1 max-h-60 overflow-y-auto rounded-md border p-1 shadow-lg"
        >
          {matches.map((o, i) => (
            <li
              key={`${o.kind}:${o.key}`}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => {
                e.preventDefault(); // keep focus (and the caret) in the textarea
                pick(o);
              }}
              onMouseEnter={() => setActive(i)}
              className={cn(
                "flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm",
                i === active && "bg-accent text-accent-foreground",
              )}
            >
              {o.thumb ? (
                <img src={o.thumb} alt="" className="size-6 rounded object-cover" />
              ) : (
                <span className="bg-muted flex size-6 items-center justify-center rounded">
                  <AtSignIcon className="text-muted-foreground size-3" />
                </span>
              )}
              <span className="truncate">{o.name}</span>
              <span className="text-muted-foreground font-mono text-xs">@{o.key}</span>
              <span className="text-muted-foreground ml-auto text-[11px]">
                {KIND_LABEL[o.kind] ?? o.kind}
              </span>
            </li>
          ))}
        </ul>
      )}
      {unknown.length > 0 && (
        <p className="text-warning mt-1 text-xs" role="alert">
          Unknown {unknown.length === 1 ? "mention" : "mentions"} {unknown.map((k) => `@${k}`).join(", ")}.
          Type @ to pick one, or write @@ for a literal @.
        </p>
      )}
    </div>
  );
}
