import { ImageIcon, Loader2Icon, SearchIcon, UsersRoundIcon } from "lucide-react";
import { useState } from "react";
import { mediaUrl, primaryImage, useImportCharacter, useLibrary } from "@/api/characters";
import type { Ingredient } from "@/api/client";
import { thumbUrl } from "@/api/imagegen";
import { useProjects } from "@/api/queries";
import { useIngredients } from "@/api/story";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input, NativeSelect } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

function Tile({
  name,
  sub,
  img,
  busy,
  disabled,
  onClick,
}: {
  name: string;
  sub: string;
  img: string | undefined;
  busy: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-label={`Import ${name}`}
      className={cn(
        "hover:border-primary/60 flex items-center gap-2 rounded-lg border p-2 text-left transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-50",
      )}
    >
      <span className="bg-muted flex size-12 shrink-0 items-center justify-center overflow-hidden rounded-md">
        {img ? <img src={img} alt="" className="size-full object-cover" /> : <ImageIcon className="size-4" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{name}</span>
        <span className="text-muted-foreground block truncate font-mono text-xs">{sub}</span>
      </span>
      {busy && <Loader2Icon className="size-4 animate-spin" />}
    </button>
  );
}

function ProjectCharacters({
  sourceId,
  importing,
  onPick,
}: {
  sourceId: string;
  importing: string | null;
  onPick: (ing: Ingredient) => void;
}) {
  const { data = [], isLoading } = useIngredients(sourceId);
  const chars = data.filter((i) => i.kind === "character");
  if (isLoading) return <Loader2Icon className="size-4 animate-spin" />;
  if (!chars.length) return <p className="text-muted-foreground text-xs">That project has no characters.</p>;
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {chars.map((i) => (
        <Tile
          key={i.id}
          name={i.name}
          sub={`@${i.key}`}
          img={thumbUrl(i.project_id, i.primary_asset_id)}
          busy={importing === i.id}
          disabled={!!importing}
          onClick={() => onPick(i)}
        />
      ))}
    </div>
  );
}

/** Cast a Cast-library character into a project, or copy one from another project (Section 11.7.2). */
export function ImportCharacterDialog({
  projectId,
  open,
  onOpenChange,
  libraryOnly = false,
  onImported,
}: {
  projectId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  libraryOnly?: boolean;
  onImported?: (ing: Ingredient) => void;
}) {
  const [tab, setTab] = useState<"library" | "project">("library");
  const [q, setQ] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [importing, setImporting] = useState<string | null>(null);
  const { data: library = [], isLoading } = useLibrary(q, open);
  const { data: projects = [] } = useProjects();
  const imp = useImportCharacter();
  const others = projects.filter((p) => p.id !== projectId);

  const run = (id: string, body: { library_character_id?: string; ingredient_id?: string }) => {
    setImporting(id);
    imp.mutate(
      { projectId, body },
      {
        onSuccess: (ing) => {
          onImported?.(ing);
          onOpenChange(false);
        },
        onSettled: () => setImporting(null),
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{libraryOnly ? "Add from the Cast library" : "Import a character"}</DialogTitle>
          <DialogDescription>
            The character is copied into this project with its images, views, info and voice. Later edits to
            the original don&apos;t change the copy until you choose Update from library.
          </DialogDescription>
        </DialogHeader>
        {!libraryOnly && (
          <div className="bg-muted flex w-fit rounded-md p-0.5 text-sm" role="tablist">
            {(
              [
                ["library", "Cast library"],
                ["project", "Another project"],
              ] as const
            ).map(([t, label]) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={cn("rounded px-3 py-1", tab === t && "bg-background shadow-sm")}
              >
                {label}
              </button>
            ))}
          </div>
        )}
        {tab === "library" || libraryOnly ? (
          <div className="space-y-3">
            <div className="relative">
              <SearchIcon className="text-muted-foreground absolute top-2.5 left-2.5 size-4" />
              <Input
                aria-label="Search the Cast library"
                placeholder="Search by name or description"
                className="pl-8"
                value={q}
                onChange={(e) => setQ(e.target.value)}
              />
            </div>
            {isLoading ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : library.length === 0 ? (
              <p className="text-muted-foreground flex items-center gap-2 text-sm">
                <UsersRoundIcon className="size-4" />
                {q ? "No matches." : "The Cast library is empty. Save a character to it first."}
              </p>
            ) : (
              <div className="grid max-h-[50vh] gap-2 overflow-y-auto sm:grid-cols-2">
                {library.map((c) => (
                  <Tile
                    key={c.id}
                    name={c.name}
                    sub={`@${c.key} · ${c.images.length} ${c.images.length === 1 ? "image" : "images"}`}
                    img={mediaUrl(primaryImage(c)?.thumb_url)}
                    busy={importing === c.id}
                    disabled={!!importing}
                    onClick={() => run(c.id, { library_character_id: c.id })}
                  />
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <NativeSelect
              aria-label="Source project"
              value={sourceId}
              onChange={(e) => setSourceId(e.target.value)}
            >
              <option value="">Choose a project…</option>
              {others.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </NativeSelect>
            {sourceId && (
              <ProjectCharacters
                sourceId={sourceId}
                importing={importing}
                onPick={(i) => run(i.id, { ingredient_id: i.id })}
              />
            )}
          </div>
        )}
        {imp.error && <p className="text-destructive text-sm">{imp.error.message}</p>}
      </DialogContent>
    </Dialog>
  );
}
