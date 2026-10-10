import { ImageIcon, SearchIcon, UserPlusIcon, UsersRoundIcon } from "lucide-react";
import { useState } from "react";
import { mediaUrl, primaryImage, useLibrary, type LibraryCharacter } from "@/api/characters";
import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Badge, Card, Input, Skeleton } from "@/components/ui/primitives";
import { CharacterWizard } from "@/features/characters/CharacterWizard";
import { LibraryCharacterDialog } from "./LibraryCharacterDialog";

function LibraryCard({ c, onOpen }: { c: LibraryCharacter; onOpen: () => void }) {
  const img = primaryImage(c);
  const views = c.images.filter((i) => !i.is_primary).length;
  return (
    <Card className="overflow-hidden" data-testid={`library-${c.key}`}>
      <button type="button" className="block w-full text-left" onClick={onOpen} aria-label={`Open ${c.name}`}>
        <div className="bg-muted flex aspect-square items-center justify-center">
          {img ? (
            <img src={mediaUrl(img.thumb_url)} alt={c.name} className="size-full object-cover" />
          ) : (
            <ImageIcon className="text-muted-foreground size-6" />
          )}
        </div>
        <div className="space-y-1 p-3">
          <p className="truncate text-sm font-medium">{c.name}</p>
          <div className="flex flex-wrap gap-1">
            <Badge variant="outline" className="font-mono">
              @{c.key}
            </Badge>
            {views > 0 && <Badge variant="secondary">{views === 1 ? "1 view" : `${views} views`}</Badge>}
          </div>
          <p className="text-muted-foreground line-clamp-2 text-xs">{c.character_info || c.visual_prompt}</p>
        </div>
      </button>
    </Card>
  );
}

export function CastLibraryPage() {
  const [q, setQ] = useState("");
  const [wizardOpen, setWizardOpen] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);
  const { data = [], isLoading } = useLibrary(q);
  const open = data.find((c) => c.id === openId);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Cast library</h1>
          <p className="text-muted-foreground text-sm">
            Reusable characters, shared across projects. Cast one into a project from here, or from a
            project&apos;s Ingredients tab.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <SearchIcon className="text-muted-foreground absolute top-2.5 left-2.5 size-4" />
            <Input
              aria-label="Search the Cast library"
              placeholder="Search"
              className="w-56 pl-8"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          <Button onClick={() => setWizardOpen(true)}>
            <UserPlusIcon />
            New character
          </Button>
        </div>
      </div>
      {isLoading ? (
        <Skeleton className="h-60" />
      ) : data.length === 0 ? (
        <EmptyState
          icon={UsersRoundIcon}
          title={q ? "No matches" : "No characters yet"}
          action={
            !q && (
              <Button variant="outline" onClick={() => setWizardOpen(true)}>
                <UserPlusIcon />
                Create a character
              </Button>
            )
          }
        >
          {!q &&
            "Create one from a template, a description or a photo, or save a project character to the library."}
        </EmptyState>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {data.map((c) => (
            <LibraryCard key={c.id} c={c} onOpen={() => setOpenId(c.id)} />
          ))}
        </div>
      )}
      {open && (
        <LibraryCharacterDialog key={open.updated_at} character={open} onClose={() => setOpenId(null)} />
      )}
      <CharacterWizard open={wizardOpen} onOpenChange={setWizardOpen} />
    </div>
  );
}
