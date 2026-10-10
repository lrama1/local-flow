import { BoxesIcon, HardDriveIcon, Loader2Icon, UserIcon } from "lucide-react";
import { Link, useSearchParams } from "react-router";
import { formatBytes, isInstalledTab, useModelOptions, useModels, type ModelOptions } from "@/api/models";
import { EmptyState } from "@/components/EmptyState";
import { Badge } from "@/components/ui/primitives";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CivitaiPanel } from "./CivitaiPanel";
import { HfPanel } from "./HfPanel";
import { ModelCard } from "./ModelCard";

const TABS = ["installed", "catalog", "huggingface", "civitai"] as const;
type Tab = (typeof TABS)[number];

function Accounts({ options }: { options: ModelOptions | undefined }) {
  if (!options) return null;
  const chip = (label: string, a: ModelOptions["hf"]) => (
    <Link to="/settings" title={`${label} account (Settings)`}>
      <Badge variant={a.configured ? "secondary" : "outline"}>
        <UserIcon className="size-3" />
        {label}: {a.configured ? (a.username ?? "key set") : "no token"}
      </Badge>
    </Link>
  );
  return (
    <div className="flex flex-wrap items-center gap-2">
      {chip("Hugging Face", options.hf)}
      {chip("Civitai", options.civitai)}
      <Badge variant="outline" title={options.models_dir}>
        <HardDriveIcon className="size-3" /> {formatBytes(options.free_bytes)} free
      </Badge>
    </div>
  );
}

export function ModelsPage() {
  const { data: models, isLoading } = useModels();
  const { data: options } = useModelOptions();
  const [params, setParams] = useSearchParams();
  const tab: Tab = (TABS as readonly string[]).includes(params.get("tab") ?? "")
    ? (params.get("tab") as Tab)
    : "installed";

  const installed = (models ?? []).filter(isInstalledTab);
  const catalog = (models ?? []).filter((m) => m.catalog_key);

  return (
    <div className="mx-auto max-w-7xl space-y-5 p-6">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="text-xl font-semibold">Models</h1>
          <p className="text-muted-foreground text-sm">
            Install recommended models, any Hugging Face repo, or Civitai checkpoints and LoRAs.
          </p>
        </div>
        <div className="ml-auto">
          <Accounts options={options} />
        </div>
      </div>
      <Tabs value={tab} onValueChange={(v) => setParams({ tab: v }, { replace: true })}>
        <TabsList>
          <TabsTrigger value="installed">
            Installed{installed.length ? ` (${installed.length})` : ""}
          </TabsTrigger>
          <TabsTrigger value="catalog">Catalog</TabsTrigger>
          <TabsTrigger value="huggingface">Hugging Face</TabsTrigger>
          <TabsTrigger value="civitai">Civitai</TabsTrigger>
        </TabsList>

        <TabsContent value="installed" className="mt-4 space-y-6">
          {isLoading && <Loader2Icon className="text-muted-foreground size-5 animate-spin" />}
          {models && installed.length === 0 && (
            <EmptyState icon={BoxesIcon} title="No models installed yet">
              Start with the{" "}
              <button className="text-primary hover:underline" onClick={() => setParams({ tab: "catalog" })}>
                Catalog
              </button>{" "}
              — SDXL (7 GB) and Wan 2.2 TI2V-5B (34 GB) are good first picks.
            </EmptyState>
          )}
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {installed.map((m) => (
              <ModelCard key={m.id} model={m} />
            ))}
          </div>
          {options && (
            <div className="text-muted-foreground space-y-1 text-xs">
              <p className="font-medium">Inference adapters</p>
              <div className="flex flex-wrap gap-1.5">
                {options.adapters.map((a) => (
                  <Badge
                    key={a.key}
                    variant={a.available ? "success" : "outline"}
                    title={a.capabilities.join(", ")}
                  >
                    {a.name}
                    {a.available ? "" : ` · ${a.milestone}`}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="catalog" className="mt-4">
          {isLoading && <Loader2Icon className="text-muted-foreground size-5 animate-spin" />}
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {catalog.map((m) => (
              <ModelCard key={m.id} model={m} />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="huggingface" className="mt-4">
          <HfPanel />
        </TabsContent>

        <TabsContent value="civitai" className="mt-4">
          <CivitaiPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
