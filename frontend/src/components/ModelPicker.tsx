import { Link } from "react-router";
import { installedFor, useModels } from "@/api/models";
import { NativeSelect } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

const KIND_LABELS: Record<string, string> = {
  t2i: "image",
  image_edit: "image-edit",
  t2v: "text-to-video",
  i2v: "image-to-video",
  flf2v: "first/last-frame video",
  extend: "video",
};

/** Installed models that can run `capability` (LoRAs/components excluded). Empty value = server default. */
export function ModelPicker({
  capability,
  value,
  onChange,
  className,
  label = "Model",
}: {
  capability: string;
  value: string | null;
  onChange: (id: string | null) => void;
  className?: string;
  label?: string;
}) {
  const { data } = useModels();
  const models = installedFor(data, capability);
  if (data && models.length === 0) {
    return (
      <Link to="/models?tab=catalog" className={cn("text-primary text-xs hover:underline", className)}>
        No {KIND_LABELS[capability] ?? capability} model installed → Models
      </Link>
    );
  }
  return (
    <NativeSelect
      aria-label={label}
      className={cn("h-8 w-auto text-xs", className)}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value || null)}
    >
      <option value="">Default model</option>
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.name}
        </option>
      ))}
    </NativeSelect>
  );
}
