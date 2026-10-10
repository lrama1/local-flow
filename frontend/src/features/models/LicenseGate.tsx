import { useModels } from "@/api/models";
import { useUi } from "@/stores/ui";
import { LicenseDialog } from "./ModelCard";

/** Mounted once in the layout: shows the license dialog requested by `gateLicense`. */
export function LicenseGate() {
  const prompt = useUi((s) => s.licensePrompt);
  const setPrompt = useUi((s) => s.setLicensePrompt);
  const { data: models } = useModels();
  const model = prompt ? models?.find((m) => m.id === prompt.modelEntryId) : undefined;
  if (!prompt || !model) return null;
  return (
    <LicenseDialog
      model={model}
      open
      onClose={() => setPrompt(null)}
      onAcknowledged={() => {
        setPrompt(null);
        prompt.retry();
      }}
    />
  );
}
