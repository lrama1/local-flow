import { ApiKeysCard } from "./ApiKeysCard";
import { DiagnosticsCard } from "./DiagnosticsCard";
import { LlmSettingsCard } from "./LlmSettingsCard";

export function SettingsPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      <div>
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-muted-foreground text-sm">API keys, default models, and diagnostics.</p>
      </div>

      <ApiKeysCard />
      <LlmSettingsCard />
      <DiagnosticsCard />
    </div>
  );
}
