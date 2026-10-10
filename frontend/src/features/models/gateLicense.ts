import { ApiError } from "@/api/client";
import { useUi } from "@/stores/ui";

/** If `error` is a `license_not_acknowledged` refusal, asks for the acknowledgement (see `LicenseGate`) and
 * then runs `retry`. Returns true when it handled the error. */
export function gateLicense(error: unknown, retry: () => void): boolean {
  if (!(error instanceof ApiError) || error.code !== "license_not_acknowledged") return false;
  const details = (error.details ?? {}) as { model_entry_id?: string };
  if (!details.model_entry_id) return false;
  useUi.getState().setLicensePrompt({ modelEntryId: details.model_entry_id, retry });
  return true;
}
