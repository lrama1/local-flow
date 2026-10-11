import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap, type Schemas } from "./client";

export type Diagnostics = Schemas["DiagnosticsOut"];
export type DiagCheck = Schemas["DiagCheck"];
export type LogTail = Schemas["LogTail"];

export const dk = {
  diagnostics: ["diagnostics"] as const,
  log: (name: string, lines: number) => ["diagnostics", "log", name, lines] as const,
};

const fetchDiagnostics = (refresh: boolean) =>
  unwrap(api.GET("/api/system/diagnostics", { params: { query: { refresh } } }));

export function useDiagnostics() {
  return useQuery({ queryKey: dk.diagnostics, queryFn: () => fetchDiagnostics(false) });
}

/** Re-runs the checks and recounts folder sizes. */
export function useRerunDiagnostics() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => fetchDiagnostics(true),
    onSuccess: (d) => qc.setQueryData(dk.diagnostics, d),
  });
}

export function useLogTail(name: string | null, lines: number) {
  return useQuery({
    queryKey: dk.log(name ?? "", lines),
    enabled: !!name,
    queryFn: () =>
      unwrap(api.GET("/api/system/logs/{name}", { params: { path: { name: name! }, query: { lines } } })),
  });
}

/** Downloads the redacted support bundle (a fetch with the token header, then a blob link). */
export async function downloadBundle(): Promise<void> {
  const { data, response } = await api.GET("/api/system/diagnostics/bundle", { parseAs: "blob" });
  if (!response.ok || !data) throw new Error(`Couldn't build the bundle (${response.status})`);
  const name = /filename="([^"]+)"/.exec(response.headers.get("content-disposition") ?? "")?.[1];
  const url = URL.createObjectURL(data);
  const a = document.createElement("a");
  a.href = url;
  a.download = name ?? "localflow-diagnostics.zip";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
