import { useJobs } from "@/api/queries";

export function useActiveJobCount(): { running: number; queued: number } {
  const { data } = useJobs();
  const items = data?.items ?? [];
  return {
    running: items.filter((j) => j.status === "running").length,
    queued: items.filter((j) => j.status === "queued").length,
  };
}
