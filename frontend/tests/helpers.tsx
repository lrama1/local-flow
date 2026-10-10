import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { MemoryRouter } from "react-router";
import type { Job } from "@/api/client";
import { TooltipProvider } from "@/components/ui/tooltip";

export function makeQueryClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
}

export function renderWithProviders(ui: ReactElement, qc: QueryClient = makeQueryClient()) {
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>
      <TooltipProvider>
        <MemoryRouter>{children}</MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
  return { qc, ...render(ui, { wrapper: Wrapper }) };
}

export function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "01JOB000000000000000000001",
    project_id: "01PROJ00000000000000000001",
    type: "t2v",
    status: "queued",
    priority: 0,
    params: { adapter_key: "fake", request: { prompt: "a fox in the snow", seed: 1 } },
    result: null,
    error: null,
    progress: 0,
    progress_message: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    started_at: null,
    finished_at: null,
    target_ref: null,
    ...overrides,
  };
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}
