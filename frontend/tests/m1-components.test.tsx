import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import type { ProviderStatus, SecretStatus, Story } from "@/api/client";
import { ApiKeysCard } from "@/features/settings/ApiKeysCard";
import { StoryTab } from "@/features/story/StoryTab";
import { jsonResponse, renderWithProviders } from "./helpers";

type Handler = (req: Request) => Response | Promise<Response>;

/** Routes `METHOD /path` to handlers; records every request. */
function mockApi(routes: Record<string, Handler>) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  const fetchMock = vi.fn(async (req: Request) => {
    const path = new URL(req.url).pathname;
    const key = `${req.method} ${path}`;
    const body =
      req.method === "GET"
        ? undefined
        : await req
            .clone()
            .json()
            .catch(() => undefined);
    calls.push({ method: req.method, path, body });
    const match = Object.entries(routes).find(([pattern]) => new RegExp(`^${pattern}$`).test(key));
    if (!match) return jsonResponse({ error: { code: "not_mocked", message: key } }, 404);
    return match[1](req);
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

const provider = (p: Partial<ProviderStatus>): ProviderStatus => ({
  provider: "openai",
  label: "OpenAI",
  configured: false,
  ok: null,
  error: null,
  source: null,
  default_model: "gpt-5.4-mini",
  ...p,
});

const emptyStory: Story = {
  id: null,
  project_id: "p1",
  revision: null,
  step: "none",
  status: "draft",
  raw_idea: "",
  options: {},
  cast_keys: [],
  refined: null,
  bible: null,
  has_breakdown: false,
  llm_provider: null,
  llm_model: null,
  template_versions: {},
  edited_fields: [],
  warnings: [],
  usage: null,
  created_at: null,
};

test("Story tab blocks until an LLM key is configured", async () => {
  mockApi({ "GET /api/llm/providers": () => jsonResponse([provider({})]) });
  renderWithProviders(<StoryTab projectId="p1" onOpenShots={() => {}} />);
  expect(await screen.findByText("Add an LLM API key to write stories")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Open Settings" })).toHaveAttribute("href", "/settings");
});

test("Story tab refines the idea with the options", async () => {
  const refined = {
    title: "The Brass Gull",
    logline: "l",
    synopsis: "s",
    story: "beat",
    estimated_runtime_s: 30,
    tone: "",
    visual_style: "",
  };
  const calls = mockApi({
    "GET /api/llm/providers": () => jsonResponse([provider({ configured: true, ok: true, source: "env" })]),
    "GET /api/projects/p1/story": () => jsonResponse(emptyStory),
    "GET /api/projects/p1/story/revisions": () => jsonResponse([]),
    "GET /api/projects/p1/ingredients": () => jsonResponse([]),
    "GET /api/projects/p1/scenes": () =>
      jsonResponse({ scenes: [], total_duration_s: 0, shot_count: 0, max_clip_seconds: 5, extra: {} }),
    "POST /api/projects/p1/story/refine": () =>
      jsonResponse({
        story: { ...emptyStory, id: "s1", revision: 1, step: "refine", status: "refined", refined },
        saved: true,
        usage: {
          provider: "openai",
          model: "gpt-5.4-mini",
          prompt_tokens: 900,
          completion_tokens: 600,
          cost_usd: 0.0034,
          attempts: 1,
        },
        warnings: [],
      }),
  });
  renderWithProviders(<StoryTab projectId="p1" onOpenShots={() => {}} />);

  const run = await screen.findByTestId("run-refine");
  expect(run).toBeDisabled();
  await userEvent.type(screen.getByLabelText("Story idea"), "A kite that won't come down.");
  await userEvent.type(screen.getByLabelText("Target length (s)"), "30");
  await userEvent.type(screen.getByLabelText("Tone"), "wistful");
  await userEvent.click(run);

  await waitFor(() => expect(calls.some((c) => c.path === "/api/projects/p1/story/refine")).toBe(true));
  const body = calls.find((c) => c.path === "/api/projects/p1/story/refine")!.body as Record<string, unknown>;
  expect(body).toMatchObject({
    raw_idea: "A kite that won't come down.",
    options: { target_length_seconds: 30, tone: "wistful" },
    dry_run: false,
  });
  expect(await screen.findByTestId("usage")).toHaveTextContent(
    "openai/gpt-5.4-mini · 900 in / 600 out · $0.0034",
  );
});

test("API keys card shows env-sourced keys without an input and saves pasted keys", async () => {
  const secret = (p: Partial<SecretStatus>): SecretStatus => ({
    name: "openai_api_key",
    label: "OpenAI",
    kind: "llm",
    configured: false,
    source: null,
    env_var: null,
    env_vars: ["OPENAI_API_KEY"],
    stored: false,
    overridden: false,
    last4: null,
    validated_at: null,
    ok: null,
    account: null,
    token_type: null,
    warnings: [],
    error: null,
    token_url: "https://example.com",
    ...p,
  });
  const calls = mockApi({
    "GET /api/settings/secrets": () =>
      jsonResponse([
        secret({ configured: true, source: "env", env_var: "OPENAI_API_KEY", last4: "PFIA", ok: true }),
        secret({ name: "anthropic_api_key", label: "Anthropic", env_vars: ["ANTHROPIC_API_KEY"] }),
      ]),
    "PUT /api/settings/secrets/anthropic_api_key": () => jsonResponse(secret({ name: "anthropic_api_key" })),
    "POST /api/settings/secrets/.*/test": () => jsonResponse(secret({})),
  });
  renderWithProviders(<ApiKeysCard />);

  expect(await screen.findByText("From environment · OPENAI_API_KEY")).toBeInTheDocument();
  expect(screen.queryByLabelText("OpenAI key")).not.toBeInTheDocument();
  await userEvent.type(screen.getByLabelText("Anthropic key"), "sk-ant-123");
  await userEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(calls.find((c) => c.method === "PUT")?.body).toEqual({ value: "sk-ant-123" }));
});
