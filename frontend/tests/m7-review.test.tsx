import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { vi } from "vitest";
import type { ShotReview } from "@/api/review";
import { ReviewCheck } from "@/features/shots/ReviewCheck";
import { jsonResponse, renderWithProviders } from "./helpers";

type Handler = (req: Request) => Response | Promise<Response>;

function mockApi(routes: Record<string, Handler>) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (req: Request) => {
      const url = new URL(req.url);
      const key = `${req.method} ${url.pathname}`;
      const body =
        req.method === "GET"
          ? undefined
          : await req
              .clone()
              .json()
              .catch(() => undefined);
      calls.push({ method: req.method, path: url.pathname, body });
      const match = Object.entries(routes).find(([pattern]) => new RegExp(`^${pattern}$`).test(key));
      if (!match) return jsonResponse({ error: { code: "not_mocked", message: key } }, 404);
      return match[1](req);
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

const REVIEW: ShotReview = {
  shot_id: "s1",
  asset_id: "k1",
  target: "keyframe",
  prompt_field: "keyframe_prompt",
  current_prompt: "@mara on the pier",
  consistent: false,
  issues: ["Mara's hair is short; her reference shows long red hair."],
  suggested_prompt_fix: "@mara on the pier, long red hair",
  references: [
    { asset_id: "r1", ingredient_id: "i1", name: "Mara", kind: "character", view: "primary", label: "Mara" },
  ],
  frames_s: [],
  usage: {
    provider: "openai",
    model: "gpt-5.4-mini",
    prompt_tokens: 1200,
    completion_tokens: 80,
    cost_usd: 0.001,
    attempts: 1,
  },
};

test("check consistency shows the issues and applies the prompt fix to the draft", async () => {
  const calls = mockApi({
    "GET /api/llm/vision": () =>
      jsonResponse({ purpose: "vision_review", available: true, model: "openai/gpt-5.4-mini", reason: null }),
    "POST /api/shots/s1/review": () => jsonResponse(REVIEW),
  });
  const onApplyFix = vi.fn();
  const beforeReview = vi.fn(() => Promise.resolve());
  renderWithProviders(
    <ReviewCheck
      shotId="s1"
      assetId="k1"
      what="keyframe"
      beforeReview={beforeReview}
      onApplyFix={onApplyFix}
    />,
  );

  const button = screen.getByRole("button", { name: "Check consistency" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  const region = await screen.findByRole("region", { name: "Consistency check" });
  expect(beforeReview).toHaveBeenCalled(); // unsaved panel edits are saved first
  expect(calls.find((c) => c.method === "POST")?.body).toEqual({ asset_id: "k1" });
  expect(within(region).getByText("1 issue")).toBeInTheDocument();
  expect(within(region).getByText(REVIEW.issues[0]!)).toBeInTheDocument();
  expect(within(region).getByText(/Compared with Mara/)).toBeInTheDocument();

  await userEvent.click(within(region).getByRole("button", { name: "Apply fix" }));
  expect(onApplyFix).toHaveBeenCalledWith("keyframe_prompt", "@mara on the pier, long red hair");
  expect(screen.queryByRole("region", { name: "Consistency check" })).not.toBeInTheDocument();
});

test("the result survives the panel remounting after saving its draft", async () => {
  mockApi({
    "GET /api/llm/vision": () =>
      jsonResponse({ purpose: "vision_review", available: true, model: "openai/gpt-5.4-mini", reason: null }),
    "POST /api/shots/s1/review": () => jsonResponse(REVIEW),
  });
  // Like ShotPanel (keyed on the shot's updated_at): saving the draft replaces the whole subtree.
  function Panel() {
    const [version, setVersion] = useState(0);
    return (
      <ReviewCheck
        key={version}
        shotId="s1"
        assetId="k1"
        what="keyframe"
        beforeReview={() => {
          setVersion((v) => v + 1);
          return Promise.resolve();
        }}
      />
    );
  }
  renderWithProviders(<Panel />);
  const button = screen.getByRole("button", { name: "Check consistency" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  const region = await screen.findByRole("region", { name: "Consistency check" });
  expect(within(region).getByText("1 issue")).toBeInTheDocument();
});

test("without a vision model the check is disabled and says why", async () => {
  mockApi({
    "GET /api/llm/vision": () =>
      jsonResponse({
        purpose: "vision_review",
        available: false,
        model: "openai/text-only",
        reason: "text-only can't read images; pick a vision model in Settings.",
      }),
  });
  renderWithProviders(<ReviewCheck shotId="s1" assetId="t1" what="take" />);
  const button = screen.getByRole("button", { name: "Check consistency" });
  await waitFor(() => expect(button).toHaveAttribute("title", expect.stringContaining("can't read images")));
  expect(button).toBeDisabled();
});
