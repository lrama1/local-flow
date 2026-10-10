import createClient, { type Middleware } from "openapi-fetch";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];
export type Project = Schemas["ProjectOut"];
export type Asset = Schemas["AssetOut"];
export type Job = Schemas["JobOut"];
export type JobStatus = Job["status"];
export type SystemStatus = Schemas["SystemStatus"];
export type WorkerInfo = Schemas["WorkerInfo"];
export type Adapter = Schemas["AdapterOut"];
export type SecretStatus = Schemas["SecretStatus"];
export type ProviderStatus = Schemas["ProviderStatus"];
export type LlmModel = Schemas["LLMModelInfo"];
export type LlmChoice = Schemas["LlmChoice"];
export type LlmProvider = LlmChoice["provider"];
export type Story = Schemas["StoryOut"];
export type StoryStepResult = Schemas["StoryStepResult"];
export type RefinedStory = Schemas["RefinedStory"];
export type StoryBible = Schemas["StoryBible"];
export type StoryOptions = Schemas["StoryOptions"];
export type UsageOut = Schemas["UsageOut"];
export type Ingredient = Schemas["IngredientOut"];
export type IngredientKind = Ingredient["kind"];
export type Board = Schemas["BoardOut"];
export type Scene = Schemas["SceneOut"];
export type Shot = Schemas["ShotOut"];
export type ShotUpdate = Schemas["ShotUpdate"];
export type CameraMove = Shot["camera_move"];
export type ShotType = Shot["shot_type"];
export type GenerationMode = Shot["generation_mode"];

export const TOKEN_HEADER = "X-LocalFlow-Token";

/** Per-launch token: injected as a <meta> tag in production, or via VITE_LOCALFLOW_TOKEN in dev. */
export function getToken(): string {
  const meta = document.querySelector<HTMLMetaElement>('meta[name="localflow-token"]');
  return meta?.content || import.meta.env.VITE_LOCALFLOW_TOKEN || "";
}

/** Media elements can't send headers; the backend accepts `?token=` on file/thumb/preview URLs. */
export function withToken(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(getToken())}`;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const auth: Middleware = {
  onRequest({ request }) {
    request.headers.set(TOKEN_HEADER, getToken());
    return request;
  },
};

// Absolute same-origin base URL (relative URLs break `new Request()` outside the browser, e.g. in tests).
export const api = createClient<paths>({
  baseUrl: window.location.origin,
  fetch: (req) => globalThis.fetch(req), // resolved per call so tests can stub fetch
});
api.use(auth);

type ErrorBody = { error?: { code?: string; message?: string; details?: unknown } };

/** Unwraps an openapi-fetch result, throwing ApiError with the backend's `{error}` payload. */
export async function unwrap<T>(
  promise: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await promise;
  if (error !== undefined || !response.ok) {
    const body = (error ?? {}) as ErrorBody;
    throw new ApiError(
      response.status,
      body.error?.code ?? "http_error",
      body.error?.message ?? `Request failed (${response.status})`,
      body.error?.details,
    );
  }
  return data as T;
}
