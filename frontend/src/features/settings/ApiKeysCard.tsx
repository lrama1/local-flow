import { CheckCircle2Icon, CircleHelpIcon, ExternalLinkIcon, Loader2Icon, XCircleIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ApiError, SecretStatus } from "@/api/client";
import { useImportHfCli } from "@/api/models";
import { useDeleteSecret, useSecrets, useSetSecret, useTestSecret } from "@/api/story";
import { Button } from "@/components/ui/button";
import { Badge, Card, CardContent, CardHeader, CardTitle, Input, Skeleton } from "@/components/ui/primitives";
import { relativeTime } from "@/lib/utils";

function SourceBadge({ s }: { s: SecretStatus }) {
  if (!s.configured) return <Badge variant="outline">Not set</Badge>;
  if (s.source === "env") return <Badge variant="secondary">From environment · {s.env_var}</Badge>;
  return <Badge variant="secondary">Stored in {s.source === "keyring" ? "OS keyring" : s.source}</Badge>;
}

function Validation({ s }: { s: SecretStatus }) {
  if (!s.configured) return null;
  if (s.ok === true) {
    return (
      <span className="text-success inline-flex items-center gap-1 text-xs">
        <CheckCircle2Icon className="size-3.5" />
        {s.account?.username ? `Signed in as ${String(s.account.username)}` : "Valid"}
        {s.validated_at && (
          <span className="text-muted-foreground">· checked {relativeTime(s.validated_at)}</span>
        )}
      </span>
    );
  }
  if (s.ok === false) {
    return (
      <span className="text-destructive inline-flex items-center gap-1 text-xs">
        <XCircleIcon className="size-3.5" />
        {s.error ?? "Invalid"}
      </span>
    );
  }
  return (
    <span className="text-muted-foreground inline-flex items-center gap-1 text-xs">
      <CircleHelpIcon className="size-3.5" /> Not checked yet
    </span>
  );
}

function SecretRow({ s }: { s: SecretStatus }) {
  const [value, setValue] = useState("");
  const set = useSetSecret();
  const del = useDeleteSecret();
  const test = useTestSecret();
  const fromEnv = s.source === "env";

  return (
    <div className="space-y-2 border-b py-3 last:border-0" data-testid={`secret-${s.name}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="w-28 text-sm font-medium">{s.label}</span>
        <SourceBadge s={s} />
        {s.last4 && <span className="text-muted-foreground font-mono text-xs">••••{s.last4}</span>}
        <Validation s={s} />
        <div className="ml-auto flex items-center gap-1">
          <a
            href={s.token_url}
            target="_blank"
            rel="noreferrer"
            className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 px-2 text-xs"
          >
            Get a key <ExternalLinkIcon className="size-3" />
          </a>
          {s.configured && (
            <Button size="sm" variant="outline" disabled={test.isPending} onClick={() => test.mutate(s.name)}>
              {test.isPending && <Loader2Icon className="animate-spin" />}
              Test
            </Button>
          )}
        </div>
      </div>
      {s.warnings.map((w) => (
        <p key={w} className="text-warning text-xs">
          {w}
        </p>
      ))}
      {s.overridden && (
        <p className="text-muted-foreground text-xs">
          A key is also stored in Settings, but {s.env_var} takes precedence.
        </p>
      )}
      {!fromEnv && (
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            set.mutate(
              { name: s.name, value },
              {
                onSuccess: () => {
                  setValue("");
                  test.mutate(s.name);
                },
              },
            );
          }}
        >
          <Input
            type="password"
            autoComplete="off"
            aria-label={`${s.label} key`}
            placeholder={`Paste your ${s.label} key, or set ${s.env_vars.join(" / ")} in your environment`}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            className="h-8 text-xs"
          />
          <Button size="sm" type="submit" disabled={!value.trim() || set.isPending}>
            Save
          </Button>
          {s.stored && (
            <Button size="sm" type="button" variant="ghost" onClick={() => del.mutate(s.name)}>
              Remove
            </Button>
          )}
        </form>
      )}
      {fromEnv && s.stored && (
        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={() => del.mutate(s.name)}>
          Remove the stored key
        </Button>
      )}
      {s.name === "hf_token" && <ImportHfLogin fromEnv={fromEnv} />}
    </div>
  );
}

/** Copies a token saved by `hf auth login` into the keyring (LocalFlow uses its own HF_HOME). */
function ImportHfLogin({ fromEnv }: { fromEnv: boolean }) {
  const imp = useImportHfCli();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        size="sm"
        variant="outline"
        className="h-7 text-xs"
        disabled={imp.isPending}
        onClick={() => imp.mutate()}
      >
        {imp.isPending && <Loader2Icon className="animate-spin" />}
        Import existing Hugging Face login
      </Button>
      {imp.isSuccess && (
        <span className="text-success text-xs">
          Imported{fromEnv ? " — HF_TOKEN from the environment still takes precedence" : ""}.
        </span>
      )}
      {imp.error && <span className="text-destructive text-xs">{(imp.error as ApiError).message}</span>}
    </div>
  );
}

export function ApiKeysCard() {
  const { data } = useSecrets();
  const test = useTestSecret();
  const autoTested = useRef(new Set<string>());

  // Validate configured keys that haven't been checked yet (cheap calls; results are cached server-side).
  useEffect(() => {
    for (const s of data ?? []) {
      if (s.configured && s.ok === null && !autoTested.current.has(s.name)) {
        autoTested.current.add(s.name);
        test.mutate(s.name);
      }
    }
  }, [data, test]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>API keys</CardTitle>
        <p className="text-muted-foreground text-xs">
          Environment variables are used first; keys pasted here are stored in your OS keyring. Values are
          never shown again.
        </p>
      </CardHeader>
      <CardContent>
        {!data ? <Skeleton className="h-40" /> : data.map((s) => <SecretRow key={s.name} s={s} />)}
      </CardContent>
    </Card>
  );
}
