import { useCallback, useRef, useState } from "react";

/** Runs one LLM step at a time with cancel support (AbortController → the server stops the call). */
export function useStepRunner() {
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<{ step: string; message: string } | null>(null);
  const ctrl = useRef<AbortController | null>(null);

  const run = useCallback(async <T>(step: string, fn: (signal: AbortSignal) => Promise<T>) => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setRunning(step);
    setError(null);
    try {
      return await fn(c.signal);
    } catch (e) {
      if (!c.signal.aborted) setError({ step, message: e instanceof Error ? e.message : String(e) });
      return undefined;
    } finally {
      if (ctrl.current === c) {
        ctrl.current = null;
        setRunning(null);
      }
    }
  }, []);

  const cancel = useCallback(() => {
    ctrl.current?.abort();
    ctrl.current = null;
    setRunning(null);
  }, []);

  return { running, error, run, cancel, clearError: () => setError(null) };
}

export type StepRunner = ReturnType<typeof useStepRunner>;
