import { useCallback, useEffect, useRef, useState } from "react";

/**
 * The preview's clock. Time advances with `requestAnimationFrame` while playing, and the media elements
 * follow it (Section 15.2: approximate preview; the export is authoritative). One clock for every clip
 * makes cuts, crossfades and audio offsets line up without chaining `ended` events.
 */
export function usePlayback(duration: number) {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timeRef = useRef(0);
  const anchor = useRef({ base: 0, at: 0 });

  useEffect(() => {
    if (!playing) return;
    anchor.current = { base: timeRef.current, at: performance.now() };
    let raf = 0;
    const tick = (now: number) => {
      const t = anchor.current.base + (now - anchor.current.at) / 1000;
      if (t >= duration) {
        timeRef.current = duration;
        setTime(duration);
        setPlaying(false);
        return;
      }
      timeRef.current = t;
      setTime(t);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, duration]);

  const seek = useCallback(
    (t: number) => {
      const next = Math.max(0, Math.min(t, duration));
      timeRef.current = next;
      anchor.current = { base: next, at: performance.now() };
      setTime(next);
    },
    [duration],
  );

  const play = useCallback(() => {
    if (duration <= 0) return;
    if (timeRef.current >= duration - 1e-3) seek(0);
    setPlaying(true);
  }, [duration, seek]);

  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => (playing ? pause() : play()), [playing, pause, play]);

  // Keep the playhead inside the timeline when it gets shorter.
  useEffect(() => {
    if (timeRef.current > duration) seek(duration);
  }, [duration, seek]);

  return { time, playing, play, pause, toggle, seek };
}
