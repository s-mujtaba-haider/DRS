"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Badge, Button } from "@/components/ui";
import { urls, type Video } from "@/lib/api";

/** Synchronised multi-camera replay on the shared capture clock.
 *  shared_ms = clip_ms + meta.capture.time_offset_ms. Clips without capture timing (uploads) are
 *  shown at offset 0 and flagged as not time-aligned. */
export default function ReplayMonitor({ videos, big = false }: { videos: Video[]; big?: boolean }) {
  const refs = useRef<(HTMLVideoElement | null)[]>([]);
  const box = useRef<HTMLDivElement>(null);
  const [t, setT] = useState(0); // shared clock, ms
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(0.25);

  const group = videos.find((v) => v.meta.capture)?.meta.capture?.clip_group;
  const cams = useMemo(
    () =>
      videos.map((v) => {
        const cap = v.meta.capture;
        const fps = v.meta.nominal_fps || 25;
        return {
          v,
          offset: cap?.time_offset_ms ?? 0,
          aligned: !!cap && cap.clip_group === group,
          fps,
          dur: (v.meta.duration_s ?? 0) * 1000,
          unc: cap?.sync_uncertainty_ms ?? null,
        };
      }),
    [videos, group],
  );
  const t0 = Math.min(...cams.map((c) => c.offset), 0);
  const t1 = Math.max(...cams.map((c) => c.offset + c.dur), 1);
  const step = Math.min(...cams.map((c) => 1000 / c.fps));

  const seekAll = useCallback(
    (ms: number) => {
      const tt = Math.max(t0, Math.min(t1, ms));
      cams.forEach((c, i) => {
        const el = refs.current[i];
        if (el) el.currentTime = Math.max(0, (tt - c.offset) / 1000);
      });
      setT(tt);
    },
    [cams, t0, t1],
  );

  useEffect(() => {
    refs.current.forEach((el) => el && (el.playbackRate = rate));
  }, [rate]);

  useEffect(() => {
    if (!playing) return;
    const els = refs.current;
    let h = 0;
    const loop = () => {
      const master = refs.current[0];
      if (master) {
        const tt = master.currentTime * 1000 + cams[0].offset;
        cams.forEach((c, i) => {
          const el = refs.current[i];
          if (i === 0 || !el) return;
          const want = (tt - c.offset) / 1000;
          if (want >= 0 && Math.abs(el.currentTime - want) > 0.5 / c.fps) el.currentTime = want;
        });
        setT(tt);
        if (master.ended) setPlaying(false);
      }
      h = requestAnimationFrame(loop);
    };
    refs.current.forEach((el) => {
      if (el) {
        el.playbackRate = rate;
        void el.play().catch(() => undefined);
      }
    });
    h = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(h);
      els.forEach((el) => el?.pause());
    };
  }, [playing, cams, rate]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest("input,textarea,select")) return;
      if (e.key === " ") {
        setPlaying((p) => !p);
        e.preventDefault();
      } else if (e.key === "." || e.key === ",") {
        setPlaying(false);
        seekAll(t + (e.key === "." ? step : -step));
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [seekAll, step, t]);

  if (!videos.length) return null;
  const cols = videos.length === 1 ? "grid-cols-1" : videos.length <= 4 ? "md:grid-cols-2" : "md:grid-cols-3";
  const maxUnc = Math.max(0, ...cams.map((c) => c.unc ?? 0));

  return (
    <div ref={box} className="space-y-2 bg-bg">
      <div className={`grid gap-2 ${cols}`}>
        {cams.map((c, i) => {
          const local = (t - c.offset) / 1000;
          return (
            <div key={c.v.id} className="relative overflow-hidden rounded border border-line bg-black">
              <video
                ref={(el) => {
                  refs.current[i] = el;
                }}
                src={urls.videoFile(c.v.id)}
                muted
                playsInline
                preload="auto"
                className={`w-full ${big ? "max-h-[80vh]" : ""}`}
              />
              <div className="absolute left-1 top-1 flex flex-wrap gap-1 text-[11px]">
                <Badge tone="info">{c.v.meta.capture?.camera_id ?? c.v.original_filename}</Badge>
                <Badge>{c.fps.toFixed(0)} fps</Badge>
                {local >= 0 && local * 1000 <= c.dur ? <Badge>f{Math.round(local * c.fps)}</Badge> : <Badge tone="warn">no frame</Badge>}
                {!c.aligned && videos.length > 1 && <Badge tone="warn">not time-aligned</Badge>}
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Button variant="primary" onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</Button>
        <Button onClick={() => { setPlaying(false); seekAll(t - step); }} title=", key">◀ frame</Button>
        <Button onClick={() => { setPlaying(false); seekAll(t + step); }} title=". key">frame ▶</Button>
        <select className="rounded border border-line bg-panel-2 px-2 py-1" value={rate} onChange={(e) => setRate(Number(e.target.value))} aria-label="Speed">
          {[0.05, 0.1, 0.25, 0.5, 1].map((r) => <option key={r} value={r}>{r}×</option>)}
        </select>
        <input type="range" min={t0} max={t1} step={step / 4} value={t} onChange={(e) => { setPlaying(false); seekAll(Number(e.target.value)); }} className="min-w-[160px] flex-1" aria-label="Shared time" />
        <span className="tabular text-muted">{(t / 1000).toFixed(3)} s</span>
        {videos.length > 1 && <Badge tone="warn">software sync ±{maxUnc.toFixed(0)} ms</Badge>}
        {!big && (
          <Button variant="ghost" onClick={() => box.current?.requestFullscreen?.()}>Full screen</Button>
        )}
      </div>
    </div>
  );
}
