"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui";
import { api } from "@/lib/api";

/** "Replay last N seconds" from every recording camera, attached to a review. */
export default function CaptureClipButton({ reviewId, onClipped }: { reviewId?: string; onClipped?: () => void }) {
  const [live, setLive] = useState(0);
  const [seconds, setSeconds] = useState(8);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    const poll = () => api.captureStatus().then((s) => setLive(s.cameras.filter((c) => c.running).length)).catch(() => setLive(0));
    poll();
    const t = setInterval(poll, 5000);
    return () => clearInterval(t);
  }, []);

  if (!live) return <span className="text-xs text-muted">No live cameras</span>;
  const clip = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.captureClip({ seconds, review_id: reviewId ?? null });
      const unc = r.sync?.max_sync_uncertainty_ms;
      setMsg(`${r.videos.length} clip(s)${unc != null ? `, software sync ±${unc.toFixed(1)} ms` : ""}`);
      onClipped?.();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex items-center gap-2 text-xs">
      {msg && <span className="text-muted">{msg}</span>}
      <select className="rounded border border-line bg-panel-2 px-1.5 py-1" value={seconds} onChange={(e) => setSeconds(Number(e.target.value))} aria-label="Clip length">
        {[4, 6, 8, 12, 20, 30].map((s) => <option key={s} value={s}>last {s} s</option>)}
      </select>
      <Button variant="primary" disabled={busy} onClick={clip}>{busy ? "Clipping..." : `Clip ${live} camera(s)`}</Button>
    </div>
  );
}
