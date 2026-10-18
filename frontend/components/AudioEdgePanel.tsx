"use client";

import { useState } from "react";

import FrameViewer, { type FrameInfo } from "@/components/FrameViewer";
import { Badge, Button, ErrorBox, Panel } from "@/components/ui";
import { api, type DecisionAnalysis, type Video } from "@/lib/api";

interface Transient { t_ms: number; snr_db: number; rise_ms: number; nearest_frame_index: number | null }
interface AudioResult {
  status: string;
  sample_rate_hz: number | null;
  highpass_hz: number | null;
  transients: Transient[];
  envelope: { t0_ms: number; dt_ms: number; db: number[] } | null;
  reasons: string[];
  warnings: string[];
}

/** Edge evidence: exact frames next to the high-frequency audio energy, with transient candidates. */
export default function AudioEdgePanel({ video, readOnly, onAnalysis }: { video: Video; readOnly: boolean; onAnalysis: (a: DecisionAnalysis) => void }) {
  const [frame, setFrame] = useState(0);
  const [info, setInfo] = useState<FrameInfo | null>(null);
  const [res, setRes] = useState<AudioResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const capAudio = video.meta.capture?.audio;
  const hasAudio = video.meta.has_audio || !!capAudio;

  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      const a = await api.audioContact({ video_id: video.id, center_ms: info?.timestampMs ?? null, half_window_ms: 400 });
      setRes(a.result as unknown as AudioResult);
      onAnalysis(a);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  const env = res?.envelope;
  const W = 1000, H = 120;
  let path = "";
  let x: (t: number) => number = () => 0;
  if (env && env.db.length > 1) {
    const lo = Math.min(...env.db), hi = Math.max(...env.db);
    const t1 = env.t0_ms + env.dt_ms * (env.db.length - 1);
    x = (t: number) => ((t - env.t0_ms) / (t1 - env.t0_ms)) * W;
    path = env.db.map((v, i) => `${i ? "L" : "M"}${(i / (env.db.length - 1)) * W},${H - ((v - lo) / (hi - lo || 1)) * (H - 10) - 5}`).join(" ");
  }
  const cursor = info?.timestampMs;

  return (
    <Panel title="Edge evidence: frames + audio (experimental)">
      {!hasAudio && <div className="mb-2 text-sm text-warn">No audio for this replay. Record a stump microphone in Live capture so clips carry clock-aligned audio.</div>}
      {capAudio && (
        <div className="mb-2 text-xs text-muted">
          Capture microphone, aligned on the shared clock (±{capAudio.uncertainty_ms.toFixed(1)} ms, {capAudio.timestamp_basis ?? "unknown"} timestamps).
        </div>
      )}
      <FrameViewer videoId={video.id} width={video.meta.width} height={video.meta.height}
        frameCount={video.meta.frame_count_decoded ?? video.meta.frame_count_declared} frame={frame} setFrame={setFrame} onFrameInfo={setInfo} />
      <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
        {!readOnly && <Button variant="primary" disabled={!hasAudio || busy || cursor == null} onClick={run}>{busy ? "Analysing..." : "Analyse audio ±400 ms around this frame"}</Button>}
        {res && <Badge tone={res.status === "TRANSIENTS_FOUND" ? "info" : "neutral"}>{res.status}</Badge>}
        {res?.sample_rate_hz && <span className="text-xs text-muted">high-pass {res.highpass_hz?.toFixed(0)} Hz</span>}
      </div>
      <ErrorBox error={err} />
      {env && (
        <div className="mt-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="h-28 w-full rounded border border-line bg-panel-2" preserveAspectRatio="none" role="img" aria-label="High-frequency audio energy">
            <path d={path} fill="none" stroke="#38bdf8" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
            {res?.transients.map((t, i) => (
              <g key={i} className="cursor-pointer" onClick={() => t.nearest_frame_index != null && setFrame(t.nearest_frame_index)}>
                <line x1={x(t.t_ms)} x2={x(t.t_ms)} y1={0} y2={H} stroke="#fb923c" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                <title>{`${t.t_ms.toFixed(1)} ms, ${t.snr_db} dB, rise ${t.rise_ms} ms`}</title>
              </g>
            ))}
            {cursor != null && <line x1={x(cursor)} x2={x(cursor)} y1={0} y2={H} stroke="#e6e9ef" strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />}
          </svg>
          <ul className="mt-1 space-y-0.5 text-xs">
            {res?.transients.map((t, i) => (
              <li key={i}>
                <button className="tabular text-accent hover:underline" onClick={() => t.nearest_frame_index != null && setFrame(t.nearest_frame_index)}>
                  {t.t_ms.toFixed(1)} ms → frame {t.nearest_frame_index ?? "?"}
                </button>{" "}
                <span className="text-muted">{t.snr_db} dB above floor, rise {t.rise_ms} ms</span>
              </li>
            ))}
          </ul>
          {res && <ul className="mt-1 list-disc pl-5 text-xs text-warn">{res.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
        </div>
      )}
    </Panel>
  );
}
