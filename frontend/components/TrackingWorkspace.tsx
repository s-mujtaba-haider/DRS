"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge, Button, Empty, KV, Panel, fmt } from "@/components/ui";
import { api, urls, type Annotation, type BallTrack, type FrameDetections, type JobResults, type TrackPoint } from "@/lib/api";

const C = { observed: "#4ade80", interp: "#fbbf24", fitted: "#22d3ee", manual: "#60a5fa", cand: "rgba(160,170,185,0.7)", bad: "#f87171" };

export interface FrameRef {
  index: number;
  ts: number;
}

export default function TrackingWorkspace({
  results,
  detections,
  annotations,
  videoId,
  onAnnotationsChanged,
  impactFrame,
  setImpactFrame,
}: {
  results: JobResults;
  detections: FrameDetections[];
  annotations: Annotation[];
  videoId: string;
  onAnnotationsChanged: () => void;
  impactFrame: number | null;
  setImpactFrame: (f: number | null) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frameIdx, setFrameIdx] = useState(0); // position in `frames`
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(0.25);
  const [annotate, setAnnotate] = useState(false);
  const [showCandidates, setShowCandidates] = useState(true);
  const [msg, setMsg] = useState<string | null>(null);

  const meta = results.metadata;
  const frames: FrameRef[] = useMemo(() => detections.map((d) => ({ index: d.frame_index, ts: d.timestamp_ms })), [detections]);
  const primary: BallTrack | undefined = results.tracking.tracks.find((t) => t.is_primary);
  const byFrame = useMemo(() => {
    const m = new Map<number, Partial<Record<TrackPoint["kind"], TrackPoint>>>();
    primary?.points.forEach((p) => m.set(p.frame_index, { ...(m.get(p.frame_index) ?? {}), [p.kind]: p }));
    return m;
  }, [primary]);
  const annByFrame = useMemo(() => {
    const m = new Map<number, Annotation>();
    annotations.forEach((a) => m.set(a.frame_index, a)); // latest wins
    return m;
  }, [annotations]);
  const cur = frames[frameIdx];
  const interval = meta.pts_delta_ms_mean ?? (meta.nominal_fps ? 1000 / meta.nominal_fps : 40);

  // ------------------------------------------------------------ sync time -> frame
  const syncFromVideo = useCallback(() => {
    const v = videoRef.current;
    if (!v || !frames.length) return;
    const tms = v.currentTime * 1000 + interval / 2;
    let lo = 0;
    let hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid].ts <= tms) lo = mid;
      else hi = mid - 1;
    }
    setFrameIdx(lo);
  }, [frames, interval]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    // requestVideoFrameCallback fires once per presented video frame (Chromium/WebKit);
    // requestAnimationFrame is the fallback elsewhere.
    const hasVfc = "requestVideoFrameCallback" in v;
    let handle = 0;
    let stop = false;
    const loop = () => {
      if (stop) return;
      syncFromVideo();
      handle = hasVfc ? v.requestVideoFrameCallback(loop) : requestAnimationFrame(loop);
    };
    loop();
    return () => {
      stop = true;
      if (hasVfc) v.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
    };
  }, [syncFromVideo]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, [rate]);

  const seekTo = useCallback(
    (i: number) => {
      const v = videoRef.current;
      if (!v || !frames.length) return;
      const k = Math.max(0, Math.min(frames.length - 1, i));
      v.pause();
      setPlaying(false);
      v.currentTime = (frames[k].ts + interval * 0.25) / 1000;
      setFrameIdx(k);
    },
    [frames, interval],
  );

  // ------------------------------------------------------------------- drawing
  useEffect(() => {
    const cv = canvasRef.current;
    const v = videoRef.current;
    if (!cv || !v || !cur) return;
    const w = v.clientWidth;
    const h = v.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    // object-fit: contain mapping
    const s = Math.min(w / meta.width, h / meta.height);
    const ox = (w - meta.width * s) / 2;
    const oy = (h - meta.height * s) / 2;
    const P = (x: number, y: number): [number, number] => [ox + x * s, oy + y * s];

    if (showCandidates) {
      const fd = detections[frameIdx];
      ctx.strokeStyle = C.cand;
      ctx.lineWidth = 1;
      fd?.detections.forEach((d) => {
        const [x1, y1] = P(d.bbox[0], d.bbox[1]);
        const [x2, y2] = P(d.bbox[2], d.bbox[3]);
        ctx.strokeRect(x1 - 2, y1 - 2, x2 - x1 + 4, y2 - y1 + 4);
      });
    }
    // trail of smoothed estimates up to the current frame
    const trail: [number, number][] = [];
    for (let k = Math.max(0, frameIdx - 40); k <= frameIdx; k++) {
      const pts = byFrame.get(frames[k].index);
      const p = pts?.filtered ?? pts?.interpolated;
      if (p) trail.push(P(p.x, p.y));
    }
    if (trail.length > 1) {
      ctx.strokeStyle = C.fitted;
      ctx.lineWidth = 2;
      ctx.beginPath();
      trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
    }
    const pts = byFrame.get(cur.index);
    const ann = annByFrame.get(cur.index);
    if (ann && (ann.action === "add" || ann.action === "move") && ann.data.x !== undefined) {
      const [x, y] = P(ann.data.x, ann.data.y!);
      ctx.strokeStyle = C.manual;
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 9, y - 9, 18, 18);
      label(ctx, "manual", x + 12, y - 10, C.manual);
    } else if (pts?.observed) {
      const [x, y] = P(pts.observed.x, pts.observed.y);
      ctx.fillStyle = C.observed;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = C.observed;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2);
      ctx.stroke();
      label(ctx, `T${primary?.track_id} ${fmt(pts.observed.confidence, 2)} ${pts.observed.detection_source ?? ""}`, x + 13, y - 12, C.observed);
    } else if (pts?.interpolated) {
      const [x, y] = P(pts.interpolated.x, pts.interpolated.y);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = C.interp;
      ctx.lineWidth = 2;
      const r = Math.max(8, 2 * (pts.interpolated.sigma_x ?? 4) * s);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      label(ctx, "interpolated (no detection)", x + r + 4, y - 8, C.interp);
    }
    if (ann && (ann.action === "reject" || ann.action === "mark_absent")) label(ctx, `manually ${ann.action === "reject" ? "rejected" : "marked absent"}`, 10, h - 12, C.bad);
  }, [frameIdx, cur, byFrame, annByFrame, detections, frames, meta.width, meta.height, showCandidates, primary]);

  // ------------------------------------------------------------- corrections
  async function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!annotate || !cur) return;
    const v = videoRef.current!;
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const s = Math.min(v.clientWidth / meta.width, v.clientHeight / meta.height);
    const ox = (v.clientWidth - meta.width * s) / 2;
    const oy = (v.clientHeight - meta.height * s) / 2;
    const x = (e.clientX - rect.left - ox) / s;
    const y = (e.clientY - rect.top - oy) / s;
    if (x < 0 || y < 0 || x > meta.width || y > meta.height) return;
    try {
      await api.addAnnotation(videoId, { frame_index: cur.index, action: byFrame.get(cur.index)?.observed ? "move" : "add", x, y, timestamp_ms: cur.ts, job_id: results.job.id });
      setMsg(`Saved manual ball position for frame ${cur.index}`);
      onAnnotationsChanged();
    } catch (err) {
      setMsg(String(err));
    }
  }

  async function rejectCurrent() {
    if (!cur) return;
    await api.addAnnotation(videoId, { frame_index: cur.index, action: "reject", job_id: results.job.id });
    setMsg(`Frame ${cur.index}: detection rejected (original output preserved)`);
    onAnnotationsChanged();
  }

  // ---------------------------------------------------------------- keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      if (e.key === "ArrowRight") seekTo(frameIdx + 1);
      else if (e.key === "ArrowLeft") seekTo(frameIdx - 1);
      else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function togglePlay() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
      setPlaying(true);
    } else {
      v.pause();
      setPlaying(false);
    }
  }

  const confData = useMemo(
    () => frames.map((f) => ({ frame: f.index, conf: byFrame.get(f.index)?.observed?.confidence ?? null, cands: detections.find((d) => d.frame_index === f.index)?.detections.length ?? 0 })),
    [frames, byFrame, detections],
  );
  const q = primary?.quality;

  return (
    <div className="grid gap-4 2xl:grid-cols-[1fr_360px]">
      <div className="space-y-3">
        <div className="relative overflow-hidden rounded-lg border border-line bg-black">
          <video
            ref={videoRef}
            src={urls.videoFile(videoId)}
            className="block aspect-video w-full object-contain"
            preload="auto"
            playsInline
            muted
            onPause={() => setPlaying(false)}
            onPlay={() => setPlaying(true)}
          />
          <canvas
            ref={canvasRef}
            onClick={onCanvasClick}
            className={`absolute inset-0 h-full w-full ${annotate ? "cursor-crosshair" : "pointer-events-none"}`}
            aria-label="Tracking overlay"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => seekTo(frameIdx - 1)} title="Previous frame (Left arrow)">Prev</Button>
          <Button variant="primary" onClick={togglePlay} title="Play/pause (Space)">{playing ? "Pause" : "Play"}</Button>
          <Button onClick={() => seekTo(frameIdx + 1)} title="Next frame (Right arrow)">Next</Button>
          <label className="flex items-center gap-1 text-sm text-muted">
            Speed
            <select className="rounded border border-line bg-panel-2 px-1.5 py-1 text-ink" value={rate} onChange={(e) => setRate(Number(e.target.value))}>
              {[0.1, 0.25, 0.5, 1].map((r) => (
                <option key={r} value={r}>{r}x</option>
              ))}
            </select>
          </label>
          <span className="tabular ml-2 text-sm">
            frame <b>{cur?.index ?? "-"}</b> <span className="text-muted">t={cur ? (cur.ts / 1000).toFixed(3) : "-"}s</span>
          </span>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-1 text-xs text-muted">
              <input type="checkbox" checked={showCandidates} onChange={(e) => setShowCandidates(e.target.checked)} /> candidates
            </label>
            <Button variant={annotate ? "primary" : "default"} onClick={() => setAnnotate(!annotate)} title="Click on the video to place the ball">
              {annotate ? "Annotating: click ball" : "Correct ball"}
            </Button>
            <Button onClick={rejectCurrent}>Reject detection</Button>
            <Button variant={impactFrame === cur?.index ? "primary" : "default"} onClick={() => cur && setImpactFrame(impactFrame === cur.index ? null : cur.index)}>
              {impactFrame === cur?.index ? "Impact frame set" : "Set as impact frame"}
            </Button>
          </div>
        </div>
        {msg && <div className="text-xs text-manual">{msg}</div>}

        <Timeline frames={frames} byFrame={byFrame} annByFrame={annByFrame} detections={detections} current={frameIdx} impactFrame={impactFrame} onSeek={seekTo} />

        <Panel title="Detection confidence (primary track)">
          {primary ? (
            <div className="h-40">
              <ResponsiveContainer>
                <LineChart data={confData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="#252c38" />
                  <XAxis dataKey="frame" stroke="#8b94a5" fontSize={11} />
                  <YAxis domain={[0, 1]} stroke="#8b94a5" fontSize={11} />
                  <Tooltip contentStyle={{ background: "#12161d", border: "1px solid #252c38", fontSize: 12 }} />
                  <Line type="monotone" dataKey="conf" stroke={C.observed} dot={false} connectNulls={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Empty>No primary track.</Empty>
          )}
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="Track quality">
          {q ? (
            <KV
              rows={[
                ["Consistency", q.reliable ? <Badge key="r" tone="good">passes checks</Badge> : <Badge key="r" tone="bad">unreliable</Badge>],
                ["Frames spanned", q.num_frames_spanned],
                ["Observed / missing", `${q.num_observed} / ${q.num_missing}`],
                ["Observed ratio", fmt(q.observed_ratio, 2)],
                ["Longest gap", `${q.longest_gap} frames`],
                ["Mean innovation", q.mean_innovation_px !== null ? `${fmt(q.mean_innovation_px, 2)} px` : null],
                ["Mean confidence", fmt(q.mean_confidence, 2)],
                ["Fragments merged", q.fragments_merged],
                ["Other tracks", results.tracking.tracks.length - 1],
              ]}
            />
          ) : (
            <Empty>No moving ball track was found.</Empty>
          )}
          {q && !q.reliable && <div className="mt-2 text-xs text-bad">{q.reliability_issues.join("; ")}</div>}
          {results.tracking.warnings.map((w) => (
            <div key={w} className="mt-2 text-xs text-warn">! {w}</div>
          ))}
        </Panel>
        <Panel title="Legend">
          <ul className="space-y-1.5 text-xs">
            <li><Swatch c={C.observed} /> Observed detection (raw model / motion output)</li>
            <li><Swatch c={C.fitted} /> Smoothed track (Kalman RTS)</li>
            <li><Swatch c={C.interp} dashed /> Interpolated: no detection in this frame</li>
            <li><Swatch c={C.manual} /> Manual correction (stored separately)</li>
            <li><Swatch c={C.cand} /> Other candidates (unassociated)</li>
          </ul>
        </Panel>
        <Panel title={`Manual corrections (${annotations.length})`}>
          {annotations.length === 0 ? (
            <Empty>None. Use &quot;Correct ball&quot; or &quot;Reject detection&quot;.</Empty>
          ) : (
            <ul className="max-h-60 space-y-1 overflow-y-auto text-xs">
              {annotations.map((a) => (
                <li key={a.id} className="flex items-center gap-2">
                  <button className="text-accent hover:underline" onClick={() => seekTo(frames.findIndex((f) => f.index === a.frame_index))}>
                    f{a.frame_index}
                  </button>
                  <Badge tone="info">{a.action}</Badge>
                  {a.data.x !== undefined && <span className="tabular text-muted">({fmt(a.data.x, 0)}, {fmt(a.data.y, 0)})</span>}
                  <button className="ml-auto text-bad hover:underline" onClick={() => api.deleteAnnotation(videoId, a.id).then(onAnnotationsChanged)}>
                    remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.font = "12px ui-monospace, monospace";
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(x - 3, y - 12, w + 6, 16);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function Swatch({ c, dashed }: { c: string; dashed?: boolean }) {
  return <span className="mr-2 inline-block h-2.5 w-4 align-middle" style={{ background: dashed ? "transparent" : c, border: dashed ? `2px dashed ${c}` : undefined }} />;
}

function Timeline({
  frames,
  byFrame,
  annByFrame,
  detections,
  current,
  impactFrame,
  onSeek,
}: {
  frames: FrameRef[];
  byFrame: Map<number, Partial<Record<TrackPoint["kind"], TrackPoint>>>;
  annByFrame: Map<number, Annotation>;
  detections: FrameDetections[];
  current: number;
  impactFrame: number | null;
  onSeek: (i: number) => void;
}) {
  if (!frames.length) return null;
  const n = frames.length;
  return (
    <div className="rounded-lg border border-line bg-panel p-3">
      <div className="mb-1.5 flex items-center justify-between text-xs text-muted">
        <span>Timeline ({n} processed frames)</span>
        <span>
          <span className="text-observed">observed</span> / <span className="text-interp">interpolated</span> / <span className="text-manual">manual</span> / grey = candidates only
        </span>
      </div>
      <svg
        viewBox={`0 0 ${n} 20`}
        preserveAspectRatio="none"
        className="h-8 w-full cursor-pointer"
        onClick={(e) => {
          const r = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
          onSeek(Math.floor(((e.clientX - r.left) / r.width) * n));
        }}
        role="slider"
        aria-valuemin={0}
        aria-valuemax={n - 1}
        aria-valuenow={current}
        aria-label="Frame timeline"
      >
        {frames.map((f, i) => {
          const p = byFrame.get(f.index);
          const a = annByFrame.get(f.index);
          const color = a ? C.manual : p?.observed ? C.observed : p?.interpolated ? C.interp : detections[i]?.detections.length ? "#3a4252" : "#161b22";
          return <rect key={f.index} x={i} y={4} width={1} height={12} fill={color} />;
        })}
        {impactFrame !== null && frames.findIndex((f) => f.index === impactFrame) >= 0 && (
          <rect x={frames.findIndex((f) => f.index === impactFrame)} y={0} width={Math.max(1, n / 300)} height={20} fill={C.bad} />
        )}
        <rect x={current} y={0} width={Math.max(1, n / 250)} height={20} fill="#ffffff" />
      </svg>
    </div>
  );
}
