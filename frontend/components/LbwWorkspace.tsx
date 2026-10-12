"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Badge, Button, Empty, ErrorBox, KV, Panel, fmt, type Tone } from "@/components/ui";
import { api, urls, type AutoCalibration, type Calibration, type JobResults, type LbwAnalysis, type LbwFacts, type LbwResult, type Verdict } from "@/lib/api";

const VERDICT: Record<Verdict, { tone: Tone; label: string }> = {
  PROJECTED_TO_HIT: { tone: "bad", label: "Projected to hit wicket" },
  PROJECTED_TO_MISS: { tone: "good", label: "Projected to miss wicket" },
  INCONCLUSIVE: { tone: "warn", label: "Inconclusive" },
  INSUFFICIENT_DATA: { tone: "neutral", label: "Insufficient data" },
};
const CHECK_TONE: Record<string, Tone> = { PASS: "good", FAIL: "bad", UNKNOWN: "warn", NOT_APPLICABLE: "neutral" };

type Pt = { name: string; image: [number, number] };

export default function LbwWorkspace({
  results,
  videoId,
  impactFrame,
  analyses,
  onNewAnalysis,
}: {
  results: JobResults;
  videoId: string;
  impactFrame: number | null;
  analyses: LbwAnalysis[];
  onNewAnalysis: () => void;
}) {
  const meta = results.metadata;
  const primary = results.tracking.tracks.find((t) => t.is_primary);
  const observed = primary?.points.filter((p) => p.kind === "observed") ?? [];
  const defaultFrame = impactFrame ?? observed[Math.floor(observed.length / 2)]?.frame_index ?? 0;
  const [frame, setFrame] = useState(defaultFrame);
  const [landmarks, setLandmarks] = useState<string[]>([]);
  const [selLm, setSelLm] = useState("striker_off_stump_base_rh");
  const [points, setPoints] = useState<Pt[]>([]);
  const [cals, setCals] = useState<Calibration[]>([]);
  const [calId, setCalId] = useState<string>("");
  const [facts, setFacts] = useState<LbwFacts>({ batter_handedness: null, no_ball: null, ball_touched_bat_first: null, genuine_attempt_to_play: null, intercepted_by_person_not_bat_hand: null });
  const [requireValidated, setRequireValidated] = useState(true);
  const [confidence, setConfidence] = useState(0.95);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected: LbwAnalysis | null = analyses.find((a) => a.id === selectedId) ?? analyses[0] ?? null;
  const [busy, setBusy] = useState(false);
  const [auto, setAuto] = useState<AutoCalibration | null>(null);
  const [error, setError] = useState<unknown>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgLoaded, setImgLoaded] = useState(0);

  useEffect(() => {
    api.landmarks().then((l) => setLandmarks(Object.keys(l.landmarks))).catch(setError);
    api.calibrations(videoId).then((c) => {
      setCals(c);
      if (c[0]) setCalId(c[0].id);
    });
  }, [videoId]);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      setImgLoaded((n) => n + 1);
    };
    img.onerror = () => setError(new Error(`Could not load frame ${frame}`));
    img.src = urls.frame(videoId, frame);
  }, [videoId, frame]);

  const cal = cals.find((c) => c.id === calId) ?? null;
  const result: LbwResult | null = selected?.result ?? null;

  // ---------------------------------------------------------------- drawing
  useEffect(() => {
    const cv = canvasRef.current;
    const img = imgRef.current;
    if (!cv || !img) return;
    cv.width = meta.width;
    cv.height = meta.height;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(img, 0, 0, meta.width, meta.height);
    const lw = Math.max(2, meta.width / 640);
    const poly = (pts: (number[] | null)[] | undefined, color: string, dash: number[] = [], close = false) => {
      const ok = (pts ?? []).filter((p): p is number[] => !!p);
      if (ok.length < 2) return;
      ctx.setLineDash(dash.map((d) => d * lw));
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      ok.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (close) ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    };
    const dot = (p: number[] | null | undefined, color: string, r = 5) => {
      if (!p) return;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p[0], p[1], r * lw * 0.6, 0, Math.PI * 2);
      ctx.fill();
    };
    if (result?.overlay) {
      const o = result.overlay;
      poly(o.wicket_outline_px, "#e6e9ef", [], true);
      poly(o.striker_crease_px, "#e6e9ef", [4, 3]);
      o.observations_px?.forEach((p) => dot(p, "#4ade80", 3));
      poly(o.fitted_path_px, "#22d3ee");
      poly(o.predicted_path_px, "#e879f9", [6, 4]);
      poly(o.stump_crossing_2sigma_px, "#fb923c", [3, 3], true);
      dot(o.pitching_point_px, "#fbbf24", 7);
      dot(o.impact_point_px, "#f87171", 7);
      dot(o.stump_crossing_px, "#e879f9", 6);
    }
    points.forEach((p) => {
      ctx.strokeStyle = "#60a5fa";
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.arc(p.image[0], p.image[1], 6 * lw * 0.6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "#60a5fa";
      ctx.font = `${12 * lw * 0.7}px monospace`;
      ctx.fillText(p.name.replace(/_rh$/, ""), p.image[0] + 8, p.image[1] - 8);
    });
  }, [imgLoaded, points, result, meta.width, meta.height]);

  function onClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * meta.width;
    const y = ((e.clientY - r.top) / r.height) * meta.height;
    setPoints((ps) => [...ps.filter((p) => p.name !== selLm), { name: selLm, image: [x, y] }]);
    const next = landmarks[(landmarks.indexOf(selLm) + 1) % landmarks.length];
    if (next) setSelLm(next);
  }

  async function autoDetect(thisFrameOnly: boolean) {
    setError(null);
    setBusy(true);
    try {
      const a = await api.autoCalibration({ video_id: videoId, ...(thisFrameOnly ? { frame_index: frame } : {}) });
      setAuto(a);
      if (a.frame_index >= 0 && a.frame_index !== frame) setFrame(a.frame_index);
      if (a.points.length) setPoints(a.points);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  async function computeCalibration() {
    setError(null);
    setBusy(true);
    try {
      const c = await api.createCalibration({ video_id: videoId, points });
      setCals((cs) => [c, ...cs]);
      setCalId(c.id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  async function run() {
    if (!calId) return;
    setError(null);
    setBusy(true);
    try {
      const a = await api.lbwAnalyze({
        job_id: results.job.id,
        calibration_id: calId,
        impact_frame: impactFrame ?? undefined,
        facts,
        decision_confidence: confidence,
        require_validated_calibration_for_definite: requireValidated,
      });
      setSelectedId(a.id);
      onNewAnalysis();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  const tri = (k: keyof LbwFacts, labels: [string, string] = ["yes", "no"]) => (
    <select
      className="rounded border border-line bg-panel-2 px-1.5 py-1 text-sm"
      value={facts[k] === null ? "" : String(facts[k])}
      onChange={(e) => setFacts({ ...facts, [k]: e.target.value === "" ? null : e.target.value === "true" })}
    >
      <option value="">unknown</option>
      <option value="true">{labels[0]}</option>
      <option value="false">{labels[1]}</option>
    </select>
  );

  return (
    <div className="space-y-4">
      <div className="rounded border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
        Experimental analysis. Geometry from a single camera relies on calibration and physical assumptions; results are not umpiring decisions and real-world accuracy is unknown.
      </div>
      <ErrorBox error={error} />
      <div className="grid gap-4 2xl:grid-cols-[1fr_420px]">
        <div className="space-y-3">
          <div className="overflow-hidden rounded-lg border border-line bg-black">
            <canvas ref={canvasRef} onClick={onClick} className="block h-auto w-full cursor-crosshair" aria-label="Calibration and LBW overlay" />
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted">Frame</span>
            <Button onClick={() => setFrame(Math.max(0, frame - 1))}>-1</Button>
            <input type="number" className="w-24 rounded border border-line bg-panel-2 px-2 py-1" value={frame} min={0} onChange={(e) => setFrame(Number(e.target.value))} />
            <Button onClick={() => setFrame(frame + 1)}>+1</Button>
            {impactFrame !== null && <Button onClick={() => setFrame(impactFrame)}>Go to impact frame</Button>}
            <span className="ml-auto text-xs text-muted">
              <span className="text-observed">observed</span> / <span className="text-fitted">fitted</span> / <span className="text-predicted">predicted</span> /{" "}
              <span className="text-interp">pitching</span> / <span className="text-bad">impact</span> / <span className="text-warn">2-sigma at stumps</span>
            </span>
          </div>
          {result && <ResultPanel result={result} handed={facts.batter_handedness} />}
        </div>

        <div className="space-y-4">
          <Panel title="1. Camera calibration">
            <div className="space-y-3 text-sm">
              {cals.length > 0 && (
                <label className="block space-y-1">
                  <span className="text-xs text-muted">Use calibration</span>
                  <select className="w-full rounded border border-line bg-panel-2 px-2 py-1.5" value={calId} onChange={(e) => setCalId(e.target.value)}>
                    {cals.map((c) => (
                      <option key={c.id} value={c.id}>
                        {new Date(c.created_at).toLocaleTimeString()} - {c.camera.calibration_type} - {fmt(c.camera.reprojection_rms_px, 2)} px
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {cal && (
                <div className="space-y-1">
                  <Badge tone={cal.camera.validated ? "good" : "warn"}>{cal.camera.validated ? "validated intrinsics" : "approximate"}</Badge>
                  <KV rows={[["Reprojection RMS", `${fmt(cal.camera.reprojection_rms_px, 2)} px`], ["Points", cal.camera.num_correspondences], ["Focal (px)", fmt(cal.camera.K[0][0], 0)]]} />
                  {[...cal.camera.assumptions, ...cal.camera.warnings].map((a) => (
                    <div key={a} className="text-xs text-warn">! {a}</div>
                  ))}
                </div>
              )}
              <div className="border-t border-line pt-3">
                <div className="mb-2 space-y-1.5">
                  <div className="flex flex-wrap gap-2">
                    <Button disabled={busy} onClick={() => autoDetect(false)}>Auto-detect (scan clip)</Button>
                    <Button variant="ghost" disabled={busy} onClick={() => autoDetect(true)}>Auto-detect (this frame)</Button>
                  </div>
                  {auto && (
                    <div className="space-y-0.5 rounded border border-line bg-panel-2 p-2 text-xs">
                      <div>
                        <Badge tone={auto.method === "seeds_only" ? "warn" : "neutral"}>
                          {auto.method === "two_wicket_stumps" ? "both wickets found" : auto.method === "crease_registration" ? "stumps + creases" : "seed points only"}
                        </Badge>{" "}
                        frame {auto.frame_index}, {auto.points.length} points placed{auto.focal_px ? `, focal ${fmt(auto.focal_px, 0)} px` : ""}
                      </div>
                      <div className="text-warn">! {auto.note}</div>
                      {auto.warnings.map((w) => (
                        <div key={w} className="text-warn">! {w}</div>
                      ))}
                    </div>
                  )}
                </div>
                <div className="mb-1 text-xs text-muted">New calibration: pick a landmark, then click it on the frame (clicking replaces an auto-placed point). At least 6 points (including stump tops) are needed without known intrinsics.</div>
                <select className="w-full rounded border border-line bg-panel-2 px-2 py-1.5" value={selLm} onChange={(e) => setSelLm(e.target.value)}>
                  {landmarks.map((l) => (
                    <option key={l} value={l}>{l}{points.some((p) => p.name === l) ? " (placed)" : ""}</option>
                  ))}
                </select>
                <ul className="mt-2 max-h-36 space-y-0.5 overflow-y-auto text-xs">
                  {points.map((p) => (
                    <li key={p.name} className="flex justify-between">
                      <span>{p.name}</span>
                      <button className="text-bad" onClick={() => setPoints(points.filter((q) => q.name !== p.name))}>remove</button>
                    </li>
                  ))}
                </ul>
                <div className="mt-2 flex gap-2">
                  <Button variant="primary" disabled={points.length < 6 || busy} onClick={computeCalibration}>Compute ({points.length})</Button>
                  <Button variant="ghost" onClick={() => setPoints([])}>Clear</Button>
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="2. Facts and settings">
            <div className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-2 text-sm">
              <span className="text-muted">Batter handedness</span>
              <select className="rounded border border-line bg-panel-2 px-1.5 py-1" value={facts.batter_handedness ?? ""} onChange={(e) => setFacts({ ...facts, batter_handedness: (e.target.value || null) as LbwFacts["batter_handedness"] })}>
                <option value="">unknown</option>
                <option value="right">right</option>
                <option value="left">left</option>
              </select>
              <span className="text-muted">No ball?</span>
              {tri("no_ball")}
              <span className="text-muted">Ball hit bat first?</span>
              {tri("ball_touched_bat_first")}
              <span className="text-muted">Genuine attempt to play?</span>
              {tri("genuine_attempt_to_play")}
              <span className="text-muted">Intercepted by body (not bat/glove)?</span>
              {tri("intercepted_by_person_not_bat_hand")}
              <span className="text-muted">Impact frame</span>
              <span className="tabular">{impactFrame ?? <span className="text-warn">not set (last observation used)</span>}</span>
              <span className="text-muted">Decision confidence</span>
              <input type="number" step="0.01" min="0.5" max="0.99" className="w-20 rounded border border-line bg-panel-2 px-1.5 py-1" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} />
              <span className="text-muted">Require validated calibration for definite result</span>
              <input type="checkbox" checked={requireValidated} onChange={(e) => setRequireValidated(e.target.checked)} />
            </div>
            <div className="mt-3">
              <Button variant="primary" disabled={!calId || busy || !primary} onClick={run}>{busy ? "Running..." : "Run LBW analysis"}</Button>
              {!primary && <div className="mt-2 text-xs text-warn">No ball track available for this session.</div>}
            </div>
          </Panel>

          {analyses.length > 0 && (
            <Panel title="History">
              <ul className="space-y-1 text-xs">
                {analyses.map((a) => (
                  <li key={a.id}>
                    <button className={`flex w-full items-center gap-2 rounded px-1 py-0.5 text-left ${selected?.id === a.id ? "bg-panel-2" : ""}`} onClick={() => setSelectedId(a.id)}>
                      <span className="tabular text-muted">{new Date(a.created_at).toLocaleTimeString()}</span>
                      <Badge tone={VERDICT[a.result.verdict].tone}>{a.result.verdict}</Badge>
                      <span className="tabular ml-auto">P(hit) {fmt(a.result.p_hit, 3)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultPanel({ result, handed }: { result: LbwResult; handed: string | null }) {
  const v = VERDICT[result.verdict];
  const wp = result.wicket_projection;
  return (
    <Panel title="Result" actions={<span className="text-[11px] text-muted">{result.engine_version}</span>}>
      <div className="grid gap-4 lg:grid-cols-[1fr_260px]">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <span className={`rounded border px-3 py-1.5 text-base font-semibold ${v.tone === "bad" ? "border-bad/50 text-bad" : v.tone === "good" ? "border-observed/50 text-observed" : "border-warn/50 text-warn"}`}>
              {v.label}
            </span>
            <span className="tabular text-sm">P(hit) = <b>{fmt(result.p_hit, 3)}</b></span>
            <Badge tone={result.calibration_validated ? "good" : "warn"}>{result.reconstruction_mode === "monocular_approximate" ? "single camera, approximate" : result.reconstruction_mode}</Badge>
            <Badge tone="neutral">rules: {result.rules_summary}</Badge>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-xs">
              <thead className="text-left uppercase tracking-wider text-muted">
                <tr>
                  <th className="pb-1">Condition</th>
                  <th className="pb-1">Law</th>
                  <th className="pb-1">Status</th>
                  <th className="pb-1">Evidence</th>
                </tr>
              </thead>
              <tbody>
                {result.rule_checks.map((c) => (
                  <tr key={c.id} className="border-t border-line align-top">
                    <td className="py-1 pr-2">{c.description}</td>
                    <td className="whitespace-nowrap pr-2 text-muted">{c.law_reference}</td>
                    <td className="pr-2"><Badge tone={CHECK_TONE[c.status]}>{c.status}</Badge></td>
                    <td className="text-muted">{c.evidence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <KV
            rows={[
              ["Impact time", result.impact_time_s !== null ? `${fmt(result.impact_time_s, 3)} s (${result.impact_time_source})` : null],
              ["Projection distance", wp?.projection_distance_m ? `${fmt(wp.projection_distance_m, 2)} m` : null],
              ["Stump-plane x, z (mean)", wp?.crossing ? `${fmt(wp.crossing.mean[0], 3)}, ${fmt(wp.crossing.mean[1], 3)} m` : null],
              ["Stump-plane 1-sigma", wp?.crossing ? `${fmt(wp.crossing.std[0] * 100, 1)} cm, ${fmt(wp.crossing.std[1] * 100, 1)} cm` : null],
              ["Model uncertainty", (result.model_uncertainty as { stump_plane_shift_with_alt_drag_m?: number }).stump_plane_shift_with_alt_drag_m !== undefined ? `${fmt(((result.model_uncertainty as { stump_plane_shift_with_alt_drag_m: number }).stump_plane_shift_with_alt_drag_m) * 100, 1)} cm shift under alternative drag` : null],
              ["Fit residual", `${fmt(result.input_quality.residual_rms as number, 2)} ${result.input_quality.residual_unit}`],
              ["Observations", String(result.input_quality.num_observations ?? "n/a")],
            ]}
          />
          {result.reasons.length > 0 && (
            <ul className="space-y-0.5 text-xs text-muted">
              {result.reasons.map((r, i) => <li key={i}>- {r}</li>)}
            </ul>
          )}
          {result.warnings.map((w, i) => <div key={i} className="text-xs text-warn">! {w}</div>)}
          <p className="text-[11px] text-muted">{result.disclaimer}</p>
        </div>
        <WicketView result={result} handed={handed} />
      </div>
    </Panel>
  );
}

/** Stump-plane view from the bowler's end. Pitch +x (bowler's left) is drawn on the left. */
function WicketView({ result, handed }: { result: LbwResult; handed: string | null }) {
  const c = result.wicket_projection?.crossing;
  const W = 0.2286;
  const H = 0.7112;
  const r = 0.0364;
  const sx = (x: number) => 130 - x * 300; // flip: +x appears on the viewer's left
  const sz = (z: number) => 250 - z * 300;
  const offLeft = handed === "right" ? "off" : handed === "left" ? "leg" : "";
  const ellipse = useMemo(() => {
    if (!c) return null;
    return { cx: sx(c.mean[0]), cy: sz(c.mean[1]), rx: Math.max(1, 2 * c.std[0] * 300), ry: Math.max(1, 2 * c.std[1] * 300) };
  }, [c]);
  return (
    <div>
      <svg viewBox="0 0 260 280" className="w-full rounded border border-line bg-panel-2" role="img" aria-label="Projected stump-plane crossing">
        <line x1="0" y1={sz(0)} x2="260" y2={sz(0)} stroke="#3a4252" />
        {[-1, 0, 1].map((k) => (
          <rect key={k} x={sx(k * (W / 2 - 0.0175)) - 5} y={sz(H)} width="10" height={H * 300} fill="#d8dce4" />
        ))}
        <rect x={sx(W / 2)} y={sz(H + 0.0127)} width={W * 300} height="4" fill="#fbbf24" />
        {ellipse && <ellipse {...ellipse} fill="rgba(251,146,60,0.18)" stroke="#fb923c" strokeDasharray="3 3" />}
        {c && <circle cx={sx(c.mean[0])} cy={sz(c.mean[1])} r={r * 300} fill="rgba(232,121,249,0.55)" stroke="#e879f9" />}
        <text x="6" y="272" fontSize="10" fill="#8b94a5">{offLeft && `${offLeft} side`}</text>
        <text x="254" y="272" fontSize="10" fill="#8b94a5" textAnchor="end">{offLeft && (offLeft === "off" ? "leg side" : "off side")}</text>
      </svg>
      <div className="mt-1 text-[11px] text-muted">View from the bowler&apos;s end. Magenta: projected ball (mean). Orange: 2-sigma region.</div>
      {!c && <Empty>No projection.</Empty>}
    </div>
  );
}
