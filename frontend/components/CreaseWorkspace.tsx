"use client";

import { useEffect, useMemo, useState } from "react";

import FrameViewer, { type FrameInfo } from "@/components/FrameViewer";
import { Badge, Button, ErrorBox, Panel, type Tone } from "@/components/ui";
import { api, type Calibration, type CreaseProposal, type CreaseProposals, type DecisionAnalysis, type Video } from "@/lib/api";

type Pt = [number, number];
type Tool = "crease" | "behind" | "grounded" | "airborne" | "roi" | "none";
type Tri = boolean | null;
interface Obs { frame_index: number; timestamp_ms: number; point: Pt | null; grounded: Tri; source: "manual" }
interface Mark { frame: number; ts: number }

const TOOL_HELP: Record<Tool, string> = {
  crease: "Click two points on the BACK edge of the popping crease marking",
  behind: "Click any point on the stump side of the crease",
  grounded: "Click the bat tip / foot point that is touching the ground",
  airborne: "Click the bat tip / foot point that is NOT touching the ground",
  roi: "Click two opposite corners of a box around the stumps and bails",
  none: "",
};
const STATE_COLOR: Record<string, string> = { IN: "#4ade80", NOT_IN_BEYOND: "#f87171", AIRBORNE_BEHIND: "#fbbf24", UNKNOWN: "#fb923c" };

export function statusToneOf(s?: string): Tone {
  if (!s) return "neutral";
  if (/INCONCLUSIVE|INSUFFICIENT|AMBIG/.test(s)) return "warn";
  if (/OUT_OF_GROUND|INVALID|NOT_MET/.test(s)) return "bad";
  return "good";
}

function extend(a: Pt, b: Pt, w: number, h: number): [Pt, Pt] {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L = Math.hypot(dx, dy) || 1;
  const k = (w + h) / L;
  return [[a[0] - dx * k, a[1] - dy * k], [b[0] + dx * k, b[1] + dy * k]];
}

function TriSelect({ label, value, onChange }: { label: string; value: Tri; onChange: (v: Tri) => void }) {
  return (
    <label className="flex items-center justify-between gap-2 text-xs">
      <span className="text-muted">{label}</span>
      <select className="rounded border border-line bg-panel-2 px-1.5 py-0.5" value={value === null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : e.target.value === "true")}>
        <option value="">unknown</option><option value="true">yes</option><option value="false">no</option>
      </select>
    </label>
  );
}

export default function CreaseWorkspace({ video, mode, readOnly, onAnalysis }: {
  video: Video; mode: "run_out" | "stumping"; readOnly: boolean; onAnalysis: (a: DecisionAnalysis) => void;
}) {
  const W = video.meta.width, H = video.meta.height;
  const [frame, setFrame] = useState(0);
  const [info, setInfo] = useState<FrameInfo | null>(null);
  const [tool, setTool] = useState<Tool>("crease");
  const [crease, setCrease] = useState<Pt[]>([]);
  const [behind, setBehind] = useState<Pt | null>(null);
  const [roi, setRoi] = useState<Pt[]>([]);
  const [obs, setObs] = useState<Obs[]>([]);
  const [intact, setIntact] = useState<Mark | null>(null);
  const [broken, setBroken] = useState<Mark | null>(null);
  const [breakSource, setBreakSource] = useState<"manual" | "auto">("manual");
  const [confirmed, setConfirmed] = useState(false);
  const [followCam, setFollowCam] = useState(false);
  const [cals, setCals] = useState<Calibration[]>([]);
  const [calId, setCalId] = useState<string>("");
  const [end, setEnd] = useState<"striker" | "bowler">("striker");
  const [facts, setFacts] = useState<Record<string, Tri | string>>({ batter: null, no_ball: null, wicket_broken_fairly: null,
    ball_touched_fielder_before_break: null, attempting_run: null, broken_by_wicket_keeper_without_intervention: null,
    running_or_diving_into_ground: null, left_ground_to_avoid_injury: null });
  const [result, setResult] = useState<DecisionAnalysis | null>(null);
  const [props, setProps] = useState<Map<number, CreaseProposal[]>>(new Map());
  const [propInfo, setPropInfo] = useState<CreaseProposals | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);

  useEffect(() => { api.calibrations(video.id).then(setCals).catch(() => undefined); }, [video.id]);

  const ts = info?.index === frame ? info.timestampMs : null;
  const click = (x: number, y: number) => {
    if (readOnly) return;
    const p: Pt = [x, y];
    if (tool === "crease") setCrease((c) => (c.length >= 2 ? [p] : [...c, p]));
    else if (tool === "behind") setBehind(p);
    else if (tool === "roi") setRoi((r) => (r.length >= 2 ? [p] : [...r, p]));
    else if ((tool === "grounded" || tool === "airborne") && ts !== null) {
      // Snap to a model proposal when the click lands on one; the operator still decides grounded / in air.
      const near = (props.get(frame) ?? []).map((q) => ({ q, d: Math.hypot(q.point[0] - x, q.point[1] - y) })).sort((a, b) => a.d - b.d)[0];
      const pt: Pt = near && near.d < Math.max(12, W / 120) ? near.q.point : p;
      setObs((o) => [...o.filter((q) => q.frame_index !== frame), { frame_index: frame, timestamp_ms: ts, point: pt, grounded: tool === "grounded", source: "manual" as const }]
        .sort((a, b) => a.frame_index - b.frame_index));
    }
  };

  const assessments = useMemo(() => {
    const m = new Map<number, string>();
    const a = (result?.result.crease as { assessments?: { frame_index: number; state: string }[] } | undefined)?.assessments;
    a?.forEach((x) => m.set(x.frame_index, x.state));
    return m;
  }, [result]);

  const roiBox = roi.length === 2 ? [Math.min(roi[0][0], roi[1][0]), Math.min(roi[0][1], roi[1][1]), Math.max(roi[0][0], roi[1][0]), Math.max(roi[0][1], roi[1][1])].map(Math.round) as [number, number, number, number] : null;
  const creaseLine = (result?.result.crease_line_px as Pt[] | undefined) ?? (crease.length === 2 ? crease : null);
  const r = Math.max(4, W / 250);

  const overlay = (
    <>
      {(props.get(frame) ?? []).map((q, i) => (
        <g key={`p${i}`}>
          <rect x={q.point[0] - r} y={q.point[1] - r} width={2 * r} height={2 * r} transform={`rotate(45 ${q.point[0]} ${q.point[1]})`}
            fill="none" stroke="#e879f9" strokeWidth={r / 4} strokeDasharray={q.grounded_hint ? undefined : `${r / 2} ${r / 3}`} />
        </g>
      ))}
      {creaseLine && (() => { const [a, b] = extend(creaseLine[0], creaseLine[1], W, H); return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#38bdf8" strokeWidth={r / 2} />; })()}
      {crease.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={r} fill="none" stroke="#38bdf8" strokeWidth={r / 3} />)}
      {behind && <g><circle cx={behind[0]} cy={behind[1]} r={r} fill="#4ade80" /><text x={behind[0] + r * 1.5} y={behind[1]} fill="#4ade80" fontSize={r * 3}>behind</text></g>}
      {roiBox && <rect x={roiBox[0]} y={roiBox[1]} width={roiBox[2] - roiBox[0]} height={roiBox[3] - roiBox[1]} fill="none" stroke="#e879f9" strokeDasharray={`${r} ${r}`} strokeWidth={r / 3} />}
      {roi.length === 1 && <circle cx={roi[0][0]} cy={roi[0][1]} r={r} fill="#e879f9" />}
      {obs.filter((o) => o.point).map((o) => {
        const cur = o.frame_index === frame;
        const col = STATE_COLOR[assessments.get(o.frame_index) ?? ""] ?? (o.grounded ? "#4ade80" : "#fbbf24");
        return <circle key={o.frame_index} cx={o.point![0]} cy={o.point![1]} r={cur ? r * 1.4 : r * 0.8} fill={cur ? col : "none"} stroke={col} strokeWidth={r / 3} opacity={cur ? 1 : 0.6} />;
      })}
    </>
  );

  const detect = async () => {
    if (!roiBox) return;
    setBusy("detect");
    setErr(null);
    try {
      // With camera-motion compensation the box is drawn on the scan's first frame, so scan from the current frame.
      const start = followCam ? frame : Math.max(0, frame - 50);
      const a = await api.detectWicketBreak({ video_id: video.id, roi: roiBox, start_frame: start, end_frame: start + 300, compensate_motion: followCam });
      const d = a.result as { status: string; last_intact_frame?: number; last_intact_ms?: number; first_broken_frame?: number; first_broken_ms?: number; reasons?: string[] };
      if (d.status === "BREAK_CANDIDATE" && d.first_broken_frame != null && d.last_intact_frame != null) {
        setIntact({ frame: d.last_intact_frame, ts: d.last_intact_ms! });
        setBroken({ frame: d.first_broken_frame, ts: d.first_broken_ms! });
        setBreakSource("auto");
        setConfirmed(false);
        setFrame(d.first_broken_frame);
      } else setErr(`Wicket break: ${d.status}. ${(d.reasons ?? []).join(" ")}`);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  };

  const propose = async () => {
    setBusy("propose");
    setErr(null);
    try {
      const body: Record<string, unknown> = { video_id: video.id, start_frame: Math.max(0, frame - 25), end_frame: frame + 25 };
      if (crease.length === 2) body.crease = { p1: crease[0], p2: crease[1] };
      if (roiBox) body.roi = roiBox;
      const d = await api.creaseProposals(body);
      setPropInfo(d);
      setProps(new Map(d.frames.map((f) => [f.frame_index, f.proposals])));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  };

  const analyze = async () => {
    setBusy("analyze");
    setErr(null);
    try {
      const creaseSpec = calId
        ? { source: "calibration", calibration_id: calId, end }
        : { source: "manual", p1: crease[0], p2: crease[1], behind_point: behind, end };
      const wb = intact && broken ? { last_intact_frame: intact.frame, last_intact_ms: intact.ts, first_broken_frame: broken.frame, first_broken_ms: broken.ts, source: breakSource, confirmed } : null;
      const f = Object.fromEntries(Object.entries(facts).filter(([, v]) => v !== null));
      const a = await api.runoutAnalyze({ video_id: video.id, dismissal: mode, crease: creaseSpec, observations: obs, wicket_break: wb, facts: f });
      setResult(a);
      onAnalysis(a);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  };

  const creaseReady = calId ? true : crease.length === 2 && behind !== null;
  const res = result?.result as undefined | {
    status: string; rules_summary: string; reasons: string[]; warnings: string[];
    rule_checks: { id: string; description: string; law_reference: string; status: string; evidence: string }[];
    timeline: { t_start_ms: number; t_end_ms: number; kind: string; description: string }[];
  };

  return (
    <Panel title={mode === "run_out" ? "Run-out review (Laws 29, 30, 38)" : "Stumping review (Laws 29, 30, 39)"}>
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <div className="space-y-2">
          {!readOnly && (
            <div className="flex flex-wrap gap-1 text-xs">
              {(["crease", "behind", "grounded", "airborne", "roi"] as Tool[]).map((t) => (
                <button key={t} onClick={() => setTool(t)} className={`rounded border px-2 py-1 ${tool === t ? "border-accent bg-accent/15 text-accent" : "border-line bg-panel-2 text-muted"}`}>
                  {{ crease: "Crease line", behind: "Behind point", grounded: "Bat/foot grounded", airborne: "Bat/foot in air", roi: "Stumps box", none: "" }[t]}
                </button>
              ))}
              <span className="self-center text-muted">{TOOL_HELP[tool]}</span>
            </div>
          )}
          <FrameViewer videoId={video.id} width={W} height={H} frameCount={video.meta.frame_count_decoded ?? video.meta.frame_count_declared}
            frame={frame} setFrame={setFrame} onFrameInfo={setInfo} onImageClick={click} overlay={overlay} crosshair />
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted">Wicket:</span>
            <Button disabled={readOnly || ts === null} onClick={() => { setIntact({ frame, ts: ts! }); setBreakSource("manual"); }}>Last intact = this frame</Button>
            <Button disabled={readOnly || ts === null} onClick={() => { setBroken({ frame, ts: ts! }); setBreakSource("manual"); }}>First broken = this frame</Button>
            <Button disabled={readOnly || !roiBox || busy !== null} onClick={detect}>{busy === "detect" ? "Detecting..." : "Auto-detect break"}</Button>
            <label className="flex items-center gap-1 text-xs" title="Camera pans or zooms: draw the stumps box on the current frame (intact wicket) and the scan starts here">
              <input type="checkbox" checked={followCam} onChange={(e) => setFollowCam(e.target.checked)} /> camera moves
            </label>
            <Button disabled={readOnly || (crease.length < 2 && !roiBox) || busy !== null} onClick={propose} title="Pose model proposes ankle/foot points for ±25 frames; click a proposal to use it">
              {busy === "propose" ? "Proposing..." : "Propose feet (model)"}
            </Button>
            {intact && broken && (
              <span className="tabular text-xs">
                break between f{intact.frame} and f{broken.frame} ({(broken.ts - intact.ts).toFixed(1)} ms window, {breakSource})
              </span>
            )}
          </div>
          {propInfo && (
            <div className="text-xs text-muted">
              Model proposals ({propInfo.model_id}, {propInfo.frames_with_batter} frames with a batter): magenta diamonds, solid = likely grounded, dashed = unsure or moving.
              They are suggestions; select the grounded / in-air tool and click one to use it. {propInfo.warnings.slice(0, 2).join(" ")}
            </div>
          )}
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" disabled={readOnly} checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
            I confirm a bail was completely removed / a stump struck out in the first-broken frame (Law 29.1)
          </label>
        </div>

        <div className="space-y-3 text-sm">
          <div className="space-y-1.5 rounded border border-line p-2">
            <div className="text-xs uppercase tracking-wider text-muted">Crease geometry</div>
            <select className="w-full rounded border border-line bg-panel-2 px-2 py-1" value={calId} onChange={(e) => setCalId(e.target.value)}>
              <option value="">Manual: clicked crease line</option>
              {cals.map((c) => <option key={c.id} value={c.id}>Calibration {c.id.slice(0, 6)} ({c.camera.calibration_type})</option>)}
            </select>
            <select className="w-full rounded border border-line bg-panel-2 px-2 py-1" value={end} onChange={(e) => setEnd(e.target.value as "striker" | "bowler")}>
              <option value="striker">Striker&apos;s end</option><option value="bowler">Bowler&apos;s end</option>
            </select>
            {!calId && <div className="text-xs text-muted">{crease.length}/2 crease points · behind point {behind ? "set" : "missing"}</div>}
          </div>

          <div className="rounded border border-line p-2">
            <div className="mb-1 text-xs uppercase tracking-wider text-muted">Batter observations ({obs.length})</div>
            <div className="max-h-40 space-y-0.5 overflow-auto text-xs">
              {obs.map((o) => (
                <div key={o.frame_index} className="flex items-center justify-between gap-2">
                  <button className="tabular text-accent hover:underline" onClick={() => setFrame(o.frame_index)}>f{o.frame_index} · {o.timestamp_ms.toFixed(1)} ms</button>
                  <span style={{ color: STATE_COLOR[assessments.get(o.frame_index) ?? ""] }}>{assessments.get(o.frame_index) ?? (o.grounded ? "grounded" : "in air")}</span>
                  {!readOnly && <button className="text-muted hover:text-bad" onClick={() => setObs(obs.filter((q) => q !== o))}>×</button>}
                </div>
              ))}
              {obs.length === 0 && <div className="text-muted">Mark the bat/foot in the frames around the break.</div>}
            </div>
          </div>

          <div className="space-y-1 rounded border border-line p-2">
            <div className="text-xs uppercase tracking-wider text-muted">Facts (unknown stays unknown)</div>
            <label className="flex items-center justify-between gap-2 text-xs">
              <span className="text-muted">Batter</span>
              <select className="rounded border border-line bg-panel-2 px-1.5 py-0.5" value={(facts.batter as string) ?? ""} onChange={(e) => setFacts({ ...facts, batter: e.target.value || null })}>
                <option value="">unknown</option><option value="striker">striker</option><option value="non_striker">non-striker</option>
              </select>
            </label>
            {([["no_ball", "No ball"], ["wicket_broken_fairly", "Wicket broken fairly (ball in hand / ball)"], ["ball_touched_fielder_before_break", "Ball touched a fielder first"],
              ["running_or_diving_into_ground", "Running/diving into ground (30.1.2)"], ["left_ground_to_avoid_injury", "Left ground to avoid injury"],
              ...(mode === "stumping" ? [["attempting_run", "Striker attempting a run"], ["broken_by_wicket_keeper_without_intervention", "Put down by keeper, no other fielder"]] : [])] as [string, string][]).map(([k, l]) => (
              <TriSelect key={k} label={l} value={facts[k] as Tri} onChange={(v) => setFacts({ ...facts, [k]: v })} />
            ))}
          </div>

          {!readOnly && <Button variant="primary" disabled={!creaseReady || obs.length === 0 || busy !== null} onClick={analyze}>{busy === "analyze" ? "Analysing..." : "Analyse and attach"}</Button>}
          <ErrorBox error={err} />
        </div>
      </div>

      {res && (
        <div className="mt-4 grid gap-4 border-t border-line pt-4 text-sm lg:grid-cols-2">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={statusToneOf(res.status)}>{res.status}</Badge>
              <Badge tone={statusToneOf(res.rules_summary)}>{res.rules_summary}</Badge>
            </div>
            <ul className="list-disc pl-5 text-muted">{res.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul>
            {res.warnings.length > 0 && <ul className="list-disc pl-5 text-warn">{res.warnings.map((x, i) => <li key={i}>{x}</li>)}</ul>}
            <div className="text-xs uppercase tracking-wider text-muted">Timeline</div>
            <ol className="space-y-0.5 text-xs">
              {res.timeline.map((t, i) => (
                <li key={i} className="tabular"><span className="text-muted">{t.t_start_ms.toFixed(1)}{t.t_end_ms !== t.t_start_ms && `–${t.t_end_ms.toFixed(1)}`} ms</span> {t.description}</li>
              ))}
            </ol>
          </div>
          <table className="w-full self-start text-xs">
            <tbody>
              {res.rule_checks.map((c) => (
                <tr key={c.id} className="border-t border-line align-top">
                  <td className="py-1 pr-2">{c.description}<div className="text-muted">{c.law_reference}</div></td>
                  <td className="pr-2"><Badge tone={c.status === "PASS" ? "good" : c.status === "FAIL" ? "bad" : "warn"}>{c.status}</Badge></td>
                  <td className="text-muted">{c.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
