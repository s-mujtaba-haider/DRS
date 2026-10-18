"use client";

import { useState } from "react";

import { statusToneOf } from "@/components/CreaseWorkspace";
import FrameViewer, { type FrameInfo } from "@/components/FrameViewer";
import { Badge, Button, ErrorBox, Panel } from "@/components/ui";
import { api, type DecisionAnalysis, type Video } from "@/lib/api";

type Pt = [number, number];
type Tool = "crease" | "behind" | "front" | "ret" | "inside" | "back";
interface Mark { frame_index: number; timestamp_ms: number; point: Pt; source: "manual" }

const HELP: Record<Tool, string> = {
  crease: "Two clicks on the BACK edge of the bowler's popping crease marking",
  behind: "One click on the bowler's stumps side of that crease",
  front: "Front foot at landing: click the ground under its rearmost part (a raised heel counts)",
  ret: "Two clicks on the inner edge of the return crease (optional, back-foot check)",
  inside: "One click between the return crease and the middle stump",
  back: "Back foot at landing: click its point nearest the return crease",
};
const LABEL: Record<Tool, string> = { crease: "Popping crease", behind: "Stumps side", front: "Front foot", ret: "Return crease", inside: "Inside point", back: "Back foot" };

function extend(a: Pt, b: Pt, w: number, h: number): [Pt, Pt] {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const k = (w + h) / (Math.hypot(dx, dy) || 1);
  return [[a[0] - dx * k, a[1] - dy * k], [b[0] + dx * k, b[1] + dy * k]];
}

export default function NoBallWorkspace({ video, readOnly, onAnalysis }: { video: Video; readOnly: boolean; onAnalysis: (a: DecisionAnalysis) => void }) {
  const W = video.meta.width, H = video.meta.height;
  const [frame, setFrame] = useState(0);
  const [info, setInfo] = useState<FrameInfo | null>(null);
  const [tool, setTool] = useState<Tool>("crease");
  const [crease, setCrease] = useState<Pt[]>([]);
  const [behind, setBehind] = useState<Pt | null>(null);
  const [ret, setRet] = useState<Pt[]>([]);
  const [inside, setInside] = useState<Pt | null>(null);
  const [front, setFront] = useState<Mark | null>(null);
  const [back, setBack] = useState<Mark | null>(null);
  const [firstContact, setFirstContact] = useState(false);
  const [side, setSide] = useState<"" | "true" | "false">("");
  const [res, setRes] = useState<DecisionAnalysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const ts = info?.index === frame ? info.timestampMs : null;

  const click = (x: number, y: number) => {
    if (readOnly) return;
    const p: Pt = [x, y];
    if (tool === "crease") setCrease((c) => (c.length >= 2 ? [p] : [...c, p]));
    else if (tool === "behind") setBehind(p);
    else if (tool === "ret") setRet((c) => (c.length >= 2 ? [p] : [...c, p]));
    else if (tool === "inside") setInside(p);
    else if (ts !== null) (tool === "front" ? setFront : setBack)({ frame_index: frame, timestamp_ms: ts, point: p, source: "manual" });
  };

  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      const facts: Record<string, boolean> = { landing_frame_is_first_contact: firstContact };
      if (side) facts.front_foot_correct_side_of_middle_stump_line = side === "true";
      const a = await api.noBallAnalyze({
        video_id: video.id,
        popping_crease: { source: "manual", p1: crease[0], p2: crease[1], behind_point: behind, end: "bowler" },
        front_marks: front ? [front] : [],
        return_crease: ret.length === 2 && inside ? { p1: ret[0], p2: ret[1], inside_point: inside } : null,
        back_marks: back && ret.length === 2 && inside ? [back] : [],
        facts,
      });
      setRes(a);
      onAnalysis(a);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  const r = Math.max(4, W / 250);
  const line = (pts: Pt[], color: string) => {
    if (pts.length < 2) return null;
    const [a, b] = extend(pts[0], pts[1], W, H);
    return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={color} strokeWidth={r / 2} />;
  };
  const dot = (p: Pt | null | undefined, color: string, on = true) => p && <circle cx={p[0]} cy={p[1]} r={on ? r * 1.2 : r * 0.7} fill="none" stroke={color} strokeWidth={r / 3} />;
  const result = res?.result as undefined | { status: string; reasons: string[]; warnings: string[]; rule_checks: { id: string; description: string; law_reference: string; status: string; evidence: string }[] };

  return (
    <Panel title="No-ball check (Law 21.5)">
      <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
        <div className="space-y-2">
          {!readOnly && (
            <div className="flex flex-wrap items-center gap-1 text-xs">
              {(Object.keys(LABEL) as Tool[]).map((t) => (
                <button key={t} onClick={() => setTool(t)} className={`rounded border px-2 py-1 ${tool === t ? "border-accent bg-accent/15 text-accent" : "border-line bg-panel-2 text-muted"}`}>{LABEL[t]}</button>
              ))}
              <span className="text-muted">{HELP[tool]}</span>
            </div>
          )}
          <FrameViewer videoId={video.id} width={W} height={H} frameCount={video.meta.frame_count_decoded ?? video.meta.frame_count_declared}
            frame={frame} setFrame={setFrame} onFrameInfo={setInfo} onImageClick={click} crosshair
            overlay={<>
              {line(crease, "#38bdf8")}
              {line(ret, "#22d3ee")}
              {crease.map((p, i) => <g key={i}>{dot(p, "#38bdf8", false)}</g>)}
              {ret.map((p, i) => <g key={`r${i}`}>{dot(p, "#22d3ee", false)}</g>)}
              {dot(behind, "#4ade80")}
              {dot(inside, "#4ade80")}
              {dot(front?.point, "#e879f9", front?.frame_index === frame)}
              {dot(back?.point, "#fbbf24", back?.frame_index === frame)}
            </>} />
        </div>
        <div className="space-y-2 text-sm">
          <div className="rounded border border-line p-2 text-xs">
            <div>Popping crease {crease.length}/2 · stumps side {behind ? "set" : "missing"}</div>
            <div>Front foot {front ? <button className="text-accent hover:underline" onClick={() => setFrame(front.frame_index)}>frame {front.frame_index}</button> : "not marked"}</div>
            <div className="text-muted">Return crease {ret.length}/2 · inside {inside ? "set" : "-"} · back foot {back ? `frame ${back.frame_index}` : "-"}</div>
          </div>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={firstContact} onChange={(e) => setFirstContact(e.target.checked)} /> Marked frame is the first frame the front foot is down</label>
          <label className="flex items-center justify-between gap-2 text-xs">
            <span className="text-muted">Front foot on the correct side of the middle-stump line</span>
            <select className="rounded border border-line bg-panel-2 px-1 py-0.5" value={side} onChange={(e) => setSide(e.target.value as "" | "true" | "false")}>
              <option value="">unknown</option><option value="true">yes</option><option value="false">no</option>
            </select>
          </label>
          {!readOnly && <Button variant="primary" disabled={busy || crease.length < 2 || !behind || !front} onClick={run}>{busy ? "Checking..." : "Check and attach"}</Button>}
          <ErrorBox error={err} />
          {result && (
            <div className="space-y-1">
              <Badge tone={result.status === "NO_BALL" ? "bad" : statusToneOf(result.status)}>{result.status.replace("_", " ")}</Badge>
              <ul className="list-disc pl-5 text-xs text-muted">{result.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul>
              {result.warnings.length > 0 && <ul className="list-disc pl-5 text-xs text-warn">{result.warnings.map((x, i) => <li key={i}>{x}</li>)}</ul>}
              {result.rule_checks.map((c) => (
                <div key={c.id} className="flex items-start gap-2 text-xs">
                  <Badge tone={c.status === "PASS" ? "good" : c.status === "FAIL" ? "bad" : "warn"}>{c.status}</Badge>
                  <span><span className="text-ink">{c.law_reference}</span> <span className="text-muted">{c.evidence}</span></span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}
