"use client";

import { useState } from "react";

import { statusToneOf } from "@/components/CreaseWorkspace";
import FrameViewer, { type FrameInfo } from "@/components/FrameViewer";
import { Badge, Button, ErrorBox, Panel } from "@/components/ui";
import { api, type DecisionAnalysis, type Video } from "@/lib/api";

type Pt = [number, number];
type Tri = boolean | null;
type Kind = "ball_ground" | "fielder_ball" | "fielder_ground" | "fielder_rope" | "ball_crossed_rope" | "catch_completed" | "ball_dead";
interface Ev { kind: Kind; timestamp_ms: number; frame_index: number; source: "manual"; [k: string]: unknown }

const KINDS: { kind: Kind; label: string; needsPoint: boolean; help: string }[] = [
  { kind: "ball_ground", label: "Ball grounds", needsPoint: true, help: "click where the ball touches the ground" },
  { kind: "fielder_ball", label: "Fielder touches ball", needsPoint: true, help: "click the fielder's ground contact (or, if airborne, their last ground contact)" },
  { kind: "fielder_ground", label: "Fielder grounds", needsPoint: true, help: "click where the fielder's foot/body touches the ground" },
  { kind: "fielder_rope", label: "Fielder touches rope", needsPoint: false, help: "" },
  { kind: "ball_crossed_rope", label: "Ball crosses rope", needsPoint: false, help: "" },
  { kind: "catch_completed", label: "Catch completed", needsPoint: false, help: "" },
  { kind: "ball_dead", label: "Ball dead", needsPoint: false, help: "" },
];
const triOpts = (v: Tri, set: (v: Tri) => void, label: string) => (
  <label className="flex items-center gap-1 text-xs">
    <span className="text-muted">{label}</span>
    <select className="rounded border border-line bg-panel-2 px-1 py-0.5" value={v === null ? "" : String(v)} onChange={(e) => set(e.target.value === "" ? null : e.target.value === "true")}>
      <option value="">unknown</option><option value="true">yes</option><option value="false">no</option>
    </select>
  </label>
);

export default function BoundaryWorkspace({ video, readOnly, onAnalysis }: { video: Video; readOnly: boolean; onAnalysis: (a: DecisionAnalysis) => void }) {
  const W = video.meta.width, H = video.meta.height;
  const [frame, setFrame] = useState(0);
  const [info, setInfo] = useState<FrameInfo | null>(null);
  const [tool, setTool] = useState<"rope" | "field" | "event">("rope");
  const [rope, setRope] = useState<Pt[]>([]);
  const [field, setField] = useState<Pt | null>(null);
  const [kind, setKind] = useState<Kind>("ball_ground");
  const [fielder, setFielder] = useState("F1");
  const [airborne, setAirborne] = useState<Tri>(false);
  const [contact, setContact] = useState<Tri>(null);
  const [inAir, setInAir] = useState<Tri>(null);
  const [events, setEvents] = useState<Ev[]>([]);
  const [facts, setFacts] = useState<Record<string, Tri>>({ struck_by_bat: null, no_ball: null, ball_touched_bat: null });
  const [analysis, setAnalysis] = useState<"boundary" | "catch">("boundary");
  const [result, setResult] = useState<DecisionAnalysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const ts = info?.index === frame ? info.timestampMs : null;
  const spec = KINDS.find((k) => k.kind === kind)!;

  const addEvent = (p: Pt | null) => {
    if (ts === null) return;
    const base: Ev = { kind, timestamp_ms: ts, frame_index: frame, source: "manual" };
    const gp = p ? { point: p, frame: "image" } : null;
    if (kind === "ball_ground") Object.assign(base, { position: gp });
    if (kind === "fielder_ball") Object.assign(base, airborne ? { fielder_id: fielder, airborne: true, last_ground_contact: gp, touching_rope: contact } : { fielder_id: fielder, airborne, ground_contacts: gp ? [gp] : [], touching_rope: contact });
    if (kind === "fielder_ground") Object.assign(base, { fielder_id: fielder, position: gp, in_contact_with_ball: contact });
    if (kind === "fielder_rope") Object.assign(base, { fielder_id: fielder, in_contact_with_ball: contact });
    if (kind === "ball_crossed_rope") Object.assign(base, { in_air: inAir });
    if (kind === "catch_completed") Object.assign(base, { fielder_id: fielder });
    setEvents((e) => [...e, base].sort((a, b) => a.timestamp_ms - b.timestamp_ms));
  };

  const click = (x: number, y: number) => {
    if (readOnly) return;
    if (tool === "rope") setRope((r) => [...r, [x, y]]);
    else if (tool === "field") setField([x, y]);
    else if (spec.needsPoint) addEvent([x, y]);
  };

  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      const a = await api.boundaryAnalyze({
        video_id: video.id, analysis, rope: { frame: "image", points: rope, field_side_point: field }, events,
        facts: Object.fromEntries(Object.entries(facts).filter(([, v]) => v !== null)),
      });
      setResult(a);
      onAnalysis(a);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };

  const r = Math.max(4, W / 250);
  const evPt = (e: Ev): Pt | null => {
    const g = (e.position ?? e.last_ground_contact ?? (e.ground_contacts as { point: Pt }[] | undefined)?.[0]) as { point: Pt } | null | undefined;
    return g?.point ?? null;
  };
  const overlay = (
    <>
      {rope.length > 1 && <polyline points={rope.map((p) => p.join(",")).join(" ")} fill="none" stroke="#38bdf8" strokeWidth={r / 2} />}
      {rope.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={r * 0.7} fill="#38bdf8" />)}
      {field && <g><circle cx={field[0]} cy={field[1]} r={r} fill="#4ade80" /><text x={field[0] + r * 1.5} y={field[1]} fill="#4ade80" fontSize={r * 3}>field</text></g>}
      {events.map((e, i) => { const p = evPt(e); return p && <circle key={i} cx={p[0]} cy={p[1]} r={e.frame_index === frame ? r * 1.4 : r * 0.8} fill="none" stroke={e.kind === "ball_ground" ? "#e879f9" : "#fbbf24"} strokeWidth={r / 3} />; })}
    </>
  );

  const res = result?.result as undefined | {
    status: string; reasons: string[]; warnings: string[]; runs_allowance?: number | null;
    rule_checks?: { id: string; description: string; law_reference: string; status: string; evidence: string }[];
    timeline?: { index: number; timestamp_ms: number; description: string; decisive?: boolean }[];
  };

  return (
    <Panel title="Boundary / boundary-catch review (Laws 19, 33)">
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <div className="space-y-2">
          {!readOnly && (
            <div className="flex flex-wrap items-center gap-1 text-xs">
              {(["rope", "field", "event"] as const).map((t) => (
                <button key={t} onClick={() => setTool(t)} className={`rounded border px-2 py-1 ${tool === t ? "border-accent bg-accent/15 text-accent" : "border-line bg-panel-2 text-muted"}`}>
                  {{ rope: "Trace rope (inner edge)", field: "Field-side point", event: "Mark events" }[t]}
                </button>
              ))}
              {tool === "rope" && <Button variant="ghost" onClick={() => setRope([])}>Clear rope</Button>}
              <span className="text-muted">
                {tool === "rope" ? "Click along the field-side edge of the rope, in order" : tool === "field" ? "Click any point inside the field of play" : spec.help}
              </span>
            </div>
          )}
          <FrameViewer videoId={video.id} width={W} height={H} frameCount={video.meta.frame_count_decoded ?? video.meta.frame_count_declared}
            frame={frame} setFrame={setFrame} onFrameInfo={setInfo} onImageClick={click} overlay={overlay} crosshair />
          <div className="text-xs text-warn">
            Image-plane rope geometry is only valid for points on the ground. A ball in the air appears to cross the rope when it may not have.
          </div>
        </div>

        <div className="space-y-3 text-sm">
          {!readOnly && (
            <div className="space-y-1.5 rounded border border-line p-2">
              <div className="text-xs uppercase tracking-wider text-muted">Event at frame {frame}</div>
              <select className="w-full rounded border border-line bg-panel-2 px-2 py-1" value={kind} onChange={(e) => { setKind(e.target.value as Kind); setTool("event"); }}>
                {KINDS.map((k) => <option key={k.kind} value={k.kind}>{k.label}</option>)}
              </select>
              {["fielder_ball", "fielder_ground", "fielder_rope", "catch_completed"].includes(kind) && (
                <label className="flex items-center gap-1 text-xs"><span className="text-muted">Fielder</span>
                  <input className="w-20 rounded border border-line bg-panel-2 px-1 py-0.5" value={fielder} onChange={(e) => setFielder(e.target.value)} /></label>
              )}
              {kind === "fielder_ball" && triOpts(airborne, setAirborne, "Airborne at touch")}
              {kind === "fielder_ball" && triOpts(contact, setContact, "Touching rope")}
              {(kind === "fielder_ground" || kind === "fielder_rope") && triOpts(contact, setContact, "In contact with ball")}
              {kind === "ball_crossed_rope" && triOpts(inAir, setInAir, "Ball in the air")}
              {spec.needsPoint ? <div className="text-xs text-muted">Click on the frame to add.</div> : <Button disabled={ts === null} onClick={() => addEvent(null)}>Add event</Button>}
            </div>
          )}
          <div className="rounded border border-line p-2">
            <div className="mb-1 text-xs uppercase tracking-wider text-muted">Sequence ({events.length})</div>
            <ol className="max-h-44 space-y-0.5 overflow-auto text-xs">
              {events.map((e, i) => (
                <li key={i} className="flex items-center justify-between gap-2">
                  <button className="tabular text-accent hover:underline" onClick={() => setFrame(e.frame_index)}>{e.timestamp_ms.toFixed(0)} ms</button>
                  <span className="flex-1">{KINDS.find((k) => k.kind === e.kind)?.label}{e.fielder_id ? ` (${String(e.fielder_id)})` : ""}{e.airborne ? ", airborne" : ""}</span>
                  {!readOnly && <button className="text-muted hover:text-bad" onClick={() => setEvents(events.filter((_, j) => j !== i))}>×</button>}
                </li>
              ))}
            </ol>
          </div>
          <div className="space-y-1 rounded border border-line p-2">
            {triOpts(facts.struck_by_bat, (v) => setFacts({ ...facts, struck_by_bat: v }), "Ball struck by bat (six possible)")}
            {triOpts(facts.ball_touched_bat, (v) => setFacts({ ...facts, ball_touched_bat: v }), "Ball touched bat (catch)")}
            {triOpts(facts.no_ball, (v) => setFacts({ ...facts, no_ball: v }), "No ball")}
            <select className="mt-1 w-full rounded border border-line bg-panel-2 px-2 py-1" value={analysis} onChange={(e) => setAnalysis(e.target.value as "boundary" | "catch")}>
              <option value="boundary">Was it a boundary? (Law 19)</option>
              <option value="catch">Was the catch fair near the boundary? (Laws 19.5, 33)</option>
            </select>
          </div>
          {!readOnly && <Button variant="primary" disabled={busy || rope.length < 2 || !field || events.length === 0} onClick={run}>{busy ? "Analysing..." : "Analyse and attach"}</Button>}
          <ErrorBox error={err} />
        </div>
      </div>
      {res && (
        <div className="mt-4 grid gap-4 border-t border-line pt-4 text-sm lg:grid-cols-2">
          <div className="space-y-2">
            <Badge tone={statusToneOf(res.status)}>{res.status}</Badge>
            <ul className="list-disc pl-5 text-muted">{res.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul>
            {res.warnings.length > 0 && <ul className="list-disc pl-5 text-warn">{res.warnings.map((x, i) => <li key={i}>{x}</li>)}</ul>}
            {res.timeline && (
              <ol className="space-y-0.5 text-xs">
                {res.timeline.map((t) => <li key={t.index} className={t.decisive ? "text-ink" : "text-muted"}><span className="tabular">{t.timestamp_ms.toFixed(0)} ms</span> {t.description}</li>)}
              </ol>
            )}
          </div>
          {res.rule_checks && (
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
          )}
        </div>
      )}
    </Panel>
  );
}
