"use client";

import { useEffect, useState } from "react";

import { Badge, Button, ErrorBox, Panel } from "@/components/ui";
import { api, type Calibration, type Job, type LbwAnalysis, type Review, type Video } from "@/lib/api";

/** Multi-camera LBW: track every camera of a clip group, pick each camera's calibration, analyse, attach. */
export default function MultiviewLbwPanel({ review, videos, onAttach, readOnly }: {
  review: Review; videos: Record<string, Video>; onAttach: (id: string) => void; readOnly: boolean;
}) {
  const vids = review.video_ids.map((v) => videos[v]).filter(Boolean);
  const group = vids.find((v) => v.meta.capture)?.meta.capture?.clip_group ?? null;
  const [jobs, setJobs] = useState<Record<string, Job | undefined>>({});
  const [cals, setCals] = useState<Record<string, Calibration[]>>({});
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const [use, setUse] = useState<Record<string, boolean>>({});
  const [handed, setHanded] = useState<"" | "right" | "left">("");
  const [impact, setImpact] = useState("");
  const [hw, setHw] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [result, setResult] = useState<LbwAnalysis | null>(null);

  const key = review.video_ids.join(",");
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let alive = true;
    Promise.all(
      key.split(",").filter(Boolean).map(async (id) => {
        const js = await api.jobs(id);
        return [id, js.find((j) => j.status === "succeeded") ?? js[0], await api.calibrations(id)] as const;
      }),
    )
      .then((rows) => {
        if (!alive) return;
        setJobs(Object.fromEntries(rows.map(([v, j]) => [v, j])));
        setCals(Object.fromEntries(rows.map(([v, , c]) => [v, c])));
        setChosen((old) => ({ ...Object.fromEntries(rows.filter(([, , c]) => c[0]).map(([v, , c]) => [v, c[0].id])), ...old }));
        setUse((old) => ({ ...Object.fromEntries(rows.map(([v]) => [v, true])), ...old }));
      })
      .catch((e) => alive && setErr(e));
    return () => {
      alive = false;
    };
  }, [key, reload]);
  const load = () => setReload((n) => n + 1);

  if (vids.length < 2) return null;

  const trackAll = async () => {
    setBusy("track");
    setErr(null);
    try {
      if (group) await api.trackClipGroup(group);
      else for (const v of vids) await api.createJob({ video_id: v.id, detector: "auto" });
      load();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  };

  const run = async () => {
    setBusy("run");
    setErr(null);
    try {
      const cams = vids.filter((v) => use[v.id]).map((v) => ({ job_id: jobs[v.id]!.id, calibration_id: chosen[v.id] }));
      const a = await api.lbwMultiview({
        cameras: cams,
        impact_time_ms: impact ? Number(impact) : null,
        facts: handed ? { batter_handedness: handed } : {},
        hardware_sync: hw,
      });
      setResult(a);
      onAttach(a.id);
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  };

  const ready = vids.filter((v) => use[v.id] && jobs[v.id]?.status === "succeeded" && chosen[v.id]).length;
  const r = result?.result;
  const mv = (r as unknown as { multiview?: { sync_position_error_m?: number; definite_allowed?: boolean; ball_speed_mps?: number } } | undefined)?.multiview;

  return (
    <Panel title="Multi-camera LBW (triangulated)" actions={
      <div className="flex gap-2">
        <Button variant="ghost" onClick={load}>Refresh</Button>
        {!readOnly && <Button onClick={trackAll} disabled={busy !== null}>{busy === "track" ? "Queueing..." : "Track all cameras"}</Button>}
      </div>
    }>
      <ErrorBox error={err} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead className="text-left text-xs uppercase tracking-wider text-muted">
            <tr><th className="pb-1">Use</th><th>Camera</th><th>Tracking</th><th>Calibration</th><th>Sync</th></tr>
          </thead>
          <tbody>
            {vids.map((v) => {
              const j = jobs[v.id];
              return (
                <tr key={v.id} className="border-t border-line">
                  <td className="py-1.5"><input type="checkbox" checked={!!use[v.id]} onChange={(e) => setUse({ ...use, [v.id]: e.target.checked })} /></td>
                  <td>{v.meta.capture?.camera_id ?? v.original_filename} <span className="text-xs text-muted">{v.meta.nominal_fps?.toFixed(0)} fps</span></td>
                  <td>{j ? <Badge tone={j.status === "succeeded" ? "good" : j.status === "failed" ? "bad" : "info"}>{j.status}</Badge> : <span className="text-muted">none</span>}</td>
                  <td>
                    <select className="rounded border border-line bg-panel-2 px-1.5 py-0.5 text-xs" value={chosen[v.id] ?? ""} onChange={(e) => setChosen({ ...chosen, [v.id]: e.target.value })}>
                      <option value="">none (calibrate in the session)</option>
                      {(cals[v.id] ?? []).map((c) => <option key={c.id} value={c.id}>{c.id.slice(0, 6)} · {c.camera.calibration_type} · {c.camera.reprojection_rms_px?.toFixed(2)} px</option>)}
                    </select>
                  </td>
                  <td className="text-xs text-muted">{v.meta.capture ? `±${v.meta.capture.sync_uncertainty_ms?.toFixed(1)} ms` : "unknown (uploaded)"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-1">Batter
          <select className="rounded border border-line bg-panel-2 px-1.5 py-0.5" value={handed} onChange={(e) => setHanded(e.target.value as "" | "right" | "left")}>
            <option value="">unknown</option><option value="right">right-handed</option><option value="left">left-handed</option>
          </select>
        </label>
        <label className="flex items-center gap-1">Impact time (shared clock, ms)
          <input className="w-24 rounded border border-line bg-panel-2 px-1.5 py-0.5" placeholder="auto" value={impact} onChange={(e) => setImpact(e.target.value)} />
        </label>
        <label className="flex items-center gap-1" title="Only tick when the cameras are triggered or PTP-locked; otherwise definite verdicts are withheld">
          <input type="checkbox" checked={hw} onChange={(e) => setHw(e.target.checked)} /> cameras hardware-synchronised
        </label>
        {!readOnly && <Button variant="primary" disabled={ready < 2 || busy !== null} onClick={run}>{busy === "run" ? "Reconstructing..." : `Analyse ${ready} cameras and attach`}</Button>}
      </div>
      {r && (
        <div className="mt-3 space-y-1 border-t border-line pt-3 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={r.verdict === "INCONCLUSIVE" || r.verdict === "INSUFFICIENT_DATA" ? "warn" : "info"}>{r.verdict}</Badge>
            <span>P(hit) {r.p_hit?.toFixed(3) ?? "n/a"}</span>
            {mv?.ball_speed_mps != null && <span className="text-muted">ball {mv.ball_speed_mps.toFixed(1)} m/s</span>}
            {mv?.sync_position_error_m != null && <span className="text-muted">sync position error {(mv.sync_position_error_m * 100).toFixed(1)} cm</span>}
            {mv?.definite_allowed === false && <Badge tone="warn">definite verdict withheld (sync)</Badge>}
          </div>
          {r.warnings.length > 0 && <ul className="list-disc pl-5 text-xs text-warn">{r.warnings.slice(0, 6).map((w, i) => <li key={i}>{w}</li>)}</ul>}
        </div>
      )}
    </Panel>
  );
}
