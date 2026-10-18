"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import FrameViewer from "@/components/FrameViewer";
import { Badge, Button, Empty, ErrorBox, Panel } from "@/components/ui";
import { api, type Annotation, type Video } from "@/lib/api";

interface Suggestion { x: number; y: number; radius_px: number; confidence: number }

export default function LabelPage() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [vid, setVid] = useState("");
  const [step, setStep] = useState(5);
  const [frames, setFrames] = useState<number[]>([]);
  const [pos, setPos] = useState(0);
  const [frame, setFrame] = useState(0);
  const [anns, setAnns] = useState<Annotation[]>([]);
  const [radius, setRadius] = useState(6);
  const [sugg, setSugg] = useState<Suggestion[] | null>(null);
  const [autoSugg, setAutoSugg] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [exportName, setExportName] = useState("ui_ball_labels");
  const [exportMsg, setExportMsg] = useState<string | null>(null);
  const video = videos.find((v) => v.id === vid);

  useEffect(() => {
    api.videos().then((vs) => {
      setVideos(vs);
      if (vs[0]) setVid((x) => x || vs[0].id);
    }).catch(setError);
  }, []);

  const loadPlan = useCallback(() => {
    if (!vid) return;
    Promise.all([api.labelPlan(vid, step), api.annotations(vid)])
      .then(([p, a]) => {
        setFrames(p.frames ?? []);
        setAnns(a);
        const labelled = new Set(a.filter((x) => x.label === "ball").map((x) => x.frame_index));
        const first = p.frames.findIndex((f) => !labelled.has(f));
        const i = first < 0 ? 0 : first;
        setPos(i);
        setFrame(p.frames[i] ?? 0);
      })
      .catch(setError);
  }, [vid, step]);
  useEffect(loadPlan, [loadPlan]);

  const latest = useMemo(() => {
    const m = new Map<number, Annotation>();
    [...anns].filter((a) => a.label === "ball").sort((a, b) => a.created_at.localeCompare(b.created_at)).forEach((a) => m.set(a.frame_index, a));
    return m;
  }, [anns]);
  const cur = latest.get(frame);
  const done = frames.filter((f) => { const a = latest.get(f); return a && a.action !== "reject"; }).length;
  const counts = useMemo(() => {
    let b = 0, n = 0;
    latest.forEach((a) => (a.action === "mark_absent" ? n++ : a.action !== "reject" && b++));
    return { b, n };
  }, [latest]);

  const go = useCallback((i: number) => {
    const j = Math.max(0, Math.min(frames.length - 1, i));
    setPos(j);
    setFrame(frames[j] ?? 0);
    setSugg(null);
  }, [frames]);

  useEffect(() => {
    if (!autoSugg || !vid) return;
    let alive = true;
    api
      .labelPrelabel(vid, frame)
      .then((d) => alive && setSugg(d.suggestions))
      .catch(() => alive && setSugg([]));
    return () => {
      alive = false;
    };
  }, [vid, frame, autoSugg]);

  const save = useCallback(async (body: Record<string, unknown>) => {
    try {
      const a = await api.addAnnotation(vid, { frame_index: frame, label: "ball", note: "labeling", ...body });
      setAnns((x) => [...x, a]);
      setError(null);
      go(pos + 1);
    } catch (e) {
      setError(e);
    }
  }, [vid, frame, pos, go]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest("input,textarea,select")) return;
      const key = e.key.toLowerCase();
      if (key === "n") void save({ action: "mark_absent" });
      else if (key === "s") go(pos + 1);
      else if (key === "backspace") go(pos - 1);
      else if (key === "enter" && sugg?.[0]) void save({ action: "add", x: sugg[0].x, y: sugg[0].y, radius_px: sugg[0].radius_px });
      else if (key === "[") setRadius((r) => Math.max(2, r - 1));
      else if (key === "]") setRadius((r) => Math.min(60, r + 1));
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [save, go, pos, sugg]);

  const doExport = async () => {
    setExportMsg(null);
    try {
      const d = await api.labelExport({ video_ids: [vid], name: exportName });
      setExportMsg(`${d.positives} ball + ${d.negatives} no-ball frames → ${d.path}. Next: ${d.next_step}`);
    } catch (e) {
      setExportMsg(String(e instanceof Error ? e.message : e));
    }
  };

  const r = Math.max(radius, 2);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Label ball frames</h1>
      <p className="max-w-3xl text-sm text-muted">
        Human labels on your own footage are the main way to improve broadcast detection (see Roadmap §1). Label every few frames:
        click the ball centre, or press <kbd>N</kbd> when no ball is visible. Model suggestions (magenta) are hints only; <kbd>Enter</kbd> accepts the top one.
        Export, then build a dataset split by video with <code>annotate_video.py build</code>.
      </p>
      <ErrorBox error={error} />
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <select className="rounded border border-line bg-panel-2 px-2 py-1" value={vid} onChange={(e) => setVid(e.target.value)} aria-label="Video">
          {videos.map((v) => <option key={v.id} value={v.id}>{v.original_filename}</option>)}
        </select>
        <label className="flex items-center gap-1">every
          <input type="number" min={1} max={500} value={step} onChange={(e) => setStep(Math.max(1, Number(e.target.value) || 1))} className="w-16 rounded border border-line bg-panel-2 px-1.5 py-1" /> frames
        </label>
        <label className="flex items-center gap-1"><input type="checkbox" checked={autoSugg} onChange={(e) => setAutoSugg(e.target.checked)} /> model suggestions</label>
        <span className="text-muted">ball radius {radius}px (<kbd>[</kbd>/<kbd>]</kbd>)</span>
      </div>
      {!video ? (
        <Empty>Upload a video first.</Empty>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[1fr_300px]">
          <FrameViewer
            videoId={video.id}
            width={video.meta.width}
            height={video.meta.height}
            frameCount={video.meta.frame_count_decoded ?? video.meta.frame_count_declared}
            frame={frame}
            setFrame={setFrame}
            onImageClick={(x, y) => void save({ action: "add", x, y, radius_px: r })}
            crosshair
            overlay={
              <>
                {sugg?.map((s, i) => (
                  <circle key={i} cx={s.x} cy={s.y} r={Math.max(s.radius_px, 4) + 4} fill="none" stroke="#e879f9" strokeWidth={i === 0 ? 2.5 : 1} opacity={0.4 + 0.6 * s.confidence} />
                ))}
                {cur && cur.action !== "mark_absent" && cur.action !== "reject" && cur.data.x != null && (
                  <circle cx={Number(cur.data.x)} cy={Number(cur.data.y)} r={Number(cur.data.radius_px ?? r)} fill="none" stroke="#4ade80" strokeWidth={2} />
                )}
              </>
            }
          />
          <div className="space-y-3 text-sm">
            <Panel title="Progress">
              <div className="tabular">frame {pos + 1} / {frames.length} in plan · labelled {done}</div>
              <div className="tabular text-muted">all frames: {counts.b} ball · {counts.n} no ball</div>
              <div className="mt-2">
                {cur ? <Badge tone={cur.action === "mark_absent" ? "neutral" : "good"}>{cur.action === "mark_absent" ? "no ball" : "ball labelled"}</Badge> : <Badge tone="warn">unlabelled</Badge>}
              </div>
              <div className="mt-3 flex flex-wrap gap-1">
                <Button onClick={() => go(pos - 1)}>◀ prev</Button>
                <Button onClick={() => go(pos + 1)}>skip ▶</Button>
                <Button onClick={() => void save({ action: "mark_absent" })}>No ball (N)</Button>
                <Button disabled={!sugg?.length} onClick={() => sugg?.[0] && void save({ action: "add", x: sugg[0].x, y: sugg[0].y, radius_px: sugg[0].radius_px })}>Accept (Enter)</Button>
              </div>
              {sugg && sugg.length > 0 && <div className="mt-2 text-xs text-muted">top suggestion confidence {sugg[0].confidence.toFixed(2)}</div>}
            </Panel>
            <Panel title="Export labelled set">
              <input className="mb-2 w-full rounded border border-line bg-panel-2 px-2 py-1" value={exportName} onChange={(e) => setExportName(e.target.value)} aria-label="Set name" />
              <Button variant="primary" onClick={doExport}>Export this video</Button>
              {exportMsg && <div className="mt-2 break-words text-xs text-muted">{exportMsg}</div>}
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
