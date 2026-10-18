"use client";

import { useCallback, useEffect, useState } from "react";

import CaptureClipButton from "@/components/CaptureClipButton";
import { Badge, Button, Empty, ErrorBox, Panel, fmt } from "@/components/ui";
import { api, urls, type CaptureStatus, type Video } from "@/lib/api";

interface Row { camera_id: string; type: "opencv" | "synthetic" | "video"; uri: string; width: string; height: string; fps: string; video_id?: string }
const input = "w-full rounded border border-line bg-panel-2 px-2 py-1 text-sm";
const STORE = "drs.capture.cameras";

function loadRows(): Row[] {
  try {
    const v = JSON.parse(localStorage.getItem(STORE) ?? "null");
    if (Array.isArray(v) && v.length) return v;
  } catch {
    /* storage unavailable */
  }
  return [{ camera_id: "end_on", type: "opencv", uri: "0", width: "1280", height: "720", fps: "60" }];
}

export default function CapturePage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [status, setStatus] = useState<CaptureStatus | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);
  const [retention, setRetention] = useState("300");
  const [videos, setVideos] = useState<Video[]>([]);
  useEffect(() => {
    api.videos().then(setVideos).catch(() => undefined);
  }, []);

  // localStorage is only readable after hydration, so the saved rows load in an effect.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setRows(loadRows()), []);
  useEffect(() => {
    try {
      if (rows.length) localStorage.setItem(STORE, JSON.stringify(rows));
    } catch {
      /* ignore */
    }
  }, [rows]);

  const refresh = useCallback(() => api.captureStatus().then(setStatus).catch(setError), []);
  useEffect(() => {
    refresh();
    const t = setInterval(() => {
      refresh();
      setTick((n) => n + 1);
    }, 700);
    return () => clearInterval(t);
  }, [refresh]);

  const live = status?.cameras.filter((c) => c.running) ?? [];
  const set = (i: number, k: keyof Row, v: string) => setRows(rows.map((r, j) => (j === i ? { ...r, [k]: v } : r)));

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const cams = rows.map((r) =>
        r.type === "video"
          ? { camera_id: r.camera_id, type: "video", video_id: r.video_id, loop: true }
          : {
              camera_id: r.camera_id,
              type: r.type,
              uri: r.type === "opencv" ? r.uri : undefined,
              width: r.width ? Number(r.width) : undefined,
              height: r.height ? Number(r.height) : undefined,
              fps: r.fps ? Number(r.fps) : undefined,
            },
      );
      setStatus(await api.captureStart({ cameras: cams, retention_s: Number(retention) || undefined }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold">Live capture</h1>
        <CaptureClipButton onClipped={refresh} />
      </div>
      <div className="rounded border border-warn/40 bg-warn/10 px-3 py-2 text-sm text-warn">
        Software synchronisation: every camera is timestamped on frame arrival at this PC on one shared clock, not at exposure. Expect
        sync error of about one frame interval plus USB/network jitter. That is roughly 3.5 cm of 3D error per millisecond for a 37 m/s
        ball. Multi-camera reconstruction from these clips is flagged as degraded. See docs/CAMERA_HARDWARE.md for hardware-triggered setups.
      </div>
      <ErrorBox error={error} />

      <Panel title="Cameras" actions={
        <div className="flex items-center gap-2 text-sm">
          <label className="flex items-center gap-1 text-muted">buffer
            <input className="w-16 rounded border border-line bg-panel-2 px-1.5 py-1" value={retention} onChange={(e) => setRetention(e.target.value)} /> s
          </label>
          <Button variant="primary" disabled={busy || rows.length === 0} onClick={start}>{busy ? "Starting..." : "Start recording"}</Button>
          <Button variant="danger" disabled={!live.length} onClick={() => api.captureStop().then(setStatus).catch(setError)}>Stop all</Button>
        </div>
      }>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="text-left text-xs uppercase tracking-wider text-muted">
              <tr><th className="pb-2">Camera id</th><th>Type</th><th>Device / URL</th><th>Width</th><th>Height</th><th>FPS</th><th /></tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-t border-line">
                  <td className="py-1.5 pr-2"><input className={input} value={r.camera_id} onChange={(e) => set(i, "camera_id", e.target.value)} /></td>
                  <td className="pr-2">
                    <select className={input} value={r.type} onChange={(e) => set(i, "type", e.target.value)}>
                      <option value="opencv">USB / stream</option>
                      <option value="video">Rehearsal: replay a video</option>
                      <option value="synthetic">Synthetic test</option>
                    </select>
                  </td>
                  <td className="pr-2">
                    {r.type === "video" ? (
                      <select className={input} value={r.video_id ?? ""} onChange={(e) => set(i, "video_id", e.target.value)}>
                        <option value="" disabled>Choose uploaded video...</option>
                        {videos.map((v) => <option key={v.id} value={v.id}>{v.original_filename}</option>)}
                      </select>
                    ) : (
                      <input className={input} disabled={r.type !== "opencv"} placeholder="0 or rtsp://..." value={r.uri} onChange={(e) => set(i, "uri", e.target.value)} />
                    )}
                  </td>
                  <td className="pr-2"><input className={input} value={r.width} onChange={(e) => set(i, "width", e.target.value)} /></td>
                  <td className="pr-2"><input className={input} value={r.height} onChange={(e) => set(i, "height", e.target.value)} /></td>
                  <td className="pr-2"><input className={input} value={r.fps} onChange={(e) => set(i, "fps", e.target.value)} /></td>
                  <td><Button variant="ghost" onClick={() => setRows(rows.filter((_, j) => j !== i))}>×</Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2">
          <Button onClick={() => setRows([...rows, { camera_id: `cam${rows.length + 1}`, type: "opencv", uri: String(rows.length), width: "1280", height: "720", fps: "60" }])}>Add camera</Button>
        </div>
      </Panel>

      {status && status.cameras.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2">
          {status.cameras.map((c) => {
            const drops = c.source_dropped_est + c.queue_dropped;
            return (
              <Panel key={c.camera_id} title={c.camera_id} actions={
                <div className="flex items-center gap-2">
                  <Badge tone={c.error ? "bad" : c.running ? "good" : "neutral"}>{c.error ? "error" : c.running ? "recording" : "stopped"}</Badge>
                  {c.running && <Button variant="ghost" onClick={() => api.captureStop(c.camera_id).then(setStatus).catch(setError)}>Stop</Button>}
                </div>
              }>
                {c.running ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={urls.capturePreview(c.camera_id, tick)} alt={`${c.camera_id} live preview`} className="mb-3 w-full rounded border border-line bg-black" />
                ) : null}
                {c.error && <div className="mb-2 text-sm text-bad">{c.error}</div>}
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
                  <div><dt className="text-muted">Resolution</dt><dd className="tabular">{c.width}×{c.height}</dd></div>
                  <div><dt className="text-muted">FPS measured / declared</dt><dd className={`tabular ${c.rate_mismatch ? "text-warn" : ""}`}>{fmt(c.measured_fps, 1)} / {fmt(c.nominal_fps, 0)}</dd></div>
                  <div><dt className="text-muted">Dropped frames</dt><dd className={`tabular ${drops ? "text-warn" : ""}`}>{drops} ({c.queue_dropped} encoder)</dd></div>
                  <div><dt className="text-muted">Encoder</dt><dd className="tabular">{c.encoder} · {fmt(c.encode_ms_per_frame, 1)} ms/f</dd></div>
                  <div><dt className="text-muted">Backlog</dt><dd className="tabular">{c.encoder_backlog} frames</dd></div>
                  <div><dt className="text-muted">Arrival jitter / late</dt><dd className={`tabular ${c.late_arrivals ? "text-warn" : ""}`}>{fmt(c.arrival_jitter_ms, 1)} ms · {c.late_arrivals ?? 0} late</dd></div>
                  <div><dt className="text-muted">Buffer</dt><dd className="tabular">{fmt(c.buffer_span_s, 0)} s · {(c.disk_bytes / 1e9).toFixed(2)} GB</dd></div>
                  <div><dt className="text-muted">Frames</dt><dd className="tabular">{c.frames_captured} / {c.frames_encoded} enc</dd></div>
                  <div><dt className="text-muted">Disk free</dt><dd className="tabular">{c.disk_free_bytes != null ? `${(c.disk_free_bytes / 1e9).toFixed(0)} GB` : "n/a"}</dd></div>
                </dl>
              </Panel>
            );
          })}
        </div>
      ) : (
        <Empty>No cameras running. Configure devices above and start recording. Clips can then be pulled into any open review.</Empty>
      )}
    </div>
  );
}
