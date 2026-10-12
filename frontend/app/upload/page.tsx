"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { Badge, Button, ErrorBox, KV, Panel, ProgressBar, fmt, statusTone } from "@/components/ui";
import { api, type Job, type Video } from "@/lib/api";

const DETECTORS = [
  { value: "auto", label: "Auto (best available)" },
  { value: "cricket", label: "Cricket-trained YOLO (requires trained weights)" },
  { value: "coco_hybrid", label: "Generic COCO 'sports ball' + motion candidates" },
  { value: "motion", label: "Motion candidates only (no learned model)" },
];

export default function UploadPage() {
  const [video, setVideo] = useState<Video | null>(null);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [cfg, setCfg] = useState({ detector: "auto", conf: 0.15, imgsz: 1280, frame_step: 1, start_frame: 0, end_frame: "", tile_size: "", batch_size: 8 });
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => void (timer.current && clearInterval(timer.current)), []);

  async function onFile(f: File) {
    setError(null);
    setVideo(null);
    setJob(null);
    setUploadPct(0);
    try {
      setVideo(await api.uploadVideo(f, setUploadPct));
    } catch (e) {
      setError(e);
    } finally {
      setUploadPct(null);
    }
  }

  async function start() {
    if (!video) return;
    setError(null);
    try {
      const body: Record<string, unknown> = {
        video_id: video.id,
        detector: cfg.detector,
        conf: Number(cfg.conf),
        imgsz: Number(cfg.imgsz),
        frame_step: Number(cfg.frame_step),
        start_frame: Number(cfg.start_frame),
        batch_size: Number(cfg.batch_size),
      };
      if (cfg.end_frame) body.end_frame = Number(cfg.end_frame);
      if (cfg.tile_size) body.tile_size = Number(cfg.tile_size);
      const j = await api.createJob(body);
      setJob(j);
      timer.current = setInterval(async () => {
        try {
          const cur = await api.job(j.id);
          setJob(cur);
          if (["succeeded", "failed", "cancelled", "interrupted"].includes(cur.status) && timer.current) clearInterval(timer.current);
        } catch (e) {
          setError(e);
        }
      }, 700);
    } catch (e) {
      setError(e);
    }
  }

  const m = video?.meta;
  const active = job && ["queued", "running"].includes(job.status);
  const field = "w-full rounded border border-line bg-panel-2 px-2 py-1.5 text-sm";

  return (
    <div className="max-w-5xl space-y-5">
      <h1 className="text-xl font-semibold">New analysis</h1>
      <ErrorBox error={error} />

      <Panel title="1. Video">
        <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded border border-dashed border-line px-4 py-8 text-sm text-muted hover:border-accent/60">
          <span>Choose a cricket video (MP4, MOV, AVI, MKV, WEBM)</span>
          <input type="file" accept="video/*,.mp4,.mov,.avi,.mkv,.webm" className="sr-only" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
          <span className="rounded border border-line bg-panel-2 px-3 py-1 text-ink">Browse</span>
        </label>
        {uploadPct !== null && (
          <div className="mt-3 space-y-1">
            <div className="text-xs text-muted">Uploading and validating... {Math.round(uploadPct * 100)}%</div>
            <ProgressBar value={uploadPct} />
          </div>
        )}
        {m && (
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <KV
              rows={[
                ["File", video!.original_filename],
                ["Resolution", `${m.width} x ${m.height}`],
                ["Declared fps", fmt(m.nominal_fps, 3)],
                ["Measured frame interval", m.pts_delta_ms_mean ? `${fmt(m.pts_delta_ms_mean, 2)} ms` : "n/a"],
                ["Variable frame rate", m.variable_frame_rate === null ? "unknown" : m.variable_frame_rate ? "yes" : "no"],
              ]}
            />
            <KV
              rows={[
                ["Duration", m.duration_s ? `${fmt(m.duration_s, 2)} s` : "unknown"],
                ["Frames (declared)", m.frame_count_declared ?? "unknown"],
                ["Codec / container", `${m.codec} / ${m.container_format}`],
                ["Audio", m.has_audio ? "yes" : "no"],
                ["Size", `${(video!.size_bytes / 1e6).toFixed(1)} MB`],
              ]}
            />
            {m.warnings.length > 0 && (
              <div className="md:col-span-2 space-y-1">
                {m.warnings.map((w) => (
                  <div key={w} className="text-xs text-warn">! {w}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </Panel>

      <Panel title="2. Processing configuration">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="space-y-1 sm:col-span-2">
            <span className="text-xs text-muted">Detector</span>
            <select className={field} value={cfg.detector} onChange={(e) => setCfg({ ...cfg, detector: e.target.value })}>
              {DETECTORS.map((d) => (
                <option key={d.value} value={d.value}>{d.label}</option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-xs text-muted">Confidence threshold</span>
            <input className={field} type="number" step="0.01" min="0.01" max="0.99" value={cfg.conf} onChange={(e) => setCfg({ ...cfg, conf: Number(e.target.value) })} />
          </label>
          <label className="space-y-1">
            <span className="text-xs text-muted">Inference size (px)</span>
            <input className={field} type="number" step="32" min="320" max="3840" value={cfg.imgsz} onChange={(e) => setCfg({ ...cfg, imgsz: Number(e.target.value) })} />
          </label>
          <label className="space-y-1">
            <span className="text-xs text-muted">Tile size (optional, small balls)</span>
            <input className={field} type="number" placeholder="off" value={cfg.tile_size} onChange={(e) => setCfg({ ...cfg, tile_size: e.target.value })} />
          </label>
          <label className="space-y-1">
            <span className="text-xs text-muted">Frame step</span>
            <input className={field} type="number" min="1" max="30" value={cfg.frame_step} onChange={(e) => setCfg({ ...cfg, frame_step: Number(e.target.value) })} />
          </label>
          <label className="space-y-1">
            <span className="text-xs text-muted">Start frame</span>
            <input className={field} type="number" min="0" value={cfg.start_frame} onChange={(e) => setCfg({ ...cfg, start_frame: Number(e.target.value) })} />
          </label>
          <label className="space-y-1">
            <span className="text-xs text-muted">End frame (optional)</span>
            <input className={field} type="number" min="1" placeholder="end" value={cfg.end_frame} onChange={(e) => setCfg({ ...cfg, end_frame: e.target.value })} />
          </label>
        </div>
        <div className="mt-4 flex items-center gap-2">
          <Button variant="primary" disabled={!video || !!active} onClick={start}>Start analysis</Button>
          {active && <Button variant="danger" onClick={() => job && api.cancelJob(job.id).then(setJob).catch(setError)}>Cancel</Button>}
        </div>
      </Panel>

      {job && (
        <Panel title="3. Progress">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Badge tone={statusTone(job.status)}>{job.status}</Badge>
              <span className="text-muted">{job.stage ?? ""}</span>
              <span className="tabular ml-auto text-muted">{Math.round(job.progress * 100)}%</span>
            </div>
            <ProgressBar value={job.progress} />
            {job.error && <ErrorBox error={job.error} />}
            {job.status === "succeeded" && (
              <div className="space-y-2 pt-2">
                {job.summary?.warnings.map((w) => (
                  <div key={w} className="text-xs text-warn">! {w}</div>
                ))}
                <Link href={`/sessions/${job.id}`} className="inline-block rounded border border-accent/60 bg-accent/15 px-3 py-1.5 text-sm text-accent">
                  Open review workspace
                </Link>
              </div>
            )}
          </div>
        </Panel>
      )}
    </div>
  );
}
