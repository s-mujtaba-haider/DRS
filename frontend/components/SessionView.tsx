"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useState } from "react";

import LbwWorkspace from "@/components/LbwWorkspace";
import TrackingWorkspace from "@/components/TrackingWorkspace";
import { Badge, Button, Empty, ErrorBox, KV, Panel, ProgressBar, fmt, statusTone } from "@/components/ui";
import { api, urls, type Annotation, type Calibration, type FrameDetections, type Job, type JobResults, type LbwAnalysis, type Video } from "@/lib/api";

const Trajectory3D = dynamic(() => import("@/components/Trajectory3D"), { ssr: false });
const TABS = ["Tracking", "LBW", "3D", "Contact", "Report"] as const;

export default function SessionView({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [video, setVideo] = useState<Video | null>(null);
  const [results, setResults] = useState<JobResults | null>(null);
  const [detections, setDetections] = useState<FrameDetections[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [analyses, setAnalyses] = useState<LbwAnalysis[]>([]);
  const [calibrations, setCalibrations] = useState<Calibration[]>([]);
  const [tab, setTab] = useState<(typeof TABS)[number]>("Tracking");
  const [impactFrame, setImpactFrame] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);

  const loadAnnotations = useCallback((vid: string) => api.annotations(vid).then(setAnnotations).catch(setError), []);
  const loadAnalyses = useCallback(() => {
    api.lbwList(id).then(setAnalyses).catch(setError);
  }, [id]);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const j = await api.job(id);
        setJob(j);
        if (!video) setVideo(await api.video(j.video_id));
        if (j.status === "succeeded") {
          const [r, d] = await Promise.all([api.results(id), api.detections(id)]);
          setResults(r);
          setDetections(d);
          loadAnnotations(j.video_id);
          loadAnalyses();
          api.calibrations(j.video_id).then(setCalibrations);
        } else if (j.status === "queued" || j.status === "running") {
          timer = setTimeout(poll, 800);
        }
      } catch (e) {
        setError(e);
      }
    };
    poll();
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    if (tab === "3D" && job) api.calibrations(job.video_id).then(setCalibrations);
  }, [tab, job]);

  if (error && !job) return <ErrorBox error={error} />;
  if (!job) return <Empty>Loading session...</Empty>;

  const s = job.summary;
  const latest = analyses[0] ?? null;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">{video?.original_filename ?? "Session"}</h1>
        <Badge tone={statusTone(job.status)}>{job.status}</Badge>
        {s && <Badge tone={s.detector.cricket_specific ? "good" : "warn"}>{s.detector.cricket_specific ? "cricket-trained detector" : "detector not cricket-trained"}</Badge>}
        <span className="text-xs text-muted">job {job.id}</span>
      </div>
      <ErrorBox error={error} />

      {(job.status === "queued" || job.status === "running") && (
        <Panel title="Processing">
          <div className="space-y-2">
            <div className="text-sm text-muted">{job.stage} - {Math.round(job.progress * 100)}%</div>
            <ProgressBar value={job.progress} />
            <Button variant="danger" onClick={() => api.cancelJob(job.id).then(setJob).catch(setError)}>Cancel</Button>
          </div>
        </Panel>
      )}
      {job.status !== "succeeded" && job.status !== "queued" && job.status !== "running" && (
        <Panel title="Job did not complete">
          <ErrorBox error={job.error ?? job.status} />
        </Panel>
      )}

      {results && video && (
        <>
          <div role="tablist" className="flex gap-1 border-b border-line">
            {TABS.map((t) => (
              <button
                key={t}
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={`-mb-px border-b-2 px-3 py-2 text-sm ${tab === t ? "border-accent text-ink" : "border-transparent text-muted hover:text-ink"}`}
              >
                {t}
              </button>
            ))}
          </div>

          {tab === "Tracking" && (
            <TrackingWorkspace
              results={results}
              detections={detections}
              annotations={annotations}
              videoId={video.id}
              onAnnotationsChanged={() => loadAnnotations(video.id)}
              impactFrame={impactFrame}
              setImpactFrame={setImpactFrame}
            />
          )}
          {tab === "LBW" && <LbwWorkspace results={results} videoId={video.id} impactFrame={impactFrame} analyses={analyses} onNewAnalysis={loadAnalyses} />}
          {tab === "3D" && <Trajectory3D analysis={latest} calibration={calibrations.find((c) => c.id === latest?.calibration_id) ?? null} />}
          {tab === "Contact" && <ContactTab jobId={job.id} />}
          {tab === "Report" && s && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Panel title="Downloads">
                <ul className="space-y-2 text-sm">
                  <li><a className="text-accent hover:underline" href={urls.report(job.id)}>PDF report</a></li>
                  <li><a className="text-accent hover:underline" href={urls.exportJson(job.id)}>Full results (JSON)</a></li>
                  <li><a className="text-accent hover:underline" href={urls.artifact(job.id, "track_points.csv")}>Track points (CSV)</a></li>
                  <li><a className="text-accent hover:underline" href={urls.artifact(job.id, "annotated.mp4")}>Annotated video (MP4)</a></li>
                  <li><a className="text-accent hover:underline" href={urls.artifact(job.id, "detections.json")}>Raw detections (JSON)</a></li>
                </ul>
              </Panel>
              <Panel title="Processing summary">
                <KV
                  rows={[
                    ["Detector", s.detector.model_id],
                    ["Frames processed", s.frames_processed],
                    ["Frames with candidates", s.frames_with_candidates],
                    ["Detections", s.num_detections],
                    ["2D trajectory", s.trajectory_2d ? `${s.trajectory_2d.model} (${s.trajectory_2d.status}), rms ${fmt(s.trajectory_2d.rms_px, 2)} px` : null],
                    ["Throughput", `${fmt(s.throughput_fps.detect, 1)} fps detect, ${fmt(s.throughput_fps.end_to_end, 1)} fps end-to-end`],
                    ["GPU peak memory", s.gpu_peak_mb ? `${fmt(s.gpu_peak_mb, 0)} MB` : "n/a"],
                    ["Validation", s.validation_status],
                  ]}
                />
                {s.detector.notes.map((n) => <div key={n} className="mt-2 text-xs text-muted">- {n}</div>)}
                {s.warnings.map((w) => <div key={w} className="mt-1 text-xs text-warn">! {w}</div>)}
              </Panel>
            </div>
          )}
        </>
      )}
    </div>
  );
}

interface ContactResult {
  status: string;
  frame_rate_fps: number | null;
  observations_used: number;
  candidates: { frame_index: number; direction_change_deg: number; speed_ratio: number }[];
  limitations: string[];
  alternative_explanations: string[];
}

function ContactTab({ jobId }: { jobId: string }) {
  const [res, setRes] = useState<ContactResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  return (
    <Panel title="Experimental bat-ball contact candidates">
      <p className="mb-3 text-sm text-muted">
        Looks for abrupt changes in the ball&apos;s image-space motion. A deflection is not proof of bat contact, and its absence does not prove there was none.
      </p>
      <Button variant="primary" onClick={() => api.contact(jobId).then((r) => setRes(r as unknown as ContactResult)).catch(setError)}>Analyse trajectory changes</Button>
      <div className="mt-3"><ErrorBox error={error} /></div>
      {res && (
        <div className="mt-3 space-y-2 text-sm">
          <Badge tone={res.status === "CANDIDATES_FOUND" ? "info" : "warn"}>{res.status}</Badge>
          <div className="text-muted">Frame rate {fmt(res.frame_rate_fps, 1)} fps, {res.observations_used} observations</div>
          {res.candidates.map((c) => (
            <div key={c.frame_index} className="tabular">frame {c.frame_index}: direction change {fmt(c.direction_change_deg, 1)} deg, speed ratio {fmt(c.speed_ratio, 2)}</div>
          ))}
          {res.limitations.map((l) => <div key={l} className="text-xs text-warn">! {l}</div>)}
          <div className="text-xs text-muted">Alternative explanations: {res.alternative_explanations.join("; ")}</div>
        </div>
      )}
    </Panel>
  );
}
