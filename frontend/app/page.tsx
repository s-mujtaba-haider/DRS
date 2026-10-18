"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Badge, Empty, ErrorBox, KV, Panel, Stat, fmt, statusTone } from "@/components/ui";
import { api, type CaptureStatus, type Job, type Match, type Stats, type SystemStatus, type Video } from "@/lib/api";

interface BenchSummary {
  dataset: string;
  num_clips: number;
  num_with_projection: number;
  stump_plane_error_m: { median: number; p90: number } | null;
  interval_90_empirical_coverage: number | null;
  definite_verdicts: number;
  definite_verdict_accuracy: number | null;
  inconclusive_rate: number | null;
  caveat: string;
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [videos, setVideos] = useState<Record<string, Video>>({});
  const [sys, setSys] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [matches, setMatches] = useState<Match[]>([]);
  const [capture, setCapture] = useState<CaptureStatus | null>(null);

  useEffect(() => {
    Promise.all([api.stats(), api.jobs(), api.videos(), api.system()])
      .then(([s, j, v, sy]) => {
        setStats(s);
        setJobs(j.slice(0, 8));
        setVideos(Object.fromEntries(v.map((x) => [x.id, x])));
        setSys(sy);
      })
      .catch(setError);
    api.matches().then((m) => setMatches(m.filter((x) => x.status === "live"))).catch(() => undefined);
    api.captureStatus().then(setCapture).catch(() => undefined);
  }, []);

  const bench = stats?.trajectory_benchmark?.summary as BenchSummary | undefined;
  const running = (stats?.jobs_by_status.running ?? 0) + (stats?.jobs_by_status.queued ?? 0);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Dashboard</h1>
          <p className="text-sm text-muted">Experimental research system. Results are not official umpiring decisions.</p>
        </div>
        <Link href="/upload" className="rounded border border-accent/60 bg-accent/15 px-3 py-1.5 text-sm text-accent hover:bg-accent/25">
          New analysis
        </Link>
      </div>
      <ErrorBox error={error} />

      {(matches.length > 0 || (capture?.cameras.some((c) => c.running) ?? false)) && (
        <Panel title="Match day">
          <div className="flex flex-wrap items-center gap-4 text-sm">
            {matches.map((m) => (
              <Link key={m.id} href={`/matches/${m.id}`} className="rounded border border-line bg-panel-2 px-3 py-2 hover:bg-line">
                <div className="font-medium">{m.name}</div>
                <div className="text-xs text-muted">
                  innings {m.current_innings} · {Object.entries(m.reviews.teams).map(([t, c]) => `${t} ${c.remaining ?? "∞"}`).join(" · ")} reviews left
                </div>
              </Link>
            ))}
            {capture && (
              <Link href="/capture" className="text-muted hover:text-ink">
                <Badge tone={capture.cameras.some((c) => c.running) ? "good" : "neutral"}>
                  {capture.cameras.filter((c) => c.running).length} camera(s) recording
                </Badge>
              </Link>
            )}
          </div>
        </Panel>
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Videos" value={stats?.videos ?? "-"} />
        <Stat label="LBW analyses" value={stats?.lbw_analyses ?? "-"} hint="Deliveries analysed with the experimental engine" />
        <Stat label="Jobs active" value={stats ? running : "-"} hint={stats ? `${stats.jobs_succeeded} succeeded, ${stats.jobs_by_status.failed ?? 0} failed` : undefined} />
        <Stat
          label="Ball track found"
          value={stats && stats.jobs_succeeded ? `${stats.jobs_with_primary_track}/${stats.jobs_succeeded}` : "-"}
          hint="A track was found. Correctness is unmeasured without ground truth."
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Panel title="Recent sessions" className="xl:col-span-2">
          {jobs.length === 0 ? (
            <Empty>No analyses yet. Upload a cricket video to start.</Empty>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="pb-2">Video</th>
                  <th className="pb-2">Status</th>
                  <th className="pb-2">Detector</th>
                  <th className="pb-2 text-right">Track obs.</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} className="border-t border-line">
                    <td className="py-2">
                      <Link className="text-accent hover:underline" href={`/sessions/${j.id}`}>
                        {videos[j.video_id]?.original_filename ?? j.video_id}
                      </Link>
                    </td>
                    <td>
                      <Badge tone={statusTone(j.status)}>{j.status}</Badge>
                    </td>
                    <td className="max-w-[220px] truncate text-muted">{j.summary?.detector.model_id ?? String(j.config.detector)}</td>
                    <td className="tabular text-right">{j.summary?.primary_track ? `${j.summary.primary_track.num_observed}/${j.summary.primary_track.num_frames_spanned}` : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        <Panel title="Models and hardware">
          {sys ? (
            <KV
              rows={[
                ["GPU", sys.gpu ? `${sys.gpu.name} (${(sys.gpu.memory_total_mb / 1024).toFixed(0)} GB)` : "not detected"],
                ["PyTorch", sys.torch.available ? `${sys.torch.version} ${sys.torch.cuda_available ? "CUDA" : "CPU"}` : <span className="text-warn">unavailable in this process</span>],
                ["Detector mode", <Badge key="m" tone={sys.cricket_detector_available ? "good" : "warn"}>{sys.effective_detector_mode}</Badge>],
                ["Checkpoints", sys.models.length ? sys.models.map((m) => m.file).join(", ") : "none"],
              ]}
            />
          ) : (
            <Empty>Loading...</Empty>
          )}
          {sys && !sys.cricket_detector_available && (
            <p className="mt-3 text-xs text-warn">
              No cricket-trained detector is installed. Detection uses generic or motion-based candidates. See docs/MODEL_CARD.md.
            </p>
          )}
        </Panel>
      </div>

      <Panel title="Trajectory benchmark (synthetic data)">
        {bench ? (
          <div className="space-y-2 text-sm">
            <KV
              rows={[
                ["Dataset", String(bench.dataset)],
                ["Clips / with projection", `${bench.num_clips} / ${bench.num_with_projection}`],
                ["Stump-plane error, median / p90", `${fmt(bench.stump_plane_error_m ? bench.stump_plane_error_m.median * 100 : null, 1)} cm / ${fmt(bench.stump_plane_error_m ? bench.stump_plane_error_m.p90 * 100 : null, 1)} cm`],
                ["90% interval coverage", fmt(bench.interval_90_empirical_coverage, 2)],
                ["Definite verdicts (accuracy)", `${bench.definite_verdicts} (${fmt(bench.definite_verdict_accuracy, 3)})`],
                ["Inconclusive rate", fmt(bench.inconclusive_rate, 2)],
              ]}
            />
            <p className="text-xs text-muted">{String(bench.caveat)}</p>
          </div>
        ) : (
          <Empty>No benchmark report found. Run scripts/evaluate_trajectory_synth.py.</Empty>
        )}
      </Panel>
    </div>
  );
}
