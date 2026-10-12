"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Badge, Empty, ErrorBox, Panel, statusTone } from "@/components/ui";
import { api, type Job, type Video } from "@/lib/api";

export default function SessionsPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [videos, setVideos] = useState<Record<string, Video>>({});
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    Promise.all([api.jobs(), api.videos()])
      .then(([j, v]) => {
        setJobs(j);
        setVideos(Object.fromEntries(v.map((x) => [x.id, x])));
      })
      .catch(setError);
  }, []);

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold">Analysis sessions</h1>
      <ErrorBox error={error} />
      <Panel>
        {!jobs ? (
          <Empty>Loading...</Empty>
        ) : jobs.length === 0 ? (
          <Empty>No sessions yet.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="pb-2">Created</th>
                  <th className="pb-2">Video</th>
                  <th className="pb-2">Status</th>
                  <th className="pb-2">Detector</th>
                  <th className="pb-2 text-right">Frames</th>
                  <th className="pb-2 text-right">Observed / spanned</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} className="border-t border-line">
                    <td className="tabular py-2 text-muted">{new Date(j.created_at).toLocaleString()}</td>
                    <td>
                      <Link className="text-accent hover:underline" href={`/sessions/${j.id}`}>
                        {videos[j.video_id]?.original_filename ?? j.video_id}
                      </Link>
                    </td>
                    <td><Badge tone={statusTone(j.status)}>{j.status}</Badge></td>
                    <td className="max-w-[240px] truncate text-muted">{j.summary?.detector.model_id ?? String(j.config.detector)}</td>
                    <td className="tabular text-right">{j.summary?.frames_processed ?? "-"}</td>
                    <td className="tabular text-right">
                      {j.summary?.primary_track ? `${j.summary.primary_track.num_observed} / ${j.summary.primary_track.num_frames_spanned}` : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
