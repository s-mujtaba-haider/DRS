"use client";

import { useCallback, useEffect, useState } from "react";

import { DECISION_LABEL } from "@/components/MatchControl";
import ReplayMonitor from "@/components/ReplayMonitor";
import { Badge, Empty, ErrorBox } from "@/components/ui";
import { api, type Review, type Video } from "@/lib/api";

/** Second-screen view for the third umpire: the review's replays only, refreshed as clips arrive. */
export default function UmpireMonitor({ id }: { id: string }) {
  const [review, setReview] = useState<Review | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [error, setError] = useState<unknown>(null);

  const load = useCallback(
    () =>
      api
        .review(id)
        .then(async (r) => {
          const vs = await Promise.all(r.video_ids.map((v) => api.video(v)));
          setReview((old) => (old && old.video_ids.join() === r.video_ids.join() && old.status === r.status ? old : r));
          setVideos((old) => (old.map((v) => v.id).join() === vs.map((v) => v.id).join() ? old : vs));
        })
        .catch(setError),
    [id],
  );
  useEffect(() => {
    void load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [load]);

  if (!review) return error ? <ErrorBox error={error} /> : <Empty>Loading...</Empty>;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-semibold">
          Review #{review.seq} · {DECISION_LABEL[review.decision_type]} {review.over_ball && <span className="text-muted">· over {review.over_ball}</span>}
        </h1>
        <Badge tone={review.status === "open" ? "warn" : "neutral"}>{review.status}</Badge>
        <span className="text-sm text-muted">on-field {review.on_field.replace("_", " ")}</span>
        <span className="text-xs text-muted">Space play/pause · , and . step one frame</span>
      </div>
      {videos.length ? <ReplayMonitor videos={videos} big /> : <Empty>Waiting for replays...</Empty>}
    </div>
  );
}
