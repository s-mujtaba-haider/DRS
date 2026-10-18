"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import CreaseWorkspace from "@/components/CreaseWorkspace";
import AudioEdgePanel from "@/components/AudioEdgePanel";
import BoundaryWorkspace from "@/components/BoundaryWorkspace";
import CaptureClipButton from "@/components/CaptureClipButton";
import { DECISION_LABEL, outcomeTone } from "@/components/MatchControl";
import MultiviewLbwPanel from "@/components/MultiviewLbwPanel";
import ReplayMonitor from "@/components/ReplayMonitor";
import { Badge, Button, Empty, ErrorBox, KV, Panel } from "@/components/ui";
import { api, urls, type DecisionAnalysis, type Job, type LbwAnalysis, type Match, type Review, type Video } from "@/lib/api";

const input = "w-full rounded border border-line bg-panel-2 px-2 py-1.5 text-sm";

export default function ReviewWorkspace({ id }: { id: string }) {
  const [review, setReview] = useState<Review | null>(null);
  const [match, setMatch] = useState<Match | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [activeVideo, setActiveVideo] = useState<string | null>(null);

  const load = useCallback(
    () =>
      api
        .review(id)
        .then(async (r) => {
          const [m, vs] = await Promise.all([api.match(r.match_id), api.videos()]);
          setReview(r);
          setMatch(m);
          setVideos(vs);
          setActiveVideo((v) => v ?? r.video_ids[0] ?? null);
        })
        .catch(setError),
    [id],
  );
  useEffect(() => void load(), [load]);

  if (!review || !match) return error ? <ErrorBox error={error} /> : <Empty>Loading review...</Empty>;
  const isOpen = review.status === "open";
  const byId = Object.fromEntries(videos.map((v) => [v.id, v]));
  const act = async (fn: () => Promise<Review>) => {
    try {
      setReview(await fn());
      setError(null);
      const r = await api.review(id);
      setReview(r);
    } catch (e) {
      setError(e);
    }
  };
  const attach = (type: "lbw" | "decision", evidenceId: string) => act(() => api.attachEvidence(id, type, evidenceId));
  const video = activeVideo ? byId[activeVideo] : null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link href={`/matches/${match.id}`} className="text-xs text-muted hover:text-ink">← {match.name}</Link>
          <h1 className="text-xl font-semibold">
            Review #{review.seq}: {DECISION_LABEL[review.decision_type]}
            {review.over_ball && <span className="text-muted"> · over {review.over_ball}</span>}
          </h1>
          <div className="text-sm text-muted">
            {review.kind === "umpire" ? "Umpire referral" : `Player review by ${review.requested_by}`} · on-field{" "}
            <span className="text-ink">{review.on_field.replace("_", " ")}</span>
            {review.batter && ` · batter ${review.batter}`}
            {review.bowler && ` · bowler ${review.bowler}`} · innings {review.innings}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={isOpen ? "warn" : "neutral"}>{review.status}</Badge>
          <a className="rounded border border-line bg-panel-2 px-3 py-1.5 text-sm hover:bg-line" href={urls.reviewPackage(review.id)} target="_blank" rel="noreferrer">
            Evidence package (JSON)
          </a>
          <a className="rounded border border-line bg-panel-2 px-3 py-1.5 text-sm hover:bg-line" href={urls.reviewReport(review.id)} target="_blank" rel="noreferrer">
            PDF report
          </a>
        </div>
      </div>
      <ErrorBox error={error} />

      <Panel
        title="Replay sources"
        actions={isOpen && <CaptureClipButton reviewId={id} onClipped={load} />}
      >
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {review.video_ids.length === 0 && <span className="text-muted">No video attached yet.</span>}
          {review.video_ids.map((v) => (
            <button key={v} onClick={() => setActiveVideo(v)}
              className={`rounded border px-2.5 py-1 ${activeVideo === v ? "border-accent bg-accent/15 text-accent" : "border-line bg-panel-2"}`}>
              {byId[v]?.original_filename ?? v}
              {byId[v] && <span className="ml-1 text-xs text-muted">{byId[v].meta.nominal_fps?.toFixed(0)} fps</span>}
            </button>
          ))}
          {isOpen && (
            <select className="rounded border border-line bg-panel-2 px-2 py-1" value="" onChange={(e) => e.target.value && act(() => api.addReviewVideo(id, e.target.value))}>
              <option value="">Attach uploaded video...</option>
              {videos.filter((v) => !review.video_ids.includes(v.id)).map((v) => (
                <option key={v.id} value={v.id}>{v.original_filename}</option>
              ))}
            </select>
          )}
          <Link href="/upload" className="text-xs text-accent hover:underline">Upload new</Link>
        </div>
      </Panel>

      {review.video_ids.length > 0 && (
        <Panel title="Synchronised replay" actions={
          <a className="text-xs text-accent hover:underline" href={`/reviews/${review.id}/monitor`} target="_blank" rel="noreferrer">Open umpire monitor ↗</a>
        }>
          <ReplayMonitor videos={review.video_ids.map((v) => byId[v]).filter(Boolean)} />
        </Panel>
      )}

      {video && (review.decision_type === "run_out" || review.decision_type === "stumping") && (
        <CreaseWorkspace key={video.id} video={video} mode={review.decision_type} readOnly={!isOpen} onAnalysis={(a) => attach("decision", a.id)} />
      )}
      {video && review.decision_type === "boundary" && (
        <BoundaryWorkspace key={video.id} video={video} readOnly={!isOpen} onAnalysis={(a) => attach("decision", a.id)} />
      )}
      {(review.decision_type === "lbw" || review.decision_type === "caught") && (
        <TrackingEvidence review={review} videos={byId} onAttach={attach} readOnly={!isOpen} />
      )}
      {video && (review.decision_type === "caught" || review.decision_type === "lbw") && (
        <AudioEdgePanel key={`a-${video.id}`} video={video} readOnly={!isOpen} onAnalysis={(a) => attach("decision", a.id)} />
      )}
      {review.decision_type === "lbw" && (
        <MultiviewLbwPanel review={review} videos={byId} onAttach={(aid) => attach("lbw", aid)} readOnly={!isOpen} />
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <EvidenceList review={review} />
        <RecommendationPanel review={review} onRecommend={() => act(() => api.recommend(id))} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <DecisionPanel review={review} match={match} onDecide={(b) => act(() => api.decide(id, b))} onWithdraw={(reason) => act(() => api.withdraw(id, reason))} />
        <Panel title="Audit log">
          <ol className="space-y-1.5 text-sm">
            {(review.events ?? []).map((e) => (
              <li key={e.id} className="flex gap-2">
                <span className="tabular shrink-0 text-xs text-muted">{new Date(e.created_at).toLocaleTimeString()}</span>
                <span><span className="text-ink">{e.action.replace("_", " ")}</span> <span className="text-muted">by {e.actor}</span></span>
              </li>
            ))}
          </ol>
        </Panel>
      </div>
    </div>
  );
}

function TrackingEvidence({ review, videos, onAttach, readOnly }: {
  review: Review; videos: Record<string, Video>; onAttach: (t: "lbw" | "decision", id: string) => void; readOnly: boolean;
}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [lbw, setLbw] = useState<Record<string, LbwAnalysis[]>>({});
  const [err, setErr] = useState<unknown>(null);

  const load = useCallback(
    () =>
      Promise.all(review.video_ids.map((v) => api.jobs(v)))
        .then(async (all) => {
          const js = all.flat();
          const entries = await Promise.all(js.filter((j) => j.status === "succeeded").map(async (j) => [j.id, await api.lbwList(j.id)] as const));
          setJobs(js);
          setLbw(Object.fromEntries(entries));
        })
        .catch(setErr),
    [review.video_ids],
  );
  useEffect(() => void load(), [load]);

  const start = async (videoId: string) => {
    try {
      await api.createJob({ video_id: videoId, detector: "auto" });
      await load();
    } catch (e) {
      setErr(e);
    }
  };

  return (
    <Panel title="Ball tracking and LBW evidence" actions={<Button variant="ghost" onClick={load}>Refresh</Button>}>
      <ErrorBox error={err} />
      {review.video_ids.length === 0 ? (
        <Empty>Attach a replay first.</Empty>
      ) : (
        <div className="space-y-3 text-sm">
          {review.video_ids.map((vid) => {
            const vj = jobs.filter((j) => j.video_id === vid);
            return (
              <div key={vid} className="rounded border border-line p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-medium">{videos[vid]?.original_filename ?? vid}</span>
                  {!readOnly && <Button onClick={() => start(vid)}>Run ball tracking</Button>}
                </div>
                {vj.length === 0 && <div className="text-muted">No tracking job yet.</div>}
                {vj.map((j) => (
                  <div key={j.id} className="border-t border-line py-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge tone={j.status === "succeeded" ? "good" : j.status === "failed" ? "bad" : "info"}>{j.status}</Badge>
                      <Link className="text-accent hover:underline" href={`/sessions/${j.id}`}>Open tracking / calibration / LBW workspace →</Link>
                      {j.summary?.primary_track && <span className="text-muted">{j.summary.primary_track.num_observed} observed points</span>}
                    </div>
                    {(lbw[j.id] ?? []).map((a) => (
                      <div key={a.id} className="mt-1.5 flex flex-wrap items-center gap-2 pl-2">
                        <span className="text-muted">LBW analysis {new Date(a.created_at).toLocaleTimeString()}</span>
                        <Badge tone={a.result.verdict === "INCONCLUSIVE" || a.result.verdict === "INSUFFICIENT_DATA" ? "warn" : "info"}>{a.result.verdict}</Badge>
                        {!readOnly && !review.evidence.some((e) => e.id === a.id) && <Button onClick={() => onAttach("lbw", a.id)}>Attach</Button>}
                        {review.evidence.some((e) => e.id === a.id) && <Badge tone="good">attached</Badge>}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

function EvidenceList({ review }: { review: Review }) {
  const [items, setItems] = useState<Record<string, DecisionAnalysis>>({});
  useEffect(() => {
    review.evidence.filter((e) => e.type === "decision" && !items[e.id]).forEach((e) =>
      api.decisionAnalysis(e.id).then((a) => setItems((m) => ({ ...m, [a.id]: a }))).catch(() => undefined));
  }, [review.evidence, items]);
  return (
    <Panel title={`Attached evidence (${review.evidence.length})`}>
      {review.evidence.length === 0 ? (
        <Empty>No analysis attached. The recommendation will defer to the third umpire.</Empty>
      ) : (
        <ul className="space-y-2 text-sm">
          {review.evidence.map((e) => {
            const a = items[e.id];
            return (
              <li key={e.id} className="rounded border border-line px-3 py-2">
                <div className="flex items-center gap-2">
                  <Badge tone="info">{e.kind}</Badge>
                  {a?.result.status && <Badge tone={/INCONCLUSIVE|INSUFFICIENT/.test(a.result.status) ? "warn" : "neutral"}>{a.result.status}</Badge>}
                  <span className="tabular text-xs text-muted">{e.id} · {new Date(e.attached_at).toLocaleTimeString()}</span>
                </div>
                {a?.result.reasons && (
                  <ul className="mt-1 list-disc pl-5 text-xs text-muted">{a.result.reasons.slice(0, 4).map((r, i) => <li key={i}>{r}</li>)}</ul>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}

function RecommendationPanel({ review, onRecommend }: { review: Review; onRecommend: () => void }) {
  const rec = review.recommendation;
  return (
    <Panel title="System recommendation (advisory)" actions={review.status === "open" && <Button variant="primary" onClick={onRecommend}>{rec ? "Recompute" : "Compute"}</Button>}>
      {!rec ? (
        <Empty>Not computed yet.</Empty>
      ) : (
        <div className="space-y-3 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={outcomeTone(rec.outcome)}>{rec.outcome.replace("_", " ")}</Badge>
            <span>suggested decision <span className="font-semibold">{rec.final_decision.replaceAll("_", " ")}</span></span>
            {rec.boundary_status && <Badge tone="info">{rec.boundary_status}</Badge>}
            {rec.review_retained !== null && rec.review_retained !== undefined && (
              <Badge tone={rec.review_retained ? "good" : "neutral"}>{rec.review_retained ? "review retained" : "review lost if upheld"}</Badge>
            )}
          </div>
          {rec.components && rec.components.length > 0 && (
            <table className="w-full text-xs">
              <tbody>
                {rec.components.map((c, i) => (
                  <tr key={i} className="border-t border-line">
                    <td className="py-1 pr-2 font-medium">{String(c.name ?? c.id ?? i)}</td>
                    <td className="pr-2">{c.status && <Badge tone={c.umpires_call ? "warn" : "neutral"}>{String(c.status)}</Badge>}</td>
                    <td className="text-muted">{c.evidence ? String(c.evidence) : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {rec.reasons.length > 0 && <ul className="list-disc pl-5 text-muted">{rec.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>}
          {rec.warnings.length > 0 && <ul className="list-disc pl-5 text-warn">{rec.warnings.map((r, i) => <li key={i}>{r}</li>)}</ul>}
          <div className="text-xs text-muted">
            Protocol {rec.protocol_id} · computed {new Date(rec.computed_at).toLocaleTimeString()}. Experimental output. The third umpire makes the decision.
          </div>
        </div>
      )}
    </Panel>
  );
}

function DecisionPanel({ review, match, onDecide, onWithdraw }: {
  review: Review; match: Match; onDecide: (b: Record<string, unknown>) => void; onWithdraw: (reason: string) => void;
}) {
  const boundary = review.decision_type === "boundary";
  const [final, setFinal] = useState<string>(review.on_field === "NONE" ? "NOT_OUT" : review.on_field);
  const [bres, setBres] = useState("NO_BOUNDARY");
  const [umpCall, setUmpCall] = useState(false);
  const [by, setBy] = useState(match.officials?.third_umpire ?? "");
  const [notes, setNotes] = useState("");

  if (review.status !== "open") {
    return (
      <Panel title="Decision">
        <KV rows={[
          ["Status", review.status],
          ["Final decision", review.final_decision?.replace("_", " ")],
          ["Outcome", review.outcome && <Badge tone={outcomeTone(review.outcome)}>{review.outcome.replace("_", " ")}</Badge>],
          ["Decided by", review.decided_by],
          ["At", review.decided_at && new Date(review.decided_at).toLocaleString()],
          ["Notes", review.notes],
        ]} />
      </Panel>
    );
  }
  const changed = review.on_field !== "NONE" && final !== review.on_field;
  const outcome = boundary || review.on_field === "NONE" ? "UPHELD" : changed ? "OVERTURNED" : umpCall ? "UMPIRES_CALL" : "UPHELD";
  return (
    <Panel title="Third umpire decision">
      <div className="space-y-3 text-sm">
        {boundary ? (
          <div className="flex gap-1">
            {["SIX", "FOUR", "NO_BOUNDARY"].map((b) => (
              <button key={b} onClick={() => setBres(b)} className={`rounded border px-3 py-1.5 ${bres === b ? "border-accent bg-accent/15 text-accent" : "border-line bg-panel-2"}`}>{b.replace("_", " ")}</button>
            ))}
          </div>
        ) : (
          <div className="flex gap-1">
            {["OUT", "NOT_OUT"].map((d) => (
              <button key={d} onClick={() => setFinal(d)} className={`rounded border px-4 py-2 text-base font-semibold ${final === d ? (d === "OUT" ? "border-bad bg-bad/15 text-bad" : "border-observed bg-observed/15 text-observed") : "border-line bg-panel-2 text-muted"}`}>
                {d.replace("_", " ")}
              </button>
            ))}
          </div>
        )}
        {review.decision_type === "lbw" && !changed && (
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={umpCall} onChange={(e) => setUmpCall(e.target.checked)} /> Stands on umpire&apos;s call (review retained)
          </label>
        )}
        <div className="text-muted">Outcome recorded: <Badge tone={outcomeTone(outcome)}>{outcome.replace("_", " ")}</Badge></div>
        <label className="block">Decided by<input className={input} value={by} onChange={(e) => setBy(e.target.value)} /></label>
        <label className="block">Notes<textarea className={input} rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} /></label>
        <div className="flex gap-2">
          <Button variant="primary" disabled={!by.trim()} onClick={() => onDecide({
            final_decision: boundary ? null : final, boundary_result: boundary ? bres : null, outcome, decided_by: by.trim(), notes: notes || null,
          })}>Record decision</Button>
          <Button variant="ghost" onClick={() => onWithdraw(notes || "Withdrawn by operator")}>Withdraw review</Button>
        </div>
      </div>
    </Panel>
  );
}
