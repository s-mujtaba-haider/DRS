"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Badge, Button, Empty, ErrorBox, Panel, type Tone } from "@/components/ui";
import { api, type DecisionType, type Match, type Review } from "@/lib/api";

const input = "w-full rounded border border-line bg-panel-2 px-2 py-1.5 text-sm";
export const DECISION_LABEL: Record<DecisionType, string> = {
  lbw: "LBW",
  caught: "Caught",
  run_out: "Run out",
  stumping: "Stumped",
  boundary: "Boundary",
  other: "Other",
};
export function outcomeTone(o: string | null | undefined): Tone {
  return ({ OVERTURNED: "info", UPHELD: "neutral", UMPIRES_CALL: "warn", INCONCLUSIVE: "warn" } as Record<string, Tone>)[o ?? ""] ?? "neutral";
}

export default function MatchControl({ id }: { id: string }) {
  const router = useRouter();
  const [match, setMatch] = useState<Match | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [f, setF] = useState({ requested_by: "", decision_type: "lbw" as DecisionType, on_field: "NOT_OUT", over_ball: "", batter: "", bowler: "" });

  const load = useCallback(() => {
    Promise.all([api.match(id), api.reviews(id)])
      .then(([m, r]) => {
        setMatch(m);
        setReviews(r);
      })
      .catch(setError);
  }, [id]);
  useEffect(load, [load]);

  if (!match) return error ? <ErrorBox error={error} /> : <Empty>Loading match...</Empty>;
  const fielding = match.batting_team === match.team_a ? match.team_b : match.team_a;
  const umpireRef = f.requested_by === "umpire";

  const patch = async (body: Record<string, unknown>) => {
    try {
      setMatch(await api.updateMatch(id, body));
      setError(null);
    } catch (e) {
      setError(e);
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const r = await api.createReview(id, {
        requested_by: f.requested_by,
        decision_type: f.decision_type,
        on_field: f.on_field,
        over_ball: f.over_ball || null,
        batter: f.batter || null,
        bowler: f.bowler || null,
      });
      router.push(`/reviews/${r.id}`);
    } catch (err) {
      setError(err);
      setBusy(false);
    }
  };

  const open = reviews.filter((r) => r.status === "open");

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link href="/matches" className="text-xs text-muted hover:text-ink">← Matches</Link>
          <h1 className="text-xl font-semibold">{match.name}</h1>
          <div className="text-sm text-muted">
            {match.team_a} v {match.team_b} · {match.venue ?? "venue n/a"} · protocol <span className="text-ink">{match.protocol_id}</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <Badge tone={match.status === "live" ? "good" : "neutral"}>{match.status}</Badge>
          <label className="flex items-center gap-1">
            Innings
            <select className="rounded border border-line bg-panel-2 px-2 py-1" value={match.current_innings} onChange={(e) => patch({ current_innings: Number(e.target.value) })}>
              {[1, 2, 3, 4].map((i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-1">
            Batting
            <select className="rounded border border-line bg-panel-2 px-2 py-1" value={match.batting_team ?? ""} onChange={(e) => patch({ batting_team: e.target.value })}>
              {[match.team_a, match.team_b].map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          {match.status === "live" ? (
            <Button variant="ghost" onClick={() => patch({ status: "completed" })}>End match</Button>
          ) : (
            <Button variant="ghost" onClick={() => patch({ status: "live" })}>Reopen</Button>
          )}
        </div>
      </div>
      <ErrorBox error={error} />

      <div className="grid gap-3 sm:grid-cols-2">
        {[match.team_a, match.team_b].map((t) => {
          const c = match.reviews.teams[t];
          return (
            <div key={t} className="rounded-lg border border-line bg-panel px-4 py-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">{t}</div>
                <div className="text-xs text-muted">{t === match.batting_team ? "batting" : "fielding"}</div>
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="tabular text-3xl font-semibold">{c?.remaining ?? "∞"}</span>
                <span className="text-sm text-muted">reviews left this innings</span>
              </div>
              <div className="text-xs text-muted">
                {c?.unsuccessful_used ?? 0} unsuccessful · {c?.retained_umpires_call ?? 0} retained on umpire&apos;s call
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid gap-5 lg:grid-cols-[1fr_380px]">
        <Panel title={`Reviews (${reviews.length})`}>
          {reviews.length === 0 ? (
            <Empty>No reviews yet.</Empty>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead className="text-left text-xs uppercase tracking-wider text-muted">
                  <tr>
                    <th className="pb-2">#</th><th className="pb-2">Inn</th><th className="pb-2">Over</th><th className="pb-2">Type</th>
                    <th className="pb-2">By</th><th className="pb-2">On field</th><th className="pb-2">Final</th><th className="pb-2">Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {reviews.map((r) => (
                    <tr key={r.id} className="border-t border-line">
                      <td className="py-2"><Link className="text-accent hover:underline" href={`/reviews/${r.id}`}>{r.seq}</Link></td>
                      <td>{r.innings}</td>
                      <td className="tabular">{r.over_ball ?? "-"}</td>
                      <td>{DECISION_LABEL[r.decision_type]}</td>
                      <td>{r.kind === "umpire" ? <Badge tone="info">umpire</Badge> : r.requested_by}</td>
                      <td>{r.on_field.replace("_", " ")}</td>
                      <td>{r.status === "open" ? <Badge tone="warn">open</Badge> : r.status === "withdrawn" ? <Badge>withdrawn</Badge> : r.final_decision?.replace("_", " ")}</td>
                      <td>{r.outcome ? <Badge tone={outcomeTone(r.outcome)}>{r.outcome.replace("_", " ")}</Badge> : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <Panel title="New review">
          {match.status !== "live" ? (
            <Empty>Match is {match.status}.</Empty>
          ) : (
            <form onSubmit={submit} className="space-y-3 text-sm">
              {open.length > 0 && (
                <div className="rounded border border-warn/40 bg-warn/10 px-2 py-1.5 text-xs text-warn">
                  {open.length} review(s) still open: {open.map((r) => <Link key={r.id} className="underline" href={`/reviews/${r.id}`}>#{r.seq} </Link>)}
                </div>
              )}
              <label className="block">
                Requested by
                <select required className={input} value={f.requested_by} onChange={(e) => setF({ ...f, requested_by: e.target.value, on_field: e.target.value === "umpire" ? "NONE" : f.on_field === "NONE" ? "NOT_OUT" : f.on_field })}>
                  <option value="" disabled>Select...</option>
                  <option value={fielding}>{fielding} (fielding)</option>
                  {match.batting_team && <option value={match.batting_team}>{match.batting_team} (batting)</option>}
                  <option value="umpire">On-field umpire referral</option>
                </select>
              </label>
              <div className="flex flex-wrap gap-1">
                {(Object.keys(DECISION_LABEL) as DecisionType[]).filter((d) => umpireRef || d !== "boundary").map((d) => (
                  <button type="button" key={d} onClick={() => setF({ ...f, decision_type: d })}
                    className={`rounded border px-2.5 py-1 ${f.decision_type === d ? "border-accent bg-accent/15 text-accent" : "border-line bg-panel-2 text-muted"}`}>
                    {DECISION_LABEL[d]}
                  </button>
                ))}
              </div>
              <label className="block">
                On-field decision
                <select className={input} value={f.on_field} onChange={(e) => setF({ ...f, on_field: e.target.value })}>
                  <option value="OUT">Out</option>
                  <option value="NOT_OUT">Not out</option>
                  {umpireRef && <option value="NONE">No decision / referral</option>}
                </select>
              </label>
              <div className="grid grid-cols-3 gap-2">
                <label className="block">Over<input className={input} placeholder="12.3" pattern="\d{1,3}\.[0-9]" value={f.over_ball} onChange={(e) => setF({ ...f, over_ball: e.target.value })} /></label>
                <label className="block">Batter<input className={input} value={f.batter} onChange={(e) => setF({ ...f, batter: e.target.value })} /></label>
                <label className="block">Bowler<input className={input} value={f.bowler} onChange={(e) => setF({ ...f, bowler: e.target.value })} /></label>
              </div>
              <Button type="submit" variant="primary" disabled={busy || !f.requested_by}>{busy ? "Opening..." : "Open review"}</Button>
            </form>
          )}
        </Panel>
      </div>
    </div>
  );
}
