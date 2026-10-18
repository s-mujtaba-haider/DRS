"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Badge, Button, Empty, ErrorBox, Panel } from "@/components/ui";
import { api, type Match, type ProtocolSummary } from "@/lib/api";

const input = "w-full rounded border border-line bg-panel-2 px-2 py-1.5 text-sm";

export default function MatchesPage() {
  const router = useRouter();
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [protocols, setProtocols] = useState<ProtocolSummary[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({ name: "", team_a: "", team_b: "", protocol_id: "", venue: "", match_date: "", third_umpire: "" });

  useEffect(() => {
    Promise.all([api.matches(), api.protocols()])
      .then(([m, p]) => {
        setMatches(m);
        setProtocols(p);
        setForm((f) => ({ ...f, protocol_id: f.protocol_id || p[0]?.id || "" }));
      })
      .catch(setError);
  }, []);

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setForm({ ...form, [k]: e.target.value });

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const m = await api.createMatch({
        name: form.name,
        team_a: form.team_a,
        team_b: form.team_b,
        protocol_id: form.protocol_id,
        venue: form.venue || null,
        match_date: form.match_date || null,
        batting_team: form.team_a,
        officials: form.third_umpire ? { third_umpire: form.third_umpire } : {},
      });
      router.push(`/matches/${m.id}`);
    } catch (err) {
      setError(err);
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold">Matches</h1>
      <ErrorBox error={error} />
      <div className="grid gap-5 lg:grid-cols-[1fr_380px]">
        <Panel title="Fixtures">
          {!matches ? (
            <Empty>Loading...</Empty>
          ) : matches.length === 0 ? (
            <Empty>No matches yet. Create one to start taking reviews.</Empty>
          ) : (
            <ul className="divide-y divide-line">
              {matches.map((m) => (
                <li key={m.id} className="flex flex-wrap items-center justify-between gap-2 py-2.5">
                  <div>
                    <Link href={`/matches/${m.id}`} className="font-medium text-accent hover:underline">{m.name}</Link>
                    <div className="text-xs text-muted">
                      {m.team_a} v {m.team_b} · {m.venue ?? "venue n/a"} · {m.match_date ?? ""} · {m.protocol_id}
                    </div>
                  </div>
                  <Badge tone={m.status === "live" ? "good" : "neutral"}>{m.status}</Badge>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="New match">
          <form onSubmit={create} className="space-y-3 text-sm">
            <label className="block">Match name<input required className={input} value={form.name} onChange={set("name")} placeholder="1st T20 · Club A v Club B" /></label>
            <div className="grid grid-cols-2 gap-2">
              <label className="block">Team A (bats first)<input required className={input} value={form.team_a} onChange={set("team_a")} /></label>
              <label className="block">Team B<input required className={input} value={form.team_b} onChange={set("team_b")} /></label>
            </div>
            <label className="block">
              DRS protocol
              <select required className={input} value={form.protocol_id} onChange={set("protocol_id")}>
                {protocols.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.version})</option>
                ))}
              </select>
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="block">Venue<input className={input} value={form.venue} onChange={set("venue")} /></label>
              <label className="block">Date<input type="date" className={input} value={form.match_date} onChange={set("match_date")} /></label>
            </div>
            <label className="block">Third umpire<input className={input} value={form.third_umpire} onChange={set("third_umpire")} /></label>
            <Button type="submit" variant="primary" disabled={busy}>{busy ? "Creating..." : "Create match"}</Button>
          </form>
        </Panel>
      </div>
    </div>
  );
}
