"use client";

import { useEffect, useState } from "react";

import { Badge, Empty, ErrorBox, Panel, type Tone } from "@/components/ui";
import { api, type DatasetEntry } from "@/lib/api";

const STATUS_TONE: Record<string, Tone> = { downloaded: "good", blocked_credentials: "warn", download_failed: "bad", unverified: "warn" };

export default function DatasetsPage() {
  const [data, setData] = useState<{ datasets: DatasetEntry[]; notes: string; last_reviewed: string } | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => void api.datasets().then(setData).catch(setError), []);

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold">Dataset registry</h1>
      <ErrorBox error={error} />
      {data && <p className="max-w-4xl text-sm text-muted">{data.notes} Last reviewed {data.last_reviewed}.</p>}
      <Panel>
        {!data ? (
          <Empty>Loading...</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[960px] text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="pb-2">Dataset</th>
                  <th className="pb-2">Status</th>
                  <th className="pb-2">License</th>
                  <th className="pb-2">Ball boxes</th>
                  <th className="pb-2">3D / traj.</th>
                  <th className="pb-2">Multi-view</th>
                  <th className="pb-2">Intended use</th>
                </tr>
              </thead>
              <tbody>
                {data.datasets.map((d) => (
                  <tr key={d.id} className="border-t border-line align-top">
                    <td className="py-2 pr-3">
                      <a href={d.url} target="_blank" rel="noreferrer" className="text-accent hover:underline">{d.name}</a>
                      <div className="text-xs text-muted">{d.source} - {d.size} - {d.verification}</div>
                    </td>
                    <td className="pr-3">
                      <Badge tone={STATUS_TONE[d.status] ?? "neutral"}>{d.status}</Badge>
                      {d.download_note && <div className="mt-1 text-xs text-muted">{d.download_note}</div>}
                      {d.last_error && <div className="mt-1 max-w-[220px] text-xs text-warn">{d.last_error}</div>}
                    </td>
                    <td className="pr-3 text-xs">{d.license}<div className="text-muted">commercial: {d.commercial_use}</div></td>
                    <td className="pr-3">{String(d.has_ball_annotations)}</td>
                    <td className="pr-3">{String(d.has_trajectories_or_3d)}</td>
                    <td className="pr-3">{String(d.multi_view)}</td>
                    <td className="text-xs text-muted">{d.intended_use}</td>
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
