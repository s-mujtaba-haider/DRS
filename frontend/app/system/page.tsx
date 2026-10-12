"use client";

import { useEffect, useState } from "react";

import { Badge, Empty, ErrorBox, KV, Panel } from "@/components/ui";
import { api, urls, type SystemStatus } from "@/lib/api";

export default function SystemPage() {
  const [sys, setSys] = useState<SystemStatus | null>(null);
  const [reports, setReports] = useState<{ file: string; summary: Record<string, unknown> | null }[]>([]);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    api.system().then(setSys).catch(setError);
    api.evaluationReports().then(setReports).catch(setError);
  }, []);

  return (
    <div className="max-w-5xl space-y-5">
      <h1 className="text-xl font-semibold">System</h1>
      <ErrorBox error={error} />
      <Panel title="Runtime" actions={<a className="text-xs text-accent hover:underline" href={urls.docs()} target="_blank" rel="noreferrer">API docs</a>}>
        {sys ? (
          <KV
            rows={[
              ["Platform", sys.platform],
              ["Python", sys.python],
              ["PyTorch", sys.torch.available ? `${sys.torch.version} (CUDA ${sys.torch.cuda_available ? "available" : "not available"})` : <span className="text-warn">{sys.torch.error}</span>],
              ["GPU", sys.gpu ? `${sys.gpu.name}, driver ${sys.gpu.driver}, ${sys.gpu.memory_used_mb}/${sys.gpu.memory_total_mb} MB used, ${sys.gpu.utilization_pct}% util` : "not detected"],
              ["Detector mode", <Badge key="d" tone={sys.cricket_detector_available ? "good" : "warn"}>{sys.effective_detector_mode}</Badge>],
            ]}
          />
        ) : (
          <Empty>Loading...</Empty>
        )}
      </Panel>
      <Panel title="Model checkpoints">
        {sys?.models.length ? (
          <ul className="space-y-1 text-sm">
            {sys.models.map((m) => (
              <li key={m.file}>{m.file} <span className="text-muted">({m.size_mb} MB)</span></li>
            ))}
          </ul>
        ) : (
          <Empty>No checkpoints in models/checkpoints. See docs/MODEL_CARD.md for training and download steps.</Empty>
        )}
      </Panel>
      <Panel title="Evaluation reports (outputs/reports)">
        {reports.length ? (
          <ul className="space-y-3 text-xs">
            {reports.map((r) => (
              <li key={r.file}>
                <div className="font-medium text-ink">{r.file}</div>
                <pre className="mt-1 max-h-48 overflow-auto rounded bg-panel-2 p-2 text-muted">{JSON.stringify(r.summary, null, 1)}</pre>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No evaluation reports yet.</Empty>
        )}
      </Panel>
    </div>
  );
}
