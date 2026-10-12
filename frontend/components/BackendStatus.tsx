"use client";

import { useEffect, useState } from "react";

import { API_BASE, api, type SystemStatus } from "@/lib/api";

export default function BackendStatus() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .system()
        .then((s) => alive && (setStatus(s), setDown(false)))
        .catch(() => alive && setDown(true));
    load();
    const t = setInterval(load, 15000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const dot = (ok: boolean | undefined) => <span className={`inline-block h-2 w-2 rounded-full ${ok ? "bg-observed" : "bg-bad"}`} />;
  return (
    <div className="space-y-1.5 text-[11px] text-muted">
      <div className="flex items-center gap-2">{dot(!down)} Backend {down ? "unreachable" : "online"}</div>
      {down && <div className="break-all text-bad">{API_BASE}</div>}
      {status && (
        <>
          <div className="flex items-center gap-2">
            {dot(!!status.gpu)} GPU {status.gpu ? status.gpu.name.replace("NVIDIA GeForce ", "") : "not detected"}
          </div>
          <div className="flex items-center gap-2">
            {dot(status.torch.cuda_available)} PyTorch {status.torch.available ? (status.torch.cuda_available ? "CUDA" : "CPU only") : "unavailable"}
          </div>
          <div className="flex items-center gap-2">
            {dot(status.cricket_detector_available)} Cricket detector {status.cricket_detector_available ? "loaded" : "not trained"}
          </div>
        </>
      )}
    </div>
  );
}
