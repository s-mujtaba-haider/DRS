"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { Button } from "@/components/ui";
import { urls } from "@/lib/api";

export interface FrameInfo {
  index: number;
  timestampMs: number | null;
  timestampSource: string | null;
}

/** Exact decoded-frame viewer (server-side decode, so frame indices are authoritative).
 *  `overlay` is rendered in an SVG whose coordinates are image pixels. */
export default function FrameViewer({
  videoId,
  width,
  height,
  frameCount,
  frame,
  setFrame,
  onImageClick,
  overlay,
  onFrameInfo,
  crosshair = false,
}: {
  videoId: string;
  width: number;
  height: number;
  frameCount: number | null;
  frame: number;
  setFrame: (f: number) => void;
  onImageClick?: (x: number, y: number) => void;
  overlay?: ReactNode;
  onFrameInfo?: (info: FrameInfo) => void;
  crosshair?: boolean;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [info, setInfo] = useState<FrameInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const svgRef = useRef<SVGSVGElement>(null);
  const last = useRef<string | null>(null);
  const max = frameCount ? frameCount - 1 : null;

  useEffect(() => {
    let cancelled = false;
    const ctrl = new AbortController();
    fetch(urls.frame(videoId, frame), { signal: ctrl.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error(r.status === 404 ? "Frame beyond end of video" : `HTTP ${r.status}`);
        const ts = r.headers.get("X-Frame-Timestamp-Ms");
        const blob = await r.blob();
        if (cancelled) return;
        const u = URL.createObjectURL(blob);
        if (last.current) URL.revokeObjectURL(last.current);
        last.current = u;
        const fi = { index: frame, timestampMs: ts ? Number(ts) : null, timestampSource: r.headers.get("X-Timestamp-Source") };
        setSrc(u);
        setInfo(fi);
        setErr(null);
        onFrameInfo?.(fi);
      })
      .catch((e) => !cancelled && e.name !== "AbortError" && setErr(String(e.message ?? e)));
    return () => {
      cancelled = true;
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoId, frame]);

  useEffect(() => () => void (last.current && URL.revokeObjectURL(last.current)), []);

  const step = useCallback(
    (d: number) => {
      const n = Math.max(0, max === null ? frame + d : Math.min(max, frame + d));
      setFrame(n);
    },
    [frame, max, setFrame],
  );

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest("input,textarea,select")) return;
      if (e.key === "ArrowRight") step(e.shiftKey ? 10 : 1);
      else if (e.key === "ArrowLeft") step(e.shiftKey ? -10 : -1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [step]);

  const click = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!onImageClick || !svgRef.current) return;
    const pt = svgRef.current.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = svgRef.current.getScreenCTM();
    if (!m) return;
    const p = pt.matrixTransform(m.inverse());
    onImageClick(Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10);
  };

  return (
    <div className="space-y-2">
      <div className="overflow-auto rounded border border-line bg-black" style={{ maxHeight: "70vh" }}>
        <div className="relative" style={{ width: `${zoom * 100}%`, aspectRatio: `${width} / ${height}` }}>
          {src && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={src} alt={`frame ${frame}`} className="absolute inset-0 h-full w-full select-none" draggable={false} />
          )}
          <svg
            ref={svgRef}
            viewBox={`0 0 ${width} ${height}`}
            className={`absolute inset-0 h-full w-full ${onImageClick ? (crosshair ? "cursor-crosshair" : "cursor-pointer") : ""}`}
            onClick={click}
          >
            {overlay}
          </svg>
          {err && <div className="absolute inset-x-0 top-0 bg-bad/80 px-2 py-1 text-xs">{err}</div>}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Button onClick={() => step(-10)} title="Shift+Left">-10</Button>
        <Button onClick={() => step(-1)} title="Left arrow">-1</Button>
        <Button onClick={() => step(1)} title="Right arrow">+1</Button>
        <Button onClick={() => step(10)} title="Shift+Right">+10</Button>
        <input
          type="range"
          min={0}
          max={max ?? Math.max(frame + 100, 100)}
          value={frame}
          onChange={(e) => setFrame(Number(e.target.value))}
          className="min-w-[160px] flex-1"
          aria-label="Frame"
        />
        <span className="tabular text-muted">
          frame <span className="text-ink">{frame}</span>
          {max !== null && ` / ${max}`} · t ={" "}
          <span className="text-ink">{info?.timestampMs != null ? `${info.timestampMs.toFixed(1)} ms` : "n/a"}</span>
          {info?.timestampSource && info.timestampSource !== "container_pts" && <span className="text-warn"> (estimated)</span>}
        </span>
        <select value={zoom} onChange={(e) => setZoom(Number(e.target.value))} className="rounded border border-line bg-panel-2 px-2 py-1" aria-label="Zoom">
          {[1, 1.5, 2, 3, 4].map((z) => (
            <option key={z} value={z}>{z}x</option>
          ))}
        </select>
      </div>
    </div>
  );
}
