"use client";

import type { ReactNode } from "react";

export function Panel({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-panel ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">{title}</h2>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-line bg-panel px-4 py-3">
      <div className="text-xs uppercase tracking-wider text-muted">{label}</div>
      <div className="tabular mt-1 text-2xl font-semibold">{value}</div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </div>
  );
}

const TONES = {
  neutral: "border-line bg-panel-2 text-muted",
  good: "border-observed/40 bg-observed/10 text-observed",
  warn: "border-warn/40 bg-warn/10 text-warn",
  bad: "border-bad/40 bg-bad/10 text-bad",
  info: "border-accent/40 bg-accent/10 text-accent",
  predicted: "border-predicted/40 bg-predicted/10 text-predicted",
};
export type Tone = keyof typeof TONES;

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] font-medium ${TONES[tone]}`}>{children}</span>;
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "default",
  type = "button",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary" | "danger" | "ghost";
  type?: "button" | "submit";
  title?: string;
}) {
  const styles = {
    default: "border-line bg-panel-2 hover:bg-line text-ink",
    primary: "border-accent/60 bg-accent/15 hover:bg-accent/25 text-accent",
    danger: "border-bad/50 bg-bad/10 hover:bg-bad/20 text-bad",
    ghost: "border-transparent hover:bg-panel-2 text-muted",
  }[variant];
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`rounded border px-3 py-1.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="rounded border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">{msg}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded border border-dashed border-line px-4 py-6 text-center text-sm text-muted">{children}</div>;
}

export function KV({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(120px,auto)_1fr] gap-x-4 gap-y-1.5 text-sm">
      {rows.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="tabular break-words">{v ?? <span className="text-muted">n/a</span>}</dd>
        </div>
      ))}
    </dl>
  );
}

export function statusTone(status: string): Tone {
  return ({ succeeded: "good", running: "info", queued: "neutral", failed: "bad", cancelled: "warn", interrupted: "warn" } as Record<string, Tone>)[status] ?? "neutral";
}

export function fmt(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(digits);
}

export function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded bg-panel-2" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className="h-full bg-accent transition-[width]" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}
