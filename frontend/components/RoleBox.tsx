"use client";

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui";
import { api, setPin } from "@/lib/api";

/** Match-day role: enter the operator or umpire PIN when the backend has PINs configured. */
export default function RoleBox() {
  const [st, setSt] = useState<{ enabled: boolean; role: string } | null>(null);
  const [pin, setPinText] = useState("");
  const [bad, setBad] = useState(false);

  const refresh = () => api.authStatus().then(setSt).catch(() => setSt(null));
  useEffect(() => {
    void refresh();
  }, []);

  if (!st || !st.enabled) return null;
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPin(pin.trim() || null);
    const s = await api.authStatus().catch(() => null);
    setBad(!!pin && s?.role === "viewer");
    if (s?.role === "viewer") setPin(null);
    setSt(s);
    setPinText("");
  };
  return (
    <div className="mt-3 space-y-1 text-xs">
      <div className="flex items-center gap-1 text-muted">
        Role <Badge tone={st.role === "umpire" ? "good" : st.role === "operator" ? "info" : "neutral"}>{st.role}</Badge>
        {st.role !== "viewer" && (
          <button className="ml-auto text-muted hover:text-ink" onClick={() => { setPin(null); void refresh(); }}>sign out</button>
        )}
      </div>
      {st.role === "viewer" && (
        <form onSubmit={submit} className="flex gap-1">
          <input type="password" autoComplete="off" placeholder="PIN" value={pin} onChange={(e) => setPinText(e.target.value)}
            className="w-full rounded border border-line bg-panel-2 px-1.5 py-1" aria-label="Operator or umpire PIN" />
          <button type="submit" className="rounded border border-line bg-panel-2 px-2">OK</button>
        </form>
      )}
      {bad && <div className="text-bad">PIN not recognised</div>}
    </div>
  );
}
