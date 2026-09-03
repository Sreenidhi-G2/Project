import type { Verdict } from "../../types/api";

interface StatusBadgeProps {
  status: Verdict;
}

const styles: Record<string, string> = {
  REAL: "border-signal-real/30 bg-signal-real/10 text-signal-real",
  FAKE: "border-signal-fake/30 bg-signal-fake/10 text-signal-fake",
  REVIEW: "border-signal-review/30 bg-signal-review/10 text-signal-review",
  OK: "border-signal-real/30 bg-signal-real/10 text-signal-real",
  OFFLINE: "border-signal-fake/30 bg-signal-fake/10 text-signal-fake",
  READY: "border-signal-info/30 bg-signal-info/10 text-signal-info",
  QUEUED: "border-white/10 bg-white/5 text-slate-300",
};

export function StatusBadge({ status }: StatusBadgeProps) {
  const normalized = String(status).toUpperCase();

  return (
    <span
      className={`inline-flex h-7 items-center rounded border px-2.5 text-xs font-semibold ${
        styles[normalized] ?? "border-white/10 bg-white/5 text-zinc-300"
      }`}
    >
      {normalized}
    </span>
  );
}
