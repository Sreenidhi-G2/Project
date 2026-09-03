interface MetricCardProps {
  label: string;
  value: string;
  detail?: string;
}

export function MetricCard({ label, value, detail }: MetricCardProps) {
  return (
    <div className="rounded-md border border-white/10 bg-surface-850 p-4 shadow-insetline">
      <div className="muted-label">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      {detail ? <div className="mt-1 text-sm text-slate-400">{detail}</div> : null}
    </div>
  );
}
