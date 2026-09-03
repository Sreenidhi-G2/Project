import type { ReactNode } from "react";

interface EmptyStateProps {
  title: string;
  children: ReactNode;
}

export function EmptyState({ title, children }: EmptyStateProps) {
  return (
    <div className="rounded-md border border-dashed border-white/10 bg-white/[0.03] p-8 text-center">
      <h2 className="text-base font-semibold text-white">{title}</h2>
      <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-slate-400">{children}</p>
    </div>
  );
}
