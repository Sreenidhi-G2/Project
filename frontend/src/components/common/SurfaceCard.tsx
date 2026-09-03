import type { ReactNode } from "react";

interface SurfaceCardProps {
  title?: string;
  label?: string;
  children: ReactNode;
  className?: string;
}

export function SurfaceCard({ title, label, children, className = "" }: SurfaceCardProps) {
  return (
    <section className={`panel p-5 ${className}`}>
      {label || title ? (
        <div className="mb-4">
          {label ? <div className="muted-label">{label}</div> : null}
          {title ? <h2 className="mt-1 text-base font-semibold text-white">{title}</h2> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}
