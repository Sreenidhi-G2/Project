import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  children: ReactNode;
}

const variants = {
  primary:
    "border-signal-info/40 bg-signal-info text-surface-980 shadow-[0_0_24px_rgba(86,183,255,0.18)] hover:bg-signal-cyan",
  secondary: "border-white/10 bg-white/[0.06] text-slate-100 hover:bg-white/[0.09]",
  ghost: "border-transparent bg-transparent text-slate-400 hover:bg-white/[0.05] hover:text-slate-100",
};

export function Button({ variant = "secondary", className = "", children, ...props }: ButtonProps) {
  return (
    <button
      className={`inline-flex h-10 items-center justify-center gap-2 rounded-md border px-3.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
