import {
  Activity,
  AlertTriangle,
  History,
  Home,
  Settings,
  ShieldCheck,
  X,
} from "lucide-react";
import { NavLink } from "react-router-dom";
import { Button } from "../common/Button";
import { LogoMark } from "../common/LogoMark";

const navItems = [
  { to: "/dashboard", label: "Command", description: "Operational overview", icon: Home },
  { to: "/verify", label: "Verification", description: "Media intake", icon: ShieldCheck },
  { to: "/history", label: "Cases", description: "Review queue", icon: History },
  { to: "/health", label: "System Health", description: "Service readiness", icon: Activity },
  { to: "/settings", label: "Settings", description: "Environment", icon: Settings },
];

interface AppSidebarProps {
  open: boolean;
  onClose: () => void;
}

function SidebarContent({ onClose }: { onClose?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-white/10 px-5 py-5">
        <div className="flex items-center justify-between gap-3">
          <LogoMark />
          {onClose ? (
            <Button
              variant="ghost"
              className="h-9 w-9 px-0 lg:hidden"
              onClick={onClose}
              aria-label="Close navigation"
            >
              <X size={18} />
            </Button>
          ) : null}
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            onClick={onClose}
            className={({ isActive }) =>
              `group flex min-h-14 items-center gap-3 rounded-md border px-3 py-2.5 text-sm transition ${
                isActive
                  ? "border-signal-info/25 bg-signal-info/10 text-white"
                  : "border-transparent text-slate-400 hover:border-white/10 hover:bg-white/[0.04] hover:text-slate-100"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={`grid h-9 w-9 shrink-0 place-items-center rounded-md border ${
                    isActive
                      ? "border-signal-info/30 bg-signal-info/15 text-signal-info"
                      : "border-white/10 bg-surface-850 text-slate-500 group-hover:text-slate-200"
                  }`}
                >
                  <item.icon size={18} />
                </span>
                <span className="min-w-0">
                  <span className="block font-semibold">{item.label}</span>
                  <span className="block truncate text-xs text-slate-500">{item.description}</span>
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="m-3 rounded-md border border-white/10 bg-surface-850 p-4 shadow-insetline">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
          <AlertTriangle size={16} className="text-signal-review" />
          Manual Review
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-500">
          Suspicious and ambiguous cases stay visible for analyst escalation.
        </p>
      </div>
    </div>
  );
}

export function AppSidebar({ open, onClose }: AppSidebarProps) {
  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-72 border-r border-white/10 bg-surface-925 lg:block">
        <SidebarContent />
      </aside>

      <div className={`fixed inset-0 z-40 lg:hidden ${open ? "block" : "hidden"}`}>
        <button
          type="button"
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          aria-label="Close navigation overlay"
          onClick={onClose}
        />
        <aside className="relative h-full w-[min(20rem,88vw)] border-r border-white/10 bg-surface-925 shadow-panel">
          <SidebarContent onClose={onClose} />
        </aside>
      </div>
    </>
  );
}
