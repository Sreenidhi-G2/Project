import { Menu, ShieldCheck } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";

const mobileNav = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/verify", label: "Verify" },
  { to: "/history", label: "History" },
  { to: "/health", label: "Health" },
];

function pageTitle(pathname: string) {
  const segment = pathname.split("/").filter(Boolean)[0] ?? "dashboard";
  return segment.charAt(0).toUpperCase() + segment.slice(1);
}

export function AppHeader() {
  const location = useLocation();

  return (
    <header className="sticky top-0 z-20 border-b border-white/10 bg-surface-950/95 backdrop-blur">
      <div className="flex min-h-16 items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-signal-info/15 text-signal-info lg:hidden">
            <ShieldCheck size={19} />
          </div>
          <div className="min-w-0">
            <div className="text-xs font-medium uppercase tracking-normal text-zinc-500">
              VeriFace
            </div>
            <h1 className="truncate text-lg font-semibold text-white">
              {pageTitle(location.pathname)}
            </h1>
          </div>
        </div>

        <button
          className="grid h-10 w-10 place-items-center rounded-lg border border-white/10 bg-white/5 text-zinc-300 lg:hidden"
          type="button"
          aria-label="Navigation"
        >
          <Menu size={18} />
        </button>

        <div className="hidden items-center gap-2 text-sm text-zinc-400 lg:flex">
          <span className="h-2 w-2 rounded-full bg-signal-info" />
          API-driven verification
        </div>
      </div>

      <nav className="flex gap-1 overflow-x-auto border-t border-white/10 px-4 py-2 lg:hidden">
        {mobileNav.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `shrink-0 rounded-lg px-3 py-2 text-sm font-medium ${
                isActive ? "bg-white/10 text-white" : "text-zinc-400"
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
