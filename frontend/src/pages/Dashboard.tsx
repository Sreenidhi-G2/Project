import { Link } from "react-router-dom";
import { MetricCard } from "../components/common/MetricCard";

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <section className="panel p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="muted-label">VeriFace console</div>
            <h2 className="mt-2 text-2xl font-semibold text-white">Identity media verification</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-400">
              Submit liveness videos to the FastAPI backend and review model plus
              supervisor results from one place.
            </p>
          </div>
          <Link
            to="/verify"
            className="inline-flex h-11 items-center justify-center rounded-lg bg-white px-4 text-sm font-semibold text-surface-950 transition hover:bg-zinc-200"
          >
            Start verification
          </Link>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
      
        <MetricCard label="Video endpoint" value="/predict/video" detail="Multipart upload" />
        <MetricCard label="History endpoint" value="/history" detail="SQLite audit trail" />
      </section>
    </div>
  );
}
