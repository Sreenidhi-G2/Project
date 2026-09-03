import { useEffect, useState } from "react";
import { API_BASE_URL } from "../api/client";
import { verifaceApi } from "../api/verifaceApi";
import { MetricCard } from "../components/common/MetricCard";
import { StatusBadge } from "../components/common/StatusBadge";
import type { HealthResponse } from "../types/api";

export default function Health() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    verifaceApi
      .getHealth()
      .then(setHealth)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Could not reach the API.");
      });
  }, []);

  return (
    <div className="space-y-5">
      <section className="panel p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="muted-label">Backend health</div>
            <h2 className="mt-2 text-lg font-semibold text-white">{API_BASE_URL}</h2>
          </div>
          <StatusBadge status={health?.status === "ok" ? "OK" : error ? "OFFLINE" : "CHECKING"} />
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <MetricCard label="API status" value={health?.status ?? "Checking"} detail={error ?? undefined} />
        <MetricCard
          label="Models loaded"
          value={health ? (health.models_loaded ? "Yes" : "No") : "Checking"}
          detail="Reported by GET /health"
        />
      </div>
    </div>
  );
}
