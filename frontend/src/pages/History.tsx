import { useEffect, useState } from "react";
import { verifaceApi } from "../api/verifaceApi";
import { EmptyState } from "../components/common/EmptyState";
import { HistoryTable } from "../components/history/HistoryTable";
import type { HistoryRecord } from "../types/api";

export default function History() {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    verifaceApi
      .getHistory()
      .then(setRecords)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Could not load history.");
      })
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <section className="panel p-5">
      <div className="mb-5">
        <div className="muted-label">Audit trail</div>
        <h2 className="mt-2 text-lg font-semibold text-white">Prediction history</h2>
      </div>

      {isLoading ? <p className="text-sm text-zinc-400">Loading history...</p> : null}
      {error ? <p className="text-sm text-signal-fake">{error}</p> : null}
      {!isLoading && !error && records.length === 0 ? (
        <EmptyState title="No predictions logged">The backend has not returned any history records.</EmptyState>
      ) : null}
      {records.length > 0 ? <HistoryTable records={records} /> : null}
    </section>
  );
}
