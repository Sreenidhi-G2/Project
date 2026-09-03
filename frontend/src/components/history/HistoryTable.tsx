import type { HistoryRecord } from "../../types/api";
import { formatDateTime, formatPercent } from "../../utils/format";
import { StatusBadge } from "../common/StatusBadge";

interface HistoryTableProps {
  records: HistoryRecord[];
}

export function HistoryTable({ records }: HistoryTableProps) {
  return (
    <div className="overflow-hidden rounded-lg border border-white/10">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-white/10 text-sm">
          <thead className="bg-surface-850 text-left text-xs uppercase tracking-normal text-zinc-500">
            <tr>
              <th className="px-4 py-3 font-medium">Time</th>
              <th className="px-4 py-3 font-medium">Type</th>
              <th className="px-4 py-3 font-medium">Verdict</th>
              <th className="px-4 py-3 font-medium">Driven by</th>
              <th className="px-4 py-3 font-medium">Face-swap</th>
              <th className="px-4 py-3 font-medium">AI-generated</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10 bg-surface-900">
            {records.map((record) => (
              <tr key={record.id}>
                <td className="whitespace-nowrap px-4 py-3 text-zinc-300">
                  {formatDateTime(record.timestamp)}
                </td>
                <td className="px-4 py-3 capitalize text-zinc-400">{record.media_type}</td>
                <td className="px-4 py-3">
                  <StatusBadge status={record.overall_verdict} />
                </td>
                <td className="px-4 py-3 text-zinc-400">{record.driven_by}</td>
                <td className="px-4 py-3 text-zinc-300">{formatPercent(record.faceswap_score)}</td>
                <td className="px-4 py-3 text-zinc-300">
                  {formatPercent(record.ai_generated_score)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
