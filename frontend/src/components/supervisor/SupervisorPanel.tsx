import type { ImageSupervisorResponse, VideoSupervisorResponse } from "../../types/api";
import { formatPercent } from "../../utils/format";
import { StatusBadge } from "../common/StatusBadge";

interface SupervisorPanelProps {
  supervisor: ImageSupervisorResponse | VideoSupervisorResponse;
}

function hasFrameMetadata(
  supervisor: ImageSupervisorResponse | VideoSupervisorResponse,
): supervisor is VideoSupervisorResponse {
  return "frames_reviewed" in supervisor;
}

export function SupervisorPanel({ supervisor }: SupervisorPanelProps) {
  return (
    <section className="panel p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="muted-label">Supervisor assessment</div>
          <h2 className="mt-2 text-lg font-semibold text-white">{supervisor.assessment}</h2>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={supervisor.verdict} />
          <span className="text-sm text-zinc-400">{formatPercent(supervisor.confidence)}</span>
        </div>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div>
          <div className="muted-label">Reasoning</div>
          <p className="mt-2 text-sm leading-6 text-zinc-300">{supervisor.reasoning}</p>
        </div>
        <div>
          <div className="muted-label">Evidence</div>
          <ul className="mt-2 space-y-2 text-sm text-zinc-300">
            {supervisor.key_evidence.map((item, index) => (
              <li key={`${item}-${index}`} className="rounded border border-white/10 bg-surface-850 p-3">
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {hasFrameMetadata(supervisor) ? (
        <div className="mt-5 rounded-lg border border-white/10 bg-surface-850 p-4 text-sm text-zinc-400">
          Reviewed {supervisor.frames_reviewed} frames
          {supervisor.frame_timestamps.length > 0
            ? ` at ${supervisor.frame_timestamps.filter((item) => item !== null).join(", ")} seconds`
            : ""}
        </div>
      ) : null}
    </section>
  );
}
