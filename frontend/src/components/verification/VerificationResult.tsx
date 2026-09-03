import type { ImagePredictionResponse, VideoPredictionResponse } from "../../types/api";
import { formatPercent } from "../../utils/format";
import { MetricCard } from "../common/MetricCard";
import { StatusBadge } from "../common/StatusBadge";
import { SupervisorPanel } from "../supervisor/SupervisorPanel";

interface VerificationResultProps {
  result: ImagePredictionResponse | VideoPredictionResponse;
}

function isVideoResult(
  result: ImagePredictionResponse | VideoPredictionResponse,
): result is VideoPredictionResponse {
  return "num_frames_analyzed" in result;
}

export function VerificationResult({ result }: VerificationResultProps) {
  const video = isVideoResult(result);

  return (
    <section className="space-y-4">
      <div className="panel p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="muted-label">Overall verdict</div>
            <div className="mt-2 flex items-center gap-3">
              <StatusBadge status={result.overall_verdict} />
              <span className="text-sm text-zinc-400">{result.driven_by}</span>
            </div>
          </div>
          <StatusBadge status={result.supervisor.verdict} />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Face-swap score"
          value={formatPercent(video ? result.faceswap_mean_score : result.faceswap_score)}
          detail={video ? `Max ${formatPercent(result.faceswap_max_score)}` : result.faceswap_verdict}
        />
        <MetricCard
          label="AI-generated score"
          value={formatPercent(video ? result.ai_generated_mean_score : result.ai_generated_score)}
          detail={
            video ? `Max ${formatPercent(result.ai_generated_max_score)}` : result.ai_generated_verdict
          }
        />
        <MetricCard
          label={video ? "Frames analyzed" : "Grad-CAM region"}
          value={video ? String(result.num_frames_analyzed) : result.gradcam_region}
          detail={video ? "Sampled by backend" : "Backend filesystem output"}
        />
        <MetricCard
          label="Supervisor confidence"
          value={formatPercent(result.supervisor.confidence)}
          detail={result.supervisor.model_agreement}
        />
      </div>

      {!video ? (
        <div className="panel p-5">
          <div className="muted-label">Model reason</div>
          <p className="mt-2 text-sm leading-6 text-zinc-300">{result.reason}</p>
          <div className="mt-3 rounded border border-white/10 bg-surface-850 px-3 py-2 text-xs text-zinc-500">
            {result.gradcam_path}
          </div>
        </div>
      ) : null}

      <SupervisorPanel supervisor={result.supervisor} />
    </section>
  );
}
