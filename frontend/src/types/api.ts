export type Verdict = "REAL" | "FAKE" | "REVIEW" | string;

export type ModelAgreement = "AGREE" | "PARTIAL_AGREEMENT" | "DISAGREE" | string;

export interface ImageSupervisorResponse {
  verdict: Verdict;
  confidence: number;
  assessment: string;
  key_evidence: string[];
  model_agreement: ModelAgreement;
  reasoning: string;
}

export interface VideoSupervisorResponse extends ImageSupervisorResponse {
  frames_reviewed: number;
  frame_timestamps: Array<number | null>;
}

export interface ImagePredictionResponse {
  overall_verdict: Verdict;
  driven_by: string;
  reason: string;
  faceswap_score: number;
  faceswap_verdict: Verdict;
  ai_generated_score: number;
  ai_generated_verdict: Verdict;
  gradcam_region: string;
  gradcam_path: string;
  supervisor: ImageSupervisorResponse;
}

export interface VideoPredictionResponse {
  overall_verdict: Verdict;
  driven_by: string;
  num_frames_analyzed: number;
  faceswap_mean_score: number;
  faceswap_max_score: number;
  faceswap_verdict: Verdict;
  ai_generated_mean_score: number;
  ai_generated_max_score: number;
  ai_generated_verdict: Verdict;
  supervisor: VideoSupervisorResponse;
}

export type PredictionResponse = ImagePredictionResponse | VideoPredictionResponse;

export interface HistoryRecord {
  id: number;
  timestamp: string;
  media_type: string;
  overall_verdict: Verdict;
  driven_by: string;
  faceswap_score: number | null;
  ai_generated_score: number | null;
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
}

export interface ApiErrorPayload {
  detail?: string;
  error?: string;
}
