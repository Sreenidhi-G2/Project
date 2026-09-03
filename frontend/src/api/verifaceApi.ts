import { apiRequest } from "./client";
import type {
  HealthResponse,
  HistoryRecord,
  ImagePredictionResponse,
  VideoPredictionResponse,
} from "../types/api";

function uploadForm(file: File): FormData {
  const formData = new FormData();
  formData.append("file", file);
  return formData;
}

export const verifaceApi = {
  predictImage(file: File): Promise<ImagePredictionResponse> {
    return apiRequest<ImagePredictionResponse>("/predict/image", {
      method: "POST",
      body: uploadForm(file),
    });
  },

  predictVideo(file: File): Promise<VideoPredictionResponse> {
    return apiRequest<VideoPredictionResponse>("/predict/video", {
      method: "POST",
      body: uploadForm(file),
    });
  },

  getHistory(limit = 50): Promise<HistoryRecord[]> {
    return apiRequest<HistoryRecord[]>(`/history?limit=${encodeURIComponent(limit)}`);
  },

  getHealth(): Promise<HealthResponse> {
    return apiRequest<HealthResponse>("/health");
  },
};
