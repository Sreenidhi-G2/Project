import { useCallback, useState } from "react";
import { ApiError } from "../api/client";
import { verifaceApi } from "../api/verifaceApi";
import type { ImagePredictionResponse, VideoPredictionResponse } from "../types/api";

type MediaType = "video";

interface VerificationState {
  isLoading: boolean;
  error: string | null;
  result: ImagePredictionResponse | VideoPredictionResponse | null;
  mediaType: MediaType;
}

export function useVerification() {
  const [state, setState] = useState<VerificationState>({
    isLoading: false,
    error: null,
    result: null,
    mediaType: "video",
  });

  const setMediaType = useCallback((mediaType: MediaType) => {
    setState((current) => ({ ...current, mediaType, error: null, result: null }));
  }, []);

  const submit = useCallback(async (file: File) => {
    setState((current) => ({ ...current, isLoading: true, error: null }));

    try {
      const result =

        await verifaceApi.predictVideo(file);

      setState((current) => ({ ...current, isLoading: false, result }));
    } catch (error) {
      const message =
        error instanceof ApiError || error instanceof Error
          ? error.message
          : "Verification failed.";
      setState((current) => ({ ...current, isLoading: false, error: message }));
    }
  }, [state.mediaType]);

  return {
    ...state,
    setMediaType,
    submit,
  };
}
