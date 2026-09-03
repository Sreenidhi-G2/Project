"""VeriFace Backend - schemas.py - Pydantic response models."""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime



class SupervisorResponse(BaseModel):
    verdict: str
    confidence: float
    assessment: str
    key_evidence: list[str]
    model_agreement: str
    reasoning: str
    frames_reviewed: int
    frame_timestamps: list[float | None]


# class ImageSupervisorResponse(BaseModel):
#     verdict: str
#     confidence: float
#     assessment: str
#     key_evidence: list[str]
#     model_agreement: str
#     reasoning: str



# class ImagePredictionResponse(BaseModel):
#     overall_verdict: str
#     driven_by: str
#     reason: str
#     faceswap_score: float
#     faceswap_verdict: str
#     ai_generated_score: float
#     ai_generated_verdict: str
#     gradcam_region: str
#     gradcam_path: str
#     supervisor: ImageSupervisorResponse

class VideoPredictionResponse(BaseModel):
    overall_verdict: str
    driven_by: str
    num_frames_analyzed: int
    faceswap_mean_score: float
    faceswap_max_score: float
    faceswap_verdict: str
    ai_generated_mean_score: float
    ai_generated_max_score: float
    ai_generated_verdict: str
    supervisor: SupervisorResponse


class ErrorResponse(BaseModel):
    error: str


class HistoryRecord(BaseModel):
    id: int
    timestamp: datetime
    media_type: str
    overall_verdict: str
    driven_by: str
    faceswap_score: Optional[float] = None
    ai_generated_score: Optional[float] = None

    class Config:
        from_attributes = True
