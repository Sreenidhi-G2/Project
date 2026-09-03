"""
VeriFace Backend - main.py
FastAPI app exposing:
  POST /predict/image  - upload an image, get a verdict + Grad-CAM heatmap
  POST /predict/video  - upload a video, get a verdict from both specialists
  GET  /history         - query past predictions (for the review-queue UI)

Run with:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m uvicorn main:app --reload
(run from inside the backend/app directory)
"""

import hashlib
import tempfile
import os
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from inference import ModelBundle,  score_video_file
from database import init_db, get_db, log_prediction, PredictionRecord
from supervisor.frame_sampler import sample_representative_frames
from supervisor.agent import run_supervisor 
from schemas import VideoPredictionResponse, HistoryRecord
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veriface")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 100

app = FastAPI(
    title="VeriFace API",
    description="Fraud-verification layer for identity submission points "
                "- flags synthetic/manipulated faces in KYC photos and "
                "liveness-check videos.",
    version="1.0.0",
)

origins = [
    "http://127.0.0.1:5174"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all request headers
)

model_bundle = None  # loaded once on startup, see startup_event below


@app.on_event("startup")
def startup_event():
    global model_bundle
    logger.info("Loading models - this happens once at startup...")
    model_bundle = ModelBundle()
    init_db()
    logger.info("VeriFace API ready.")


def hash_filename(filename: str) -> str:
    """Store a hash, not the raw filename, in the prediction-history log -
    a lightweight privacy precaution for what's effectively an audit trail."""
    return hashlib.sha256(filename.encode()).hexdigest()[:16]


# @app.post("/predict/image", response_model=ImagePredictionResponse)
# async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     if file.content_type not in ALLOWED_IMAGE_TYPES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file type '{file.content_type}'. "
#                    f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
#         )

#     contents = await file.read()
#     size_mb = len(contents) / (1024 * 1024)
#     if size_mb > MAX_IMAGE_SIZE_MB:
#         raise HTTPException(
#             status_code=400,
#             detail=f"File too large ({size_mb:.1f}MB). Max: {MAX_IMAGE_SIZE_MB}MB.",
#         )

#     suffix = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(contents)
#         image_path = tmp.name

#     request_id = hashlib.sha256(contents).hexdigest()[:12]
#     try:
#         result = score_image_bytes(model_bundle, contents, request_id)

#         if "error" in result:
#             raise HTTPException(status_code=422, detail=result["error"])

#         supervisor_result = await run_image_supervisor(
#             image_path=image_path,
#             model_result=result,
#         )
#         result["supervisor"] = supervisor_result
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Image inference failed")
#         raise HTTPException(status_code=500, detail="Inference failed. See server logs.")
#     finally:
#         if os.path.exists(image_path):
#             os.unlink(image_path)

#     log_prediction(
#         db, media_type="image", filename_hash=hash_filename(file.filename or "unknown"),
#         overall_verdict=result["overall_verdict"], driven_by=result["driven_by"],
#         faceswap_score=result["faceswap_score"],
#         ai_generated_score=result["ai_generated_score"],
#     )
#     logger.info(f"[image] verdict={result['overall_verdict']} "
#                 f"driven_by={result['driven_by']}")

#     return result


@app.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max: {MAX_VIDEO_SIZE_MB}MB.",
        )

    # OpenCV's VideoCapture needs a real file path, not in-memory bytes -
    # write to a temp file, process, then clean up
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    request_id = hashlib.sha256(contents).hexdigest()[:12]
    try:
        result = score_video_file(
            model_bundle,
            tmp_path
        )

        if "error" in result:
            raise HTTPException(
                status_code=422,
                detail=result["error"]
            )

        supervisor_frame_dir = os.path.join(
            tempfile.gettempdir(),
            f"veriface_supervisor_{request_id}"
        )

        representative_frames = sample_representative_frames(
            video_path=tmp_path,
            output_dir=supervisor_frame_dir,
            num_frames=8,
            device=model_bundle.device
        )

        if not representative_frames:
            raise HTTPException(
                status_code=422,
                detail="Could not extract representative frames."
            )

        supervisor_result = await run_supervisor(
            frames=representative_frames,
            media_type="video",
            model_result=result
        )
        result["supervisor"] = supervisor_result    

    except HTTPException:
        raise

    except Exception:
        logger.exception("Video inference failed")
        raise HTTPException(
            status_code=500,
            detail="Inference failed. See server logs."
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    log_prediction(
        db, media_type="video", filename_hash=hash_filename(file.filename or "unknown"),
        overall_verdict=result["overall_verdict"], driven_by=result["driven_by"],
        faceswap_score=result["faceswap_mean_score"],
        ai_generated_score=result["ai_generated_mean_score"],
    )
    logger.info(f"[video] verdict={result['overall_verdict']} "
                f"driven_by={result['driven_by']} "
                f"frames={result['num_frames_analyzed']}")

    return result


@app.get("/history", response_model=list[HistoryRecord])
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    return records


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": model_bundle is not None}
