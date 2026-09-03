
"""
VeriFace Supervisor Agent

Pipeline:

    Existing ML Models
            +
    Representative Video Frames
            ↓
       GPT-5.6 Luna
       Direct OpenAI API
            ↓
    REAL / FAKE / REVIEW

The supervisor is an independent adjudicator.

It receives:
    1. Representative frames from the video
    2. Outputs from the FaceSwap specialist
    3. Outputs from the AI-generated-content specialist
    4. Existing ML ensemble verdict

The supervisor is allowed to disagree with the ML ensemble.

IMPORTANT:
    REVIEW means MANUAL REVIEW.
    REVIEW is NOT called "uncertain".

The supervisor does NOT receive Grad-CAM images.
Only representative frames are sent.
"""


import os
import json
import base64
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    OUTPUT_SCHEMA,
    
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gpt-5.6-luna"

# Only representative frames are sent to the LLM.
MAX_FRAMES_TO_SEND = 8

# Keep the response compact.
MAX_OUTPUT_TOKENS = 2000

# Low temperature because this is a verification/adjudication task.
TEMPERATURE = 0.1

# OpenAI API
OPENAI_BASE_URL = "https://api.openai.com/v1"


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("veriface.supervisor")


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set."
    )


# ============================================================
# OPENAI CLIENT
# ============================================================

client = AsyncOpenAI(
    api_key=api_key,
    base_url=OPENAI_BASE_URL,
)


# ============================================================
# IMAGE ENCODING
# ============================================================

def image_to_data_url(image_path: str) -> str:
    """
    Convert a representative frame into a base64 data URL.

    This allows the image to be sent directly to the OpenAI
    Responses API without uploading it separately.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Supervisor frame does not exist: {image_path}"
        )

    with open(path, "rb") as f:
        image_bytes = f.read()

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"

    elif suffix == ".png":
        mime_type = "image/png"

    elif suffix == ".webp":
        mime_type = "image/webp"

    else:
        raise ValueError(
            f"Unsupported supervisor image format: {suffix}"
        )

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# MODEL RESULT PREPARATION
# ============================================================

def build_model_summary(model_result: dict) -> dict:
    """
    Convert the existing ML inference result into a compact
    representation for the supervisor.

    We deliberately send only information useful for
    adjudication.
    """

    return {
        "faceswap": {
            "mean_score": model_result.get(
                "faceswap_mean_score"
            ),
            "max_score": model_result.get(
                "faceswap_max_score"
            ),
            "variance": model_result.get(
                "faceswap_variance"
            ),
            "verdict": model_result.get(
                "faceswap_verdict"
            ),
        },

        "ai_generated": {
            "mean_score": model_result.get(
                "ai_generated_mean_score"
            ),
            "max_score": model_result.get(
                "ai_generated_max_score"
            ),
            "variance": model_result.get(
                "ai_generated_variance"
            ),
            "verdict": model_result.get(
                "ai_generated_verdict"
            ),
        },

        "ensemble": {
            "overall_verdict": model_result.get(
                "overall_verdict"
            ),
            "driven_by": model_result.get(
                "driven_by"
            ),
        },

        "num_frames_analyzed": model_result.get(
            "num_frames_analyzed"
        ),
    }


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(
    media_type: str,
    model_result: dict,
    num_frames_sent: int,
) -> str:
    """
    Build the textual part of the supervisor request.
    """

    summary = build_model_summary(
        model_result
    )

    faceswap_result = json.dumps(
        summary["faceswap"],
        indent=2,
    )

    ai_generated_result = json.dumps(
        summary["ai_generated"],
        indent=2,
    )

    ensemble_result = json.dumps(
        summary["ensemble"],
        indent=2,
    )

    video_information = json.dumps(
        {
            "frames_analyzed_by_detectors":
                summary["num_frames_analyzed"],

            "representative_frames_sent_to_supervisor":
                num_frames_sent,
        },
        indent=2,
    )

    return USER_PROMPT_TEMPLATE.format(
        media_type=media_type,

        faceswap_result=faceswap_result,

        ai_generated_result=ai_generated_result,

        ensemble_result=ensemble_result,

        video_information=video_information,
    )


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_from_response(response) -> dict:
    """
    Extract structured JSON from the OpenAI Responses API.

    Preferred path:
        response.output_text

    Fallback:
        inspect output items manually.

    Raises:
        RuntimeError if no usable JSON is found.
    """

    raw_output = getattr(
        response,
        "output_text",
        None,
    )

    if raw_output:
        raw_output = raw_output.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    if raw_output:
        try:
            return json.loads(raw_output)

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # Fallback: inspect response.output
    # --------------------------------------------------------

    output_items = getattr(
        response,
        "output",
        None,
    )

    if output_items:

        for item in output_items:

            content_items = getattr(
                item,
                "content",
                None,
            )

            if not content_items:
                continue

            for content in content_items:

                text = getattr(
                    content,
                    "text",
                    None,
                )

                if not text:
                    continue

                try:
                    return json.loads(text.strip())

                except json.JSONDecodeError:
                    continue

    # --------------------------------------------------------
    # Nothing worked
    # --------------------------------------------------------

    raise RuntimeError(
        "GPT-5.6 Luna returned invalid or empty JSON."
    )


# ============================================================
# SUPERVISOR
# ============================================================


async def run_supervisor(
    frames: list,
    media_type: str,
    model_result: dict,
):
    """
    Main supervisor entry point.

    Parameters
    ----------
    frames:
        Representative frame metadata returned by
        frame_sampler.py.

        Expected format:

        [
            {
                "path": "...",
                "frame_index": 100,
                "timestamp_seconds": 4.2
            },
            ...
        ]

    media_type:
        "video" or "image".

    model_result:
        Existing output from the two ML specialists.

    Returns
    -------
    dict
        Supervisor verdict.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if not frames:
        raise ValueError(
            "Supervisor received no visual frames."
        )

    # ========================================================
    # LIMIT FRAMES
    # ========================================================

    frames = frames[
        :MAX_FRAMES_TO_SEND
    ]

    logger.info(
        "Supervisor analyzing %d representative frames",
        len(frames),
    )

    # ========================================================
    # TEXT PROMPT
    # ========================================================

    user_prompt = build_user_prompt(
        media_type=media_type,
        model_result=model_result,
        num_frames_sent=len(frames),
    )

    # ========================================================
    # MULTIMODAL INPUT
    # ========================================================

    content = [
        {
            "type": "input_text",
            "text": user_prompt,
        }
    ]

    for index, frame in enumerate(frames):

        image_path = frame["path"]

        frame_index = frame.get(
            "frame_index"
        )

        timestamp = frame.get(
            "timestamp_seconds"
        )

        # ----------------------------------------------------
        # Frame description
        # ----------------------------------------------------

        frame_description = (
            f"Representative frame {index + 1}.\n"
            f"Video frame index: {frame_index}.\n"
            f"Timestamp: {timestamp} seconds.\n"
            "Inspect this frame as visual evidence."
        )

        content.append(
            {
                "type": "input_text",
                "text": frame_description,
            }
        )

        # ----------------------------------------------------
        # Actual image
        # ----------------------------------------------------

        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(
                    image_path
                ),
                "detail": "high",
            }
        )

    # ========================================================
    # OPENAI RESPONSES API
    # ========================================================

    try:

        response = await client.responses.create(

            model=MODEL_NAME,

            instructions=SYSTEM_PROMPT,

            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],

            # ------------------------------------------------
            # Structured output
            # ------------------------------------------------
            text={
                "format": {
                    "type": "json_schema",
                    "name": "veriface_supervisor_result",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                }
            },

            # ------------------------------------------------
            # Keep output concise.
            # ------------------------------------------------
            max_output_tokens=2500,

            # ------------------------------------------------
            # Low reasoning randomness.
            # ------------------------------------------------
            #temperature=TEMPERATURE,
        )

    except Exception as e:

        logger.exception(
            "GPT-5.6 Luna supervisor request failed"
        )

        raise RuntimeError(
            f"Supervisor API request failed: {str(e)}"
        ) from e

    # ========================================================
    # EXTRACT JSON
    # ========================================================

    try:

        result = extract_json_from_response(
            response
        )

    except Exception as e:

        # Log the raw output for debugging without
        # exposing it as an API response.

        raw_output = getattr(
            response,
            "output_text",
            None,
        )

        logger.error(
            "Supervisor returned malformed JSON: %s",
            raw_output,
        )

        raise RuntimeError(
            "GPT-5.6 Luna returned invalid JSON."
        ) from e

    # ========================================================
    # VALIDATE BASIC SUPERVISOR RESULT
    # ========================================================

    required_fields = [
        "verdict",
        "confidence",
        "assessment",
        "key_evidence",
        "model_agreement",
        "reasoning",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing_fields:

        raise RuntimeError(
            "Supervisor response is missing required fields: "
            + ", ".join(missing_fields)
        )

    # ========================================================
    # NORMALIZE VERDICT
    # ========================================================

    verdict = str(
        result["verdict"]
    ).upper().strip()

    allowed_verdicts = {
        "REAL",
        "FAKE",
        "REVIEW",
    }

    if verdict not in allowed_verdicts:

        raise RuntimeError(
            f"Supervisor returned invalid verdict: {verdict}"
        )

    result["verdict"] = verdict

    # ========================================================
    # NORMALIZE CONFIDENCE
    # ========================================================

    try:

        confidence = float(
            result["confidence"]
        )

    except (TypeError, ValueError):

        confidence = 0.0

    # Clamp confidence to [0, 1].

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    result["confidence"] = confidence

    # ========================================================
    # SERVER-SIDE METADATA
    # ========================================================

    result["frames_reviewed"] = len(
        frames
    )

    result["frame_timestamps"] = [
        frame.get("timestamp_seconds")
        for frame in frames
    ]

    # ========================================================
    # LOG RESULT
    # ========================================================

    logger.info(
        "[supervisor] verdict=%s confidence=%.2f "
        "agreement=%s frames=%d",
        result["verdict"],
        result["confidence"],
        result.get("model_agreement"),
        len(frames),
    )

    return result



def _validate_and_normalize(result: dict, num_frames: int, timestamps: list) -> dict:
    required_fields = [
        "independent_visual_findings",
        "verdict",
        "confidence",
        "assessment",
        "key_evidence",
        "model_agreement",
        "reasoning",
    ]
    missing_fields = [f for f in required_fields if f not in result]
    if missing_fields:
        raise RuntimeError(
            "Supervisor response is missing required fields: "
            + ", ".join(missing_fields)
        )
 
    verdict = str(result["verdict"]).upper().strip()
    allowed_verdicts = {"REAL", "FAKE", "REVIEW"}
    if verdict not in allowed_verdicts:
        raise RuntimeError(f"Supervisor returned invalid verdict: {verdict}")
    result["verdict"] = verdict
 
    try:
        confidence = float(result["confidence"])
    except (TypeError, ValueError):
        confidence = 0.0
    result["confidence"] = max(0.0, min(1.0, confidence))
 
    result["frames_reviewed"] = num_frames
    result["frame_timestamps"] = timestamps
 
    logger.info(
        "[supervisor] verdict=%s confidence=%.2f agreement=%s frames=%d",
        result["verdict"], result["confidence"],
        result.get("model_agreement"), num_frames,
    )
 
    return result



