# VeriFace

**An agentic eKYC video-fraud detection system — built for Razorpay's Open Track.**

VeriFace watches a video-KYC / liveness-check clip and answers one question: *is this actually a real, live person, or has this video been manipulated or AI-generated?* It combines two specialist deepfake-detection models with an independent LLM supervisor agent that reviews the actual visual evidence before signing off on a final verdict — not just a classifier score, but a reasoned, evidence-grounded decision.

---

## The problem

Razorpay's video-KYC and liveness-check flows assume the face on camera is a real, live person matching the claimed identity. That assumption is breaking in two different ways:

1. **Face-swap / reenactment deepfakes** — a real video gets a different face swapped in, or someone else's expressions reenacted onto it.
2. **Fully AI-generated video** — no real video is touched at all. A fraudster generates a synthetic clip from scratch using tools like Gemini or Sora, no face-swap software required.

I confirmed this second path is real and current by generating a test video with Gemini myself and running it through the pipeline — more on that below.

---

## What VeriFace does

For every submitted video, VeriFace runs a two-stage pipeline:

**Stage 1 — ML ensemble.** OpenCV samples frames, MTCNN crops the face in each, and **two specialist models** score every frame independently:
- a **face-swap/reenactment detector**, fine-tuned on FaceForensics++
- an **AI-generated-content detector**, trained on a mix of Midjourney, DALL-E, and Stable Diffusion output

Each specialist gets its own mean/max/variance-based verdict (REAL / REVIEW / FAKE), and the two are combined by taking whichever one raised the more severe concern — an ensemble, not a single generalist model, because the two attack types leave genuinely different artifacts.

**Stage 2 — Supervisor agent.** The ML ensemble's output, along with a handful of representative frames from the video, is handed to an LLM supervisor. The supervisor is explicitly told the model scores are **evidence, not ground truth** — it independently inspects the actual frames, writes down concrete visual findings *before* it's allowed to consider the specialist scores, and only then decides whether to agree, partially agree, disagree, or escalate to manual review. It can override the ML ensemble entirely if the visual evidence doesn't support it.

The final response includes the ML scores, the supervisor's independent findings, its reasoning, and a confidence value — enough for a fraud analyst to actually understand *why* a case was flagged, not just that it was.

---

## Why an agent, not just a bigger classifier

A pure classifier ensemble can tell you a probability. It can't explain itself, and it can't reason about context the way "this score is high but the visual evidence looks clean" requires. The supervisor agent adds a layer that:

- Grounds its decision in the actual submitted frames, not just numbers
- Can catch cases where the ML ensemble and reality disagree
- Produces a human-readable justification a review-queue analyst can act on
- Uses REAL / FAKE / **REVIEW** as first-class outcomes — REVIEW isn't "the model is unsure," it's a deliberate decision to route a case to a human, which is how a real fraud-ops pipeline should behave for ambiguous evidence

Getting the agent to actually reason independently, rather than just rubber-stamping the ML scores, took real iteration — covered below.

---

## Results (the honest version)

### Face-swap / reenactment detector
Fine-tuned EfficientNet-B4, trained on FaceForensics++ (300 videos/category).

| Test | AUROC |
|---|---|
| Held-out test split (same dataset) | 0.79 |
| Cross-dataset generalization (Celeb-DF, official benchmark split) | 0.63 |

Per manipulation method: Deepfakes **0.92** → FaceSwap **0.84** → Face2Face **0.82** → NeuralTextures **0.70** — matches published difficulty rankings almost exactly (NeuralTextures only manipulates the mouth region, a well-known hard case in this literature).

### AI-generated-content detector
Trained on a mixed generator set (Midjourney + DALL-E + Stable Diffusion).

| Test | AUROC |
|---|---|
| Held-out test split (same distribution) | 0.95 |
| Cross-generator check (Stable Diffusion XL, unseen during training) | 0.56 |

That 0.56 isn't great, and I'm not going to dress it up. Cross-generator generalization for AI-image detection is a genuinely hard, actively-researched open problem. Broadening training data from a single generator (SDXL-only, which scored 0.52 — pure chance — on unseen generators) to a mixed set improved things modestly, but didn't solve it.

---

## The debugging story

**Finding #1 — identity memorization, not manipulation detection.** The first training run looked badly broken: 0.98 AUROC on training data, 0.55 on validation. A diagnostic that broke validation performance down *by manipulation method* showed near-chance performance uniformly across every method, including Deepfakes — normally the easiest to catch. That ruled out "some methods are hard" and pointed at something structural: only 100 source videos per category meant the model was memorizing ~100 specific faces, not learning general manipulation cues. Scaling to 300 videos/category fixed it — validation AUROC jumped from 0.57 to 0.82.

**Finding #2 — a shortcut disguised as a perfect score.** An early AI-generated-photo detector scored a suspicious 0.9999 validation AUROC. A cross-generator test (Stable Diffusion training → Midjourney/DALL-E evaluation) confirmed the suspicion: performance collapsed to 0.52. The model had learned to distinguish two different data-export pipelines, not "AI-generated vs. real."

**Finding #3 — a Gemini-generated video that almost slipped through.** I generated a test video with Gemini and ran it through the pipeline. On average across frames, the face-swap specialist scored it low enough to auto-approve — but the *maximum* single-frame score was 0.92. That's why the video decision logic escalates to manual review on a single high-confidence outlier frame, not just the mean; averaging alone would have missed this one.

**Finding #4 — the supervisor agent was rubber-stamping.** Once the agent layer was working end-to-end, it became clear it was just agreeing with whatever the ML ensemble said — a classic LLM-as-judge anchoring failure, caused by showing the model the specialist scores before it had looked at the actual frames. Fixed by restructuring the prompt and output schema: images are now shown first, and the model must populate a required `independent_visual_findings` field — specific, concrete observations — before it's allowed to state a verdict, with an explicit rule that vague agreement without supporting evidence isn't acceptable.

**Finding #5 — my own camera footage triggered false positives.** Testing against my own webcam clips (not benchmark data) showed the face-swap specialist producing borderline-to-high scores on genuine footage — a real domain gap between FaceForensics++'s studio/YouTube-sourced training data and consumer camera video.

None of these were comfortable to find in the moment, but they're why I trust the final system more than I would if everything had worked on the first try.

---

## Known limitations

- **Cross-generator AI-content detection is weak** (0.56 AUROC on an unseen generator) — a real open problem, not a quick fix.
- **False-positive tendency on personal camera footage** — a domain gap between benchmark training data and real consumer video that would need broader real-world calibration data to fix properly.
- **The supervisor agent has a real, recurring cost** — every video sent to the agent spends real API credits (OpenRouter/OpenAI), unlike the local ML models which run free on-device. This is worth factoring into any production cost model.
- **Anchoring mitigation is a mitigation, not a guarantee** — the images-first, findings-before-verdict prompt structure meaningfully reduces rubber-stamping, but a sufficiently anchored model can still produce vague "findings" that don't reflect genuine independent reasoning. Worth spot-checking outputs periodically.
- **Out of scope for this build**: real-time video-call stream analysis, voice/audio deepfake detection, adversarial robustness hardening.

---

## Architecture

```
veriface/
  ml/
    preprocessing/       # frame extraction, face crop/align, dataset prep
    training/             # model.py, dataset.py, train.py, evaluate.py,
                          # cross_dataset_eval.py, cross_generator_eval.py,
                          # video_inference_ensemble.py, gradcam_explain.py,
                          # diagnose_by_category.py
    checkpoints/           # face-swap detector weights
    checkpoints_ai_v2/      # AI-generated-content detector weights
  backend/
    app/
      main.py               # FastAPI: POST /predict/video, GET /history, GET /health
      inference.py           # ModelBundle - both specialists loaded once at startup
      database.py             # SQLite prediction history / audit trail
      schemas.py               # response models
      supervisor/
        agent.py                # supervisor LLM call, images-first message ordering
        prompts.py                # system/user prompts, output JSON schema
        frame_sampler.py           # picks ~6-8 representative frames per video
```

**Models**: EfficientNet-B4 (`timm`), fine-tuned for binary classification, both specialists sharing the same architecture.
**Face detection/cropping**: MTCNN (`facenet-pytorch`).
**Supervisor**: structured JSON output via an OpenAI-compatible Responses API, REAL/FAKE/REVIEW verdicts, images sent before specialist scores to reduce anchoring.
**Backend**: FastAPI, SQLite for prediction history, models loaded once at startup (not per-request).

## Running it

```bash
# ML environment
pip install torch torchvision opencv-python facenet-pytorch timm albumentations \
            scikit-learn pandas tqdm

# Backend
pip install fastapi uvicorn python-multipart sqlalchemy pillow openai python-dotenv

cd backend/app
python -m uvicorn main:app --reload
# then open http://127.0.0.1:8000/docs and try POST /predict/video
```

You'll need an API key for the supervisor agent (set as an environment variable, loaded via `.env` — see `supervisor/agent.py` for the exact variable name it expects).

## A note on data

The raw video datasets (FaceForensics++, Celeb-DF) and extracted face-crop datasets are **not in this repo** — they're large, and both carry usage terms that don't allow public redistribution. They're stored separately in Google Drive: **https://drive.google.com/drive/folders/1OXvFCpYAFtl7zKF8ULPOlZtt9beYiNnp?usp=drive_link**. Model checkpoints (`ml/checkpoints/`, `ml/checkpoints_ai_v2/`) are small enough to keep in the repo directly.

## Acknowledgments

Built as an academic AI/ML project at BNM Institute of Technology, Bengaluru, under the guidance of Dr. Mahanthesha U.
