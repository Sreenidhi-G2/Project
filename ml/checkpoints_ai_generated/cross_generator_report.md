# VeriFace - AI-Generated-Face Detector: Cross-Generator Check

Checkpoint: epoch 11 (trained on FFHQ-real vs SDXL-fake, val AUROC 0.9999)

Tested on: julienlucas/midjourney-dalle-sd-dataset (Midjourney/DALL-E/Stable Diffusion mix, different collection pipeline)

| Metric | Value |
|---|---|
| AUROC | 0.5169 |
| Precision | 0.4978 |
| Recall | 0.9000 |
| F1 | 0.6410 |

**Caveat:** this test set contains general images, not necessarily face-cropped
portraits like the training data - a lower score may partly reflect input-
distribution mismatch rather than pure generalization failure. Report this
honestly alongside the number rather than over-claiming either way.
