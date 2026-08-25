# VeriFace - Cross-Dataset Generalization (trained on FF++, tested on Celeb-DF)

Checkpoint: epoch 5 (FF++ val AUROC 0.8205)

**Cross-dataset AUROC: 0.6304**

This model was trained entirely on FaceForensics++ and evaluated here on
Celeb-DF, a dataset it never saw during training. A drop relative to the
same-dataset FF++ test AUROC is expected and well-documented in the
deepfake-detection literature - it reflects the model partially relying on
FF++-specific compression/artifact signatures that don't fully transfer to
Celeb-DF's different generation pipeline and video characteristics.

## Threshold sweep (operating-point tradeoff for a fraud-prevention use case)

| Threshold | Precision | Recall | F1 | False-Negative Rate |
|---|---|---|---|---|
| 0.2 | 0.6825 | 0.8645 | 0.7628 | 0.1355 |
| 0.3 | 0.6922 | 0.8023 | 0.7432 | 0.1977 |
| 0.4 | 0.7020 | 0.7483 | 0.7244 | 0.2517 |
| 0.5 | 0.7086 | 0.6956 | 0.7020 | 0.3044 |
| 0.6 | 0.7198 | 0.6432 | 0.6793 | 0.3568 |
| 0.7 | 0.7274 | 0.5786 | 0.6445 | 0.4214 |

False-Negative Rate = fraction of actual deepfakes the model missed. In a
KYC/fraud-prevention deployment, a missed deepfake (false negative) is
typically more costly than a false positive (a genuine user flagged for
manual review), so the operating threshold should be chosen with that
asymmetry in mind rather than defaulting to 0.5.
