# VeriFace - Same-Dataset Baseline (FaceForensics++ test split)

Checkpoint: epoch 15, val AUROC 0.9393

| Metric | Value |
|---|---|
| Test AUROC | 0.9486 |
| Precision | 0.8802 |
| Recall | 0.8820 |
| F1 | 0.8811 |
| Test set size | 1000 images (500 real, 500 fake) |

Confusion matrix (rows=true, cols=pred, order=[real, fake]):
```
[[440  60]
 [ 59 441]]
```

Note: this is a same-dataset (FaceForensics++) held-out test result. Cross-dataset
generalization (evaluating on Celeb-DF/DFDC, which the model never saw during
training) is reported separately, and is expected to be lower - see
cross_dataset_eval.py results.
