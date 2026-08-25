# VeriFace - Same-Dataset Baseline (FaceForensics++ test split)

Checkpoint: epoch 15, val AUROC 0.9393

| Metric | Value |
|---|---|
| Test AUROC | 0.5627 |
| Precision | 0.4707 |
| Recall | 0.9567 |
| F1 | 0.6310 |
| Test set size | 589 images (312 real, 277 fake) |

Confusion matrix (rows=true, cols=pred, order=[real, fake]):
```
[[ 14 298]
 [ 12 265]]
```

Note: this is a same-dataset (FaceForensics++) held-out test result. Cross-dataset
generalization (evaluating on Celeb-DF/DFDC, which the model never saw during
training) is reported separately, and is expected to be lower - see
cross_dataset_eval.py results.
