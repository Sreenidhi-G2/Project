# VeriFace - Same-Dataset Baseline (FaceForensics++ test split)

Checkpoint: epoch 5, val AUROC 0.8205

| Metric | Value |
|---|---|
| Test AUROC | 0.7906 |
| Precision | 0.8936 |
| Recall | 0.7567 |
| F1 | 0.8195 |
| Test set size | 3508 images (721 real, 2787 fake) |

Confusion matrix (rows=true, cols=pred, order=[real, fake]):
```
[[ 470  251]
 [ 678 2109]]
```

Note: this is a same-dataset (FaceForensics++) held-out test result. Cross-dataset
generalization (evaluating on Celeb-DF/DFDC, which the model never saw during
training) is reported separately, and is expected to be lower - see
cross_dataset_eval.py results.
