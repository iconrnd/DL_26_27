"""Verification of every hand computation typeset in the Lecture 1 deck.

Run:  python verify_numerics.py   ->  prints one PASS line per check.
(The bias-variance closure check lives in 02_bias_variance.py, which asserts
and prints its own PASS; recorded in PLAN.md.)
"""
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score, mean_absolute_error,
                             root_mean_squared_error)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def check(name, got, want, tol=1e-9):
    ok = np.allclose(got, want, atol=tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, hand value {want}")
    assert ok


# 1 ---- accuracy's failure on 99:1 imbalance ---------------------------------
y = np.r_[np.zeros(990), np.ones(10)]
y_hat = np.zeros(1000)                      # the "always negative" classifier
check("imbalance accuracy", (y == y_hat).mean(), 0.99)
check("imbalance recall", recall_score(y, y_hat, zero_division=0), 0.0)

# 2 ---- precision/recall/F1 from the frozen breast-cancer confusion matrix ---
# frozen by 03_metrics.py: TN=89, FP=1, FN=4, TP=49
tn, fp, fn, tp = 89, 1, 4, 49
y_true = np.r_[np.zeros(tn + fp), np.ones(fn + tp)]
y_pred = np.r_[np.zeros(tn), np.ones(fp), np.zeros(fn), np.ones(tp)]
check("precision = 49/50", precision_score(y_true, y_pred), 49 / 50)
check("recall = 49/53", recall_score(y_true, y_pred), 49 / 53)
p, r = 49 / 50, 49 / 53
check("F1 = 2pr/(p+r)", f1_score(y_true, y_pred), 2 * p * r / (p + r))
check("F1 numeric", f1_score(y_true, y_pred), 0.9515, tol=5e-5)

# 3 ---- the 4-example ROC by hand --------------------------------------------
y4, s4 = np.array([1, 0, 1, 0]), np.array([0.9, 0.7, 0.6, 0.2])
fpr, tpr, _ = roc_curve(y4, s4)
hand_points = [(0, 0), (0, .5), (.5, .5), (.5, 1), (1, 1)]
check("ROC vertex FPRs", fpr, [p_[0] for p_ in hand_points])
check("ROC vertex TPRs", tpr, [p_[1] for p_ in hand_points])
check("AUC by trapezoid = 0.75", roc_auc_score(y4, s4), 0.75)
# AUC = P(score of a positive > score of a negative): 3 of the 4 pos-neg pairs
pairs = [(sp > sn) for sp in s4[y4 == 1] for sn in s4[y4 == 0]]
check("AUC = correctly ranked pairs / all pairs", np.mean(pairs), 0.75)

# 4 ---- RMSE vs MAE on the 5-point outlier example ---------------------------
yt = np.array([2, 3, 5, 4, 10.])
yp = np.array([2.5, 3, 4, 4, 4.])
check("MAE = 7.5/5 = 1.5", mean_absolute_error(yt, yp), 1.5)
check("RMSE = sqrt(37.25/5)", root_mean_squared_error(yt, yp),
      np.sqrt(37.25 / 5))
check("RMSE numeric 2.7295", root_mean_squared_error(yt, yp), 2.7295, 5e-5)

# 5 ---- stratification: class proportions under a small split ----------------
data = load_breast_cancer()
yb = (data.target == 0).astype(int)
frac = yb.mean()
rng = np.random.default_rng(0)
sub = rng.choice(len(yb), 60, replace=False)
fr_rand, fr_strat = [], []
for s in range(200):
    _, te_r = train_test_split(sub, test_size=0.5, random_state=s)
    _, te_s = train_test_split(sub, test_size=0.5, random_state=s,
                               stratify=yb[sub])
    fr_rand.append(yb[te_r].mean()); fr_strat.append(yb[te_s].mean())
print(f"INFO  malignant fraction in the 60-example subsample = "
      f"{yb[sub].mean():.3f}; over 200 splits the test-half fraction ranges "
      f"{min(fr_rand):.2f}-{max(fr_rand):.2f} unstratified vs "
      f"{min(fr_strat):.2f}-{max(fr_strat):.2f} stratified")
print("All hand computations verified.")
