"""Classification metrics on Breast Cancer Wisconsin + the 4-example hand ROC.

Model: StandardScaler + LogisticRegression in a Pipeline (fit on train only).
Figures: 03_pr_curve.pdf, 03_roc_curve.pdf, 03_roc_hand.pdf, 03_calibration.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, precision_recall_curve, roc_curve, auc,
                             roc_auc_score)
from _style import BLUE, VERMILLION, GRAY, save

# label convention: positive class = malignant. sklearn stores malignant as 0,
# so relabel: y = 1 for malignant.
data = load_breast_cancer()
X, y = data.data, (data.target == 0).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
print(f"train {X_tr.shape[0]} / test {X_te.shape[0]} examples, "
      f"positives (malignant) in test: {y_te.sum()}/{y_te.size}")

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
clf.fit(X_tr, y_tr)
scores = clf.decision_function(X_te)
y_hat = (scores > 0).astype(int)

cm = confusion_matrix(y_te, y_hat)         # rows: true 0/1, cols: pred 0/1
tn, fp, fn, tp = cm.ravel()
print(f"confusion matrix [ [TN FP] [FN TP] ] = [[{tn} {fp}] [{fn} {tp}]]")
print(f"accuracy  = {(tp+tn)/(tp+tn+fp+fn):.4f}")
print(f"precision = {precision_score(y_te, y_hat):.4f}")
print(f"recall    = {recall_score(y_te, y_hat):.4f}")
print(f"F1        = {f1_score(y_te, y_hat):.4f}")

# --- PR curve ----------------------------------------------------------------
prec, rec, _ = precision_recall_curve(y_te, scores)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(rec, prec, color=BLUE)
ax.scatter([recall_score(y_te, y_hat)], [precision_score(y_te, y_hat)],
           color=VERMILLION, zorder=3, label="threshold 0")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc="lower left")
ax.set_title("precision–recall (breast cancer, test set)")
save(fig, "03_pr_curve.pdf")

# --- ROC curve ---------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_te, scores)
A = roc_auc_score(y_te, scores)
print(f"test-set ROC AUC = {A:.4f}")
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(fpr, tpr, color=BLUE, label=f"logistic model, AUC = {A:.3f}")
ax.plot([0, 1], [0, 1], color=GRAY, ls=":", label="chance, AUC = 0.5")
ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
ax.legend(fontsize=9, loc="lower right")
ax.set_title("ROC (breast cancer, test set)")
save(fig, "03_roc_curve.pdf")

# --- the 4-example ROC, by hand then by sklearn ------------------------------
y4 = np.array([1, 0, 1, 0])
s4 = np.array([0.9, 0.7, 0.6, 0.2])
fpr4, tpr4, thr4 = roc_curve(y4, s4)
A4 = roc_auc_score(y4, s4)
print("hand-ROC examples: labels", y4, "scores", s4)
print("sklearn ROC points (FPR, TPR):", list(zip(fpr4.round(2), tpr4.round(2))))
print(f"sklearn AUC = {A4:.4f}  (hand value: 0.75)")

# --- calibration: reliability diagram + Brier score ------------------------
from sklearn.calibration import calibration_curve

proba = clf.predict_proba(X_te)[:, 1]
frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=5,
                                        strategy="quantile")
brier = np.mean((proba - y_te) ** 2)
print("calibration (5 quantile bins): mean predicted ->"
      " observed positive fraction")
for mp, fp_ in zip(mean_pred, frac_pos):
    print(f"  {mp:.3f} -> {fp_:.3f}")
print(f"Brier score = mean (p_hat - y)^2 = {brier:.4f}")

fig, ax = plt.subplots(figsize=(4.4, 3.6))
ax.plot([0, 1], [0, 1], color=GRAY, ls=":", label="perfectly calibrated")
ax.plot(mean_pred, frac_pos, "o-", color=BLUE, ms=5, label="logistic model")
ax.set_xlabel("mean predicted probability (bin)")
ax.set_ylabel("observed positive fraction")
ax.legend(fontsize=9, loc="upper left")
ax.set_title("reliability diagram (test set)")
save(fig, "03_calibration.pdf")

fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.step(fpr4, tpr4, where="post", color=BLUE)
ax.plot(fpr4, tpr4, "o", color=VERMILLION, ms=6, zorder=3)
ax.plot([0, 1], [0, 1], color=GRAY, ls=":")
for x_, y_, t_ in [(0, 0, "t>0.9"), (0, .5, "t=0.8"), (.5, .5, "t=0.65"),
                   (.5, 1, "t=0.4"), (1, 1, "t=0.1")]:
    ax.annotate(t_, (x_, y_), textcoords="offset points", xytext=(8, -4),
                fontsize=9)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("the 4-example ROC (AUC = 0.75)")
save(fig, "03_roc_hand.pdf")
