"""Logistic regression as the one-neuron classifier, and softmax regression
on the three Iris classes.

Figures: 07_logit_curve.pdf (probability vs petal width, Geron's example),
07_softmax_regions.pdf (three-class decision regions). Accuracies printed.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from _style import BLUE, VERMILLION, GREEN, GRAY, save

iris = load_iris()

# --- 1-D logistic regression: P(virginica) vs petal width --------------------
Xw = iris.data[:, [3]]
yv = (iris.target == 2).astype(int)
clf = LogisticRegression().fit(Xw, yv)
xs = np.linspace(0, 3, 300)[:, None]
proba = clf.predict_proba(xs)[:, 1]
boundary = xs[np.argmin(np.abs(proba - 0.5))][0]
print(f"1-D logistic: decision boundary at petal width = {boundary:.2f} cm")

fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.plot(xs, proba, color=BLUE, label=r"$\hat p$(virginica)")
ax.plot(xs, 1 - proba, color=VERMILLION, ls="--",
        label=r"$\hat p$(not virginica)")
ax.axvline(boundary, color=GRAY, ls=":", lw=1.5)
ax.scatter(Xw[yv == 1], np.ones(yv.sum()) * 1.02, marker="^", s=12,
           color=BLUE)
ax.scatter(Xw[yv == 0], np.zeros((yv == 0).sum()) - 0.02, marker="v", s=12,
           color=VERMILLION)
ax.set_xlabel("petal width [cm]")
ax.set_ylabel("probability")
ax.legend(fontsize=9, loc="center right")
save(fig, "07_logit_curve.pdf")

# --- softmax regression on all three classes, two features -------------------
X = iris.data[:, [2, 3]]                      # petal length, petal width
y = iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=42)
soft = make_pipeline(StandardScaler(), LogisticRegression())
soft.fit(X_tr, y_tr)
print(f"softmax regression on Iris (3 classes, 2 features): "
      f"test accuracy = {soft.score(X_te, y_te):.3f}")

xx, yy = np.meshgrid(np.linspace(0.5, 7.5, 400), np.linspace(-0.2, 3, 400))
Z = soft.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.contourf(xx, yy, Z, alpha=0.15, levels=[-0.5, 0.5, 1.5, 2.5],
            colors=[BLUE, VERMILLION, GREEN])
for k, (name, c, mk) in enumerate([("setosa", BLUE, "o"),
                                   ("versicolor", VERMILLION, "s"),
                                   ("virginica", GREEN, "^")]):
    ax.scatter(*X[y == k].T, color=c, s=16, marker=mk, label=name)
ax.set_xlabel("petal length [cm]")
ax.set_ylabel("petal width [cm]")
ax.legend(fontsize=9, loc="upper left")
save(fig, "07_softmax_regions.pdf")
