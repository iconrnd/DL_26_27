"""Adaline (adaptive linear neuron) with full-batch gradient descent.

The pivot: minimize MSE on the continuous score z = w.x + b instead of
updating on thresholded errors. Shows the two-learning-rate failure pair
and the feature-scaling rescue (Raschka Ch. 2's arc).
Figures: 02_lr_curves.pdf, 02_scaling.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from _style import BLUE, VERMILLION, GREEN, save


class AdalineGD:
    """Linear unit trained by full-batch gradient descent on MSE(z, y)."""

    def __init__(self, eta=0.01, n_epochs=30):
        self.eta, self.n_epochs = eta, n_epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1]); self.b = 0.0
        self.losses_ = []
        for _ in range(self.n_epochs):
            z = X @ self.w + self.b            # forward
            err = y - z
            loss = (err ** 2).mean() / 2       # loss
            grad_w = -X.T @ err / len(y)       # backward
            self.w -= self.eta * grad_w        # step
            self.b += self.eta * err.mean()
            self.losses_.append(loss)
        return self


iris = load_iris()
mask = iris.target < 2
X = iris.data[mask][:, [0, 2]]
y = np.where(iris.target[mask] == 0, -1.0, 1.0)

# --- learning-rate pathology pair (raw features) -----------------------------
hi = AdalineGD(eta=0.1, n_epochs=20).fit(X, y)
lo = AdalineGD(eta=0.0001, n_epochs=20).fit(X, y)
print(f"eta=0.1 (raw): loss epoch 1 = {hi.losses_[0]:.2f}, "
      f"epoch 20 = {hi.losses_[-1]:.2e}  (diverges)")
print(f"eta=0.0001 (raw): loss epoch 1 = {lo.losses_[0]:.4f}, "
      f"epoch 20 = {lo.losses_[-1]:.4f}  (barely moves)")

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
axes[0].plot(range(1, 21), np.log10(hi.losses_), "o-", color=VERMILLION, ms=4)
axes[0].set_title(r"$\eta = 0.1$: divergence")
axes[0].set_ylabel(r"$\log_{10}$ MSE")
axes[1].plot(range(1, 21), lo.losses_, "o-", color=BLUE, ms=4)
axes[1].set_title(r"$\eta = 0.0001$: barely moves")
axes[1].set_ylabel("MSE")
for ax in axes:
    ax.set_xlabel("epoch")
save(fig, "02_lr_curves.pdf")

# --- the rescue: standardize, then a sane eta works --------------------------
Xs = (X - X.mean(axis=0)) / X.std(axis=0)
ok = AdalineGD(eta=0.1, n_epochs=20).fit(Xs, y)
print(f"eta=0.1 (standardized): loss epoch 1 = {ok.losses_[0]:.4f}, "
      f"epoch 20 = {ok.losses_[-1]:.4f}  (converges)")

fig, ax = plt.subplots(figsize=(4.8, 3.0))
ax.plot(range(1, 21), ok.losses_, "o-", color=GREEN, ms=4,
        label=r"standardized, $\eta = 0.1$")
ax.plot(range(1, 21), lo.losses_, "o-", color=BLUE, ms=4,
        label=r"raw, $\eta = 0.0001$")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE")
ax.legend(fontsize=9)
ax.set_title("the feature-scaling rescue")
save(fig, "02_scaling.pdf")
