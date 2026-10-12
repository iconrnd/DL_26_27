"""Bias-variance decomposition, verified by simulation.

Truth y = sin(2 pi x) + eps, eps ~ N(0, 0.3^2). Estimator: degree-d polynomial
fit on a fresh training set of size 30, repeated R = 2000 times. At each test x*
we estimate bias^2, variance, and check that
    E[(y - fhat)^2] = bias^2 + variance + sigma^2
holds (averaged over a fixed test grid). Figure: 02_bias_variance.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from _style import BLUE, VERMILLION, GREEN, GRAY, ORANGE, save

rng = np.random.default_rng(42)
SIGMA, N, R = 0.3, 30, 2000
xg = np.linspace(0.02, 0.98, 60)          # fixed test grid
f_true = np.sin(2 * np.pi * xg)


def fit_predict(degree):
    """R independent training sets -> R prediction vectors on the grid."""
    preds = np.empty((R, xg.size))
    for r in range(R):
        x = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x) + rng.normal(0, SIGMA, N)
        coef = np.polyfit(x, y, degree)
        preds[r] = np.polyval(coef, xg)
    return preds


degrees = np.arange(1, 8)   # degrees >= 8 explode numerically on [0,1] polyfit
bias2, var, direct = [], [], []
for d in degrees:
    P = fit_predict(d)
    fbar = P.mean(axis=0)                      # E_D[fhat(x*)]
    b2 = ((fbar - f_true) ** 2).mean()         # avg bias^2 over grid
    v = P.var(axis=0).mean()                   # avg variance over grid
    # direct estimate of expected error: fresh noisy y* for every replicate
    ystar = f_true + rng.normal(0, SIGMA, P.shape)
    mse = ((ystar - P) ** 2).mean()
    bias2.append(b2); var.append(v); direct.append(mse)
bias2, var, direct = map(np.array, (bias2, var, direct))
total = bias2 + var + SIGMA**2

print(" d   bias^2  variance  sigma^2   sum     direct MSE")
for d, b, v, t, m in zip(degrees, bias2, var, total, direct):
    print(f"{d:2d}   {b:.4f}   {v:.4f}   {SIGMA**2:.4f}   {t:.4f}   {m:.4f}")
gap = (np.abs(total - direct) / direct).max()
print(f"max relative |sum - direct| over degrees = {gap:.4f}  (simulation closure)")
assert gap < 0.02, "decomposition failed to close within tolerance"
print("PASS: bias^2 + variance + noise matches direct MSE estimate")

fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.plot(degrees, bias2, "o-", color=BLUE, ms=4, label="bias$^2$")
ax.plot(degrees, var, "o-", color=VERMILLION, ms=4, label="variance")
ax.axhline(SIGMA**2, color=GRAY, ls=":", lw=1.5, label="noise $\\sigma^2$")
ax.plot(degrees, total, "-", color=GREEN, lw=2.5, alpha=0.85,
        label="bias$^2$+var+$\\sigma^2$")
ax.plot(degrees, direct, "s", color=ORANGE, ms=5, mfc="none", mew=1.8,
        label="direct $\\mathbb{E}[(y-\\hat f)^2]$")
ax.set_xlabel("polynomial degree (capacity)")
ax.set_ylabel("error (test point average)")
ax.legend(fontsize=9, ncol=2)
save(fig, "02_bias_variance.pdf")
