"""Under/overfitting panel, validation curve, learning curves (all synthetic).

Truth: f(x) = cos(1.5 pi x) on [0,1]; noise sigma = 0.25; polynomial regression.
Figures: 01_fit_panel.pdf, 01_validation_curve.pdf, 01_learning_curves.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, validation_curve, ShuffleSplit
from _style import BLUE, VERMILLION, GREEN, GRAY, save

rng = np.random.default_rng(42)
SIGMA = 0.25


def f(x):
    return np.cos(1.5 * np.pi * x)


def sample(n):
    x = rng.uniform(0, 1, n)
    return x, f(x) + rng.normal(0, SIGMA, n)


def poly_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


# --- Figure 1: the fit panel (degrees 1 / 4 / 15 on n=30 points) -------------
x_tr, y_tr = sample(30)
xs = np.linspace(0, 1, 300)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
for ax, deg, label in zip(axes, [1, 4, 15],
                          ["degree 1 — underfit", "degree 4 — good fit",
                           "degree 15 — overfit"]):
    m = poly_model(deg).fit(x_tr[:, None], y_tr)
    ax.plot(xs, f(xs), color=GREEN, lw=1.5, ls="--", label="truth $f$")
    ax.scatter(x_tr, y_tr, s=18, color=GRAY, zorder=3, label="train data")
    ax.plot(xs, m.predict(xs[:, None]), color=BLUE, label="model $\\hat f$")
    ax.set_ylim(-1.9, 1.9)
    ax.set_title(label)
    ax.set_xlabel("$x$")
axes[0].set_ylabel("$y$")
axes[0].legend(fontsize=9, loc="lower left")
save(fig, "01_fit_panel.pdf")

# --- Figure 2: validation curve over degree (the capacity dial) --------------
X_big, y_big = sample(120)
degrees = np.arange(1, 16)
cv = ShuffleSplit(n_splits=40, test_size=0.25, random_state=42)
tr_sc, va_sc = validation_curve(poly_model(1), X_big[:, None], y_big,
                                param_name="polynomialfeatures__degree",
                                param_range=degrees, cv=cv,
                                scoring="neg_mean_squared_error")
tr_mse, va_mse = -tr_sc.mean(axis=1), -va_sc.mean(axis=1)
best = degrees[np.argmin(va_mse)]
print(f"validation curve: best degree = {best}, "
      f"val MSE = {va_mse.min():.4f}, noise floor sigma^2 = {SIGMA**2:.4f}")

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(degrees, tr_mse, "o-", color=BLUE, ms=4, label="training error")
ax.plot(degrees, va_mse, "o-", color=VERMILLION, ms=4, label="validation error")
ax.axhline(SIGMA**2, color=GRAY, ls=":", lw=1.5, label="noise floor $\\sigma^2$")
ax.axvline(best, color=GRAY, lw=0.8, alpha=0.6)
ax.set_yscale("log")
ax.set_xlabel("polynomial degree (capacity)")
ax.set_ylabel("MSE")
ax.legend(fontsize=10)
ax.set_title("the capacity dial")
save(fig, "01_validation_curve.pdf")

# --- Figure 3: learning curves for a high-bias vs high-variance model --------
X_lc, y_lc = sample(400)
sizes = np.linspace(0.06, 1.0, 12)
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharex=True)
for ax, deg, title in [(axes[0], 1, "degree 1: high bias"),
                       (axes[1], 15, "degree 15: high variance")]:
    n, tr, va = learning_curve(poly_model(deg), X_lc[:, None], y_lc,
                               train_sizes=sizes, cv=cv,
                               scoring="neg_mean_squared_error")
    ax.plot(n, -tr.mean(axis=1), "o-", color=BLUE, ms=4, label="training error")
    ax.plot(n, -va.mean(axis=1), "o-", color=VERMILLION, ms=4,
            label="validation error")
    ax.axhline(SIGMA**2, color=GRAY, ls=":", lw=1.5, label="noise floor $\\sigma^2$")
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("training-set size $m$")
    ax.set_title(title)
axes[0].set_ylabel("MSE")
axes[0].legend(fontsize=9)
save(fig, "01_learning_curves.pdf")
