"""PCA in two views (the L12 seed).

View 1: projection onto the max-variance direction (2-D toy data, residuals
shown). View 2: reconstruction error / cumulative explained variance on the
housing numeric features. Figures: 06_pca_projection.pdf, 06_pca_variance.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from _style import BLUE, VERMILLION, GREEN, GRAY, save

rng = np.random.default_rng(42)

# --- view 1: 2-D correlated cloud, PC1, projections --------------------------
C = np.array([[1.0, 0.85], [0.85, 1.0]])
X = rng.multivariate_normal([0, 0], C, 120)
pca = PCA(n_components=2).fit(X)
w1 = pca.components_[0]                       # unit vector, max-variance dir
proj = X @ w1[:, None] * w1[None, :]          # reconstruction from 1 component
evr = pca.explained_variance_ratio_
print(f"toy 2-D data: PC1 explains {evr[0]*100:.1f}% of variance, "
      f"PC2 {evr[1]*100:.1f}%")
mse_rec = ((X - proj) ** 2).sum(axis=1).mean()
print(f"mean squared reconstruction error with 1 component = {mse_rec:.4f} "
      f"= variance NOT captured ({pca.explained_variance_[1]:.4f} up to m/(m-1))")

fig, ax = plt.subplots(figsize=(4.6, 4.2))
for a, b in zip(X, proj):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=GRAY, lw=0.5, alpha=0.5)
ax.scatter(*X.T, s=14, color=BLUE, zorder=3, label="data")
ax.scatter(*proj.T, s=10, color=VERMILLION, zorder=4, label="projection on PC1")
t = np.linspace(-3.4, 3.4, 2)
ax.plot(t * w1[0], t * w1[1], color=GREEN, lw=2, label="PC1 direction")
ax.set_aspect("equal"); ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
save(fig, "06_pca_projection.pdf")

# --- view 2: explained variance on housing numerics --------------------------
housing = pd.read_csv(Path(__file__).parent / "data/housing/housing.csv")
Xh = housing.drop(columns=["ocean_proximity", "median_house_value"]).dropna()
Z = StandardScaler().fit_transform(Xh)
pca = PCA().fit(Z)
cum = np.cumsum(pca.explained_variance_ratio_)
print("housing numerics: cumulative explained variance:",
      np.round(cum, 3))

fig, ax = plt.subplots(figsize=(4.8, 3.2))
k = np.arange(1, len(cum) + 1)
ax.bar(k, pca.explained_variance_ratio_, color=BLUE, label="per component")
ax.plot(k, cum, "o-", color=VERMILLION, ms=4, label="cumulative")
ax.axhline(0.95, color=GRAY, ls=":", lw=1.5)
ax.set_xlabel("number of components $k$")
ax.set_ylabel("explained variance ratio")
ax.legend(fontsize=9)
ax.set_title("housing (8 numeric features), standardized")
save(fig, "06_pca_variance.pdf")
