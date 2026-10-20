"""Ridge vs lasso: coefficient paths over the regularization strength on the
housing numeric features (standardized, train part only).

Figure: 06_coef_paths.pdf; sparsity counts printed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from _style import BLUE, save

housing = pd.read_csv(Path(__file__).parent / "data/housing/housing.csv")
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
train, _ = train_test_split(housing, test_size=0.2,
                            stratify=housing["income_cat"], random_state=42)
train = train.drop(columns=["income_cat", "ocean_proximity"])
X = train.drop(columns="median_house_value")
y = train["median_house_value"].to_numpy()
feats = X.columns.tolist()
Xs = make_pipeline(SimpleImputer(strategy="median"),
                   StandardScaler()).fit_transform(X)

alphas = np.logspace(-1, 5, 40)
coef_r = np.array([Ridge(alpha=a).fit(Xs, y).coef_ for a in alphas])
coef_l = np.array([Lasso(alpha=a, max_iter=50000).fit(Xs, y).coef_
                   for a in alphas])

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
for ax, C, ttl in [(axes[0], coef_r, "ridge: smooth shrinkage"),
                   (axes[1], coef_l, "lasso: features zeroed one by one")]:
    for j, f in enumerate(feats):
        ax.plot(alphas, C[:, j], lw=1.6,
                label=f if ax is axes[1] else None)
    ax.set_xscale("log")
    ax.set_xlabel(r"regularization strength $\lambda$")
    ax.set_title(ttl)
axes[0].set_ylabel("standardized coefficient")
axes[1].legend(fontsize=6.5, loc="upper right")
save(fig, "06_coef_paths.pdf")

for a in (1e2, 1e3, 1e4):
    nz = (np.abs(Lasso(alpha=a, max_iter=50000).fit(Xs, y).coef_) > 1e-6).sum()
    print(f"lasso, lambda = {a:>7.0f}: {nz}/{len(feats)} nonzero coefficients")
print(f"ridge, lambda =   10000: "
      f"{(np.abs(Ridge(alpha=1e4).fit(Xs, y).coef_) > 1e-6).sum()}"
      f"/{len(feats)} nonzero (shrinks, never zeroes)")
