"""Linear regression two ways (normal equation vs gradient descent), the
housing continuity run, and polynomial learning curves.

Figures: 05_poly_lc.pdf. All decision numbers printed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit
from _style import BLUE, VERMILLION, GRAY, save

rng = np.random.default_rng(42)

# --- normal equation vs GD on one synthetic problem --------------------------
m = 100
x = 2 * rng.uniform(size=m)
y = 4 + 3 * x + rng.normal(0, 1, m)
X = np.c_[np.ones(m), x]                      # design matrix with bias column
theta_ne = np.linalg.solve(X.T @ X, X.T @ y)  # normal equation
theta = np.zeros(2)                            # batch GD on the same objective
eta = 0.1
for _ in range(2000):
    theta -= eta * X.T @ (X @ theta - y) / m
print(f"normal equation: theta = {np.round(theta_ne, 4)}")
print(f"batch GD (2000 steps): theta = {np.round(theta, 4)}  (identical)")

# --- the hand numeric of the slides, replayed --------------------------------
Xh = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
yh = np.array([1.0, 2.0, 2.0])
print("hand normal equation on {(0,1),(1,2),(2,2)}:",
      np.linalg.solve(Xh.T @ Xh, Xh.T @ yh), "= (7/6, 1/2)")

# --- housing continuity: L1's pipeline, linear model (frozen L1 number) ------
housing = pd.read_csv(Path(__file__).parent / "data/housing/housing.csv")
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
from sklearn.model_selection import train_test_split
train, _ = train_test_split(housing, test_size=0.2,
                            stratify=housing["income_cat"], random_state=42)
train = train.drop(columns="income_cat")
X_tr = train.drop(columns="median_house_value")
y_tr = train["median_house_value"]
num = X_tr.select_dtypes(include=np.number).columns.tolist()
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
     num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]),
])
pipe = make_pipeline(prep, LinearRegression())
rmse = -cross_val_score(pipe, X_tr, y_tr, cv=5,
                        scoring="neg_root_mean_squared_error")
print(f"housing, linear regression, 5-fold CV RMSE = {rmse.mean():.0f} "
      f"+- {rmse.std():.0f}  (L1 frozen value: 68322 +- 354)")

# --- polynomial learning curves (Geron's quadratic synthetic) ----------------
mq = 400
xq = 6 * rng.uniform(size=mq) - 3
yq = 0.5 * xq ** 2 + xq + 2 + rng.normal(0, 1, mq)
cv = ShuffleSplit(n_splits=30, test_size=0.25, random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), sharey=True)
for ax, deg, ttl in [(axes[0], 1, "degree 1: high bias"),
                     (axes[1], 10, "degree 10: high variance")]:
    model = make_pipeline(PolynomialFeatures(deg), StandardScaler(),
                          LinearRegression())
    n, tr, va = learning_curve(model, xq[:, None], yq,
                               train_sizes=np.linspace(0.05, 1.0, 12), cv=cv,
                               scoring="neg_root_mean_squared_error")
    ax.plot(n, -tr.mean(axis=1), "o-", color=BLUE, ms=4, label="training RMSE")
    ax.plot(n, -va.mean(axis=1), "o-", color=VERMILLION, ms=4,
            label="validation RMSE")
    ax.axhline(1.0, color=GRAY, ls=":", lw=1.5, label=r"noise floor $\sigma$")
    ax.set_xlabel("training-set size $m$")
    ax.set_title(ttl)
    ax.set_ylim(0, 2.6)
axes[0].set_ylabel("RMSE")
axes[0].legend(fontsize=9)
save(fig, "05_poly_lc.pdf")
print("polynomial learning curves written (truth: quadratic, sigma = 1)")
