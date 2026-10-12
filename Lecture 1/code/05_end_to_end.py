"""The end-to-end arc (Geron Ch. 2, compressed): California housing.

Uses Geron's housing.csv (frozen in code/data/housing/): 20640 districts,
missing total_bedrooms values, one categorical column (ocean_proximity).
Figures: 05_hist.pdf, 05_geo.pdf. All decision numbers printed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from _style import save

housing = pd.read_csv(Path(__file__).parent / "data/housing/housing.csv")
print(f"dataset: {housing.shape[0]} districts x {housing.shape[1]} columns; "
      f"missing total_bedrooms: {housing['total_bedrooms'].isna().sum()}")

# ---- split FIRST (stratified on binned income), explore the train part only
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
train, test = train_test_split(housing, test_size=0.2,
                               stratify=housing["income_cat"], random_state=42)
for df in (train, test):
    df.drop(columns="income_cat", inplace=True)
print(f"split: {len(train)} train / {len(test)} test (stratified on income)")

# ---- explore (train only) ---------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.6))
for ax, col in zip(axes.ravel(),
                   ["median_income", "housing_median_age",
                    "median_house_value", "population"]):
    train[col].hist(bins=50, ax=ax, color="#0072B2")
    ax.set_title(col, fontsize=11)
    if train[col].max() > 10_000:   # compress wide tick labels: 500000 -> 500k
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1000:g}k"))
fig.tight_layout()
save(fig, "05_hist.pdf")

fig, ax = plt.subplots(figsize=(5.6, 4.4))
sc = ax.scatter(train["longitude"], train["latitude"], s=train["population"]/80,
                c=train["median_house_value"], cmap="viridis", alpha=0.35)
fig.colorbar(sc, label="median house value [$]")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
save(fig, "05_geo.pdf")
corr = train.corr(numeric_only=True)["median_house_value"].sort_values(
    ascending=False)
print("correlation with target:\n", corr.round(2).to_string())

# ---- prepare: the leakage-proof preprocessing + model pipeline --------------
X_tr = train.drop(columns="median_house_value")
y_tr = train["median_house_value"]
num = X_tr.select_dtypes(include=np.number).columns.tolist()
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
     num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]),
])

# ---- shortlist with 5-fold CV (RMSE) ----------------------------------------
models = {"dummy (mean)": DummyRegressor(),
          "linear regression": LinearRegression(),
          "decision tree": DecisionTreeRegressor(random_state=42),
          "random forest": RandomForestRegressor(random_state=42, n_jobs=-1)}
for name, model in models.items():
    pipe = make_pipeline(prep, model)
    rmse = -cross_val_score(pipe, X_tr, y_tr, cv=5,
                            scoring="neg_root_mean_squared_error")
    fit_rmse = np.sqrt(np.mean(
        (pipe.fit(X_tr, y_tr).predict(X_tr) - y_tr) ** 2))
    print(f"{name:18s}  CV RMSE = {rmse.mean():7.0f} +- {rmse.std():4.0f}   "
          f"train RMSE = {fit_rmse:7.0f}")

# ---- tune the winner (randomized search, small honest budget) ---------------
pipe = make_pipeline(prep, RandomForestRegressor(random_state=42, n_jobs=-1))
search = RandomizedSearchCV(
    pipe,
    {"randomforestregressor__max_features": stats.randint(2, 12),
     "randomforestregressor__n_estimators": stats.randint(100, 300)},
    n_iter=8, cv=3, scoring="neg_root_mean_squared_error", random_state=42)
search.fit(X_tr, y_tr)
best = {k.split("__")[1]: v for k, v in search.best_params_.items()}
print(f"randomized search (8 candidates, 3-fold): best {best}, "
      f"CV RMSE = {-search.best_score_:.0f}")

# ---- evaluate ONCE on the test set ------------------------------------------
X_te = test.drop(columns="median_house_value")
y_te = test["median_house_value"]
err2 = (search.best_estimator_.predict(X_te) - y_te) ** 2
rmse_te = np.sqrt(err2.mean())
lo, hi = np.sqrt(stats.t.interval(0.95, len(err2) - 1, loc=err2.mean(),
                                  scale=stats.sem(err2)))
print(f"FINAL test RMSE = {rmse_te:.0f}  (95% CI [{lo:.0f}, {hi:.0f}])")
