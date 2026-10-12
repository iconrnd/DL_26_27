"""The leakage demonstration: feature selection BEFORE the split vs inside a
Pipeline. Pure-noise data (no real signal): honest accuracy must be ~0.5.

n = 100 examples, p = 5000 i.i.d. N(0,1) features, random labels.
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)
n, p, k = 100, 5000, 20
X = rng.normal(size=(n, p))
y = rng.integers(0, 2, n)              # coin flips: NO relation to X
cv = StratifiedKFold(5, shuffle=True, random_state=42)
print(f"data: {n} examples, {p} pure-noise features, random labels; "
      f"select k={k} best, logistic regression, 5-fold CV")

# WRONG: select features on ALL data, then cross-validate the classifier
X_sel = SelectKBest(f_classif, k=k).fit_transform(X, y)
wrong = cross_val_score(LogisticRegression(max_iter=5000), X_sel, y, cv=cv)
print(f"WRONG  (select on all data, then CV):  accuracy = {wrong.mean():.3f}")

# RIGHT: selection inside the pipeline -> refit on each fold's train part only
pipe = make_pipeline(SelectKBest(f_classif, k=k),
                     LogisticRegression(max_iter=5000))
right = cross_val_score(pipe, X, y, cv=cv)
print(f"RIGHT  (selection inside the CV loop): accuracy = {right.mean():.3f}")
print("true attainable accuracy: 0.5 (labels are coin flips)")
