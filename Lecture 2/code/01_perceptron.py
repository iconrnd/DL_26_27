"""The perceptron from scratch (NumPy), trained on two Iris classes.

Follows Raschka Ch. 2's arc in course notation: labels y in {-1,+1},
score z = w.x + b, update w += eta*(y - yhat)*x on errors only.
Figures: 01_errors.pdf, 01_regions.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from _style import BLUE, VERMILLION, GRAY, save

rng = np.random.default_rng(42)


class Perceptron:
    """Rosenblatt perceptron: linear threshold unit with error-driven updates."""

    def __init__(self, eta=0.1, n_epochs=10):
        self.eta, self.n_epochs = eta, n_epochs

    def score(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.score(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_ = []
        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))
                self.w += update * xi          # 0 when prediction correct
                self.b += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


# --- data: Iris setosa vs versicolor, 2 features (sepal len, petal len) ------
iris = load_iris()
mask = iris.target < 2
X = iris.data[mask][:, [0, 2]]
y = np.where(iris.target[mask] == 0, -1, 1)
print(f"data: {X.shape[0]} examples, 2 features, labels -1 (setosa) / +1 (versicolor)")

p = Perceptron(eta=0.1, n_epochs=10).fit(X, y)
print("errors per epoch:", p.errors_)
print(f"final w = {p.w}, b = {p.b:.2f}")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(range(1, len(p.errors_) + 1), p.errors_, "o-", color=BLUE)
ax.set_xlabel("epoch")
ax.set_ylabel("updates (errors)")
ax.set_title("perceptron on Iris: errors per epoch")
save(fig, "01_errors.pdf")

# --- decision regions --------------------------------------------------------
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300),
                     np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
Z = p.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.contourf(xx, yy, Z, alpha=0.15, levels=[-2, 0, 2],
            colors=[BLUE, VERMILLION])
ax.scatter(*X[y == -1].T, color=BLUE, s=18, label="setosa ($y=-1$)")
ax.scatter(*X[y == 1].T, color=VERMILLION, s=18, marker="s",
           label="versicolor ($y=+1$)")
ax.set_xlabel("sepal length [cm]")
ax.set_ylabel("petal length [cm]")
ax.legend(fontsize=9, loc="upper left")
save(fig, "01_regions.pdf")

# --- the hand-worked single update (slide numeric), replayed -----------------
w, b, eta = np.array([0.5, -1.0]), 0.0, 0.1
x1, y1 = np.array([1.0, 2.0]), 1
z = w @ x1 + b
yhat = 1 if z >= 0 else -1
update = eta * (y1 - yhat)
w_new, b_new = w + update * x1, b + update
print(f"hand update: z = {z}, yhat = {yhat}, update = {update}, "
      f"w -> {w_new}, b -> {b_new}")
