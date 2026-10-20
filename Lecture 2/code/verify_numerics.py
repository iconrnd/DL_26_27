"""Verification of every hand computation typeset in the Lecture 2 deck.

Run:  python verify_numerics.py  ->  one PASS line per check.
"""
import numpy as np


def check(name, got, want, tol=1e-9):
    ok = np.allclose(got, want, atol=tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {np.round(got, 6)}, "
          f"hand value {want}")
    assert ok


# 1 ---- one perceptron update by hand ----------------------------------------
w, b, eta = np.array([0.5, -1.0]), 0.0, 0.1
x, y = np.array([1.0, 2.0]), 1
z = w @ x + b                                # 0.5 - 2 = -1.5
check("perceptron score z", z, -1.5)
yhat = 1 if z >= 0 else -1                   # -1: an error
update = eta * (y - yhat)                    # 0.1 * 2 = 0.2
check("update magnitude", update, 0.2)
check("new w", w + update * x, [0.7, -0.6])
check("new b", b + update, 0.2)

# 2 ---- GD iterates on J(w) = w^2 --------------------------------------------
def iterates(w0, eta, n):
    ws = [w0]
    for _ in range(n):
        ws.append(ws[-1] * (1 - 2 * eta))
    return ws

check("eta=0.1 iterates", iterates(1.0, 0.1, 4),
      [1.0, 0.8, 0.64, 0.512, 0.4096])
check("eta=1.1 iterates", iterates(1.0, 1.1, 4),
      [1.0, -1.2, 1.44, -1.728, 2.0736])

# 3 ---- one GD step on MSE (yhat = w x, data {(1,2),(2,3)}) ------------------
X = np.array([1.0, 2.0]); Y = np.array([2.0, 3.0])
w0, eta = 0.0, 0.1
grad0 = ((w0 * X - Y) * X).mean()            # (-2*1 + -3*2)/2 = -4
check("MSE gradient at w=0", grad0, -4.0)
w1 = w0 - eta * grad0
check("w after step 1", w1, 0.4)
grad1 = ((w1 * X - Y) * X).mean()            # (-1.6 - 4.4)/2 = -3
check("MSE gradient at w=0.4", grad1, -3.0)
check("w after step 2", w1 - eta * grad1, 0.7)

# 4 ---- softmax + CE on z = (2, 0, 1) ----------------------------------------
z3 = np.array([2.0, 0.0, 1.0])
p3 = np.exp(z3) / np.exp(z3).sum()
check("softmax(2,0,1)", p3, [0.665241, 0.090031, 0.244728], tol=5e-7)
check("CE for true class 3", -np.log(p3[2]), 1.407606, tol=5e-7)
# shift invariance: softmax(z + 100) identical
p_shift = np.exp(z3 + 100 - (z3 + 100).max())
check("shift invariance", p_shift / p_shift.sum(), p3)

# 5 ---- normal equation by hand on {(0,1),(1,2),(2,2)} -----------------------
Xh = np.array([[1, 0], [1, 1], [1, 2]], dtype=float)
yh = np.array([1.0, 2.0, 2.0])
check("XtX", Xh.T @ Xh, [[3, 3], [3, 5]])
check("Xty", Xh.T @ yh, [5, 6])
check("theta = (7/6, 1/2)", np.linalg.solve(Xh.T @ Xh, Xh.T @ yh),
      [7 / 6, 0.5])

# 6 ---- sigmoid / BCE spot values --------------------------------------------
sig = 1 / (1 + np.exp(-0.5))
check("sigma(0.5)", sig, 0.622459, tol=5e-7)
check("BCE(y=1, p=sigma(0.5))", -np.log(sig), 0.474077, tol=5e-7)

print("All hand computations verified.")
