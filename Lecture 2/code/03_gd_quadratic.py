"""Gradient descent on J(w) = w^2: the learning-rate pathologies, with the
exact iterates that the slides quote (hand-verified in verify_numerics.py).

Update: w <- w - eta * J'(w) = w - 2*eta*w = (1 - 2*eta) * w.
Figure: 03_lr_panel.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from _style import BLUE, VERMILLION, GREEN, GRAY, save


def iterates(w0, eta, n):
    w = [w0]
    for _ in range(n):
        w.append(w[-1] - eta * 2 * w[-1])
    return np.array(w)


w_small = iterates(1.0, 0.1, 4)
w_good = iterates(1.0, 0.4, 4)
w_big = iterates(1.0, 1.1, 4)
print("eta=0.1 :", np.round(w_small, 4), "(crawls)")
print("eta=0.4 :", np.round(w_good, 4), "(converges fast)")
print("eta=1.1 :", np.round(w_big, 4), "(diverges, sign-flipping)")

ws = np.linspace(-2.6, 2.6, 300)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
for ax, w, ttl, c in [(axes[0], w_small, r"$\eta=0.1$: slow", BLUE),
                      (axes[1], w_good, r"$\eta=0.4$: fast", GREEN),
                      (axes[2], w_big, r"$\eta=1.1$: divergence", VERMILLION)]:
    ax.plot(ws, ws ** 2, color=GRAY, lw=1.5)
    ax.plot(w, w ** 2, "o-", color=c, ms=5)
    for i, wi in enumerate(w[:4]):
        ax.annotate(str(i), (wi, wi ** 2), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
    ax.set_title(ttl)
    ax.set_xlabel("$w$")
    ax.set_ylim(-0.3, 6.5)
axes[0].set_ylabel("$J(w)=w^2$")
save(fig, "03_lr_panel.pdf")
