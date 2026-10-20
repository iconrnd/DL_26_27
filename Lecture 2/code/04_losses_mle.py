"""Losses from maximum likelihood: sigmoid/BCE plots and the softmax
numerical-stability demonstration (naive overflow vs. max-shifted).

Figures: 04_sigmoid_bce.pdf. The stability demo prints frozen output.
"""
import numpy as np
import matplotlib.pyplot as plt
from _style import BLUE, VERMILLION, save

# --- sigmoid and the two branches of BCE -------------------------------------
z = np.linspace(-6, 6, 300)
sig = 1 / (1 + np.exp(-z))
p = np.linspace(1e-3, 1 - 1e-3, 300)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
axes[0].plot(z, sig, color=BLUE)
axes[0].axhline(0.5, color="gray", ls=":", lw=1)
axes[0].set_xlabel("$z$")
axes[0].set_title(r"$\sigma(z) = 1/(1+e^{-z})$")
axes[1].plot(p, -np.log(p), color=BLUE, label=r"$-\ln \hat p$  ($y=1$)")
axes[1].plot(p, -np.log(1 - p), color=VERMILLION,
             label=r"$-\ln(1-\hat p)$  ($y=0$)")
axes[1].set_xlabel(r"$\hat p$")
axes[1].set_title("binary cross-entropy")
axes[1].legend(fontsize=9)
save(fig, "04_sigmoid_bce.pdf")

# --- softmax numerical stability ---------------------------------------------
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()


def softmax_stable(z):
    e = np.exp(z - z.max())
    return e / e.sum()


z_big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="warn", invalid="warn"):
    naive = softmax_naive(z_big)
print("logits:", z_big)
print("naive softmax:  ", naive, " (exp overflowed to inf -> nan)")
print("shifted softmax:", np.round(softmax_stable(z_big), 4))

# --- the worked softmax numeric of the slides, replayed ----------------------
z3 = np.array([2.0, 0.0, 1.0])
p3 = softmax_stable(z3)
print(f"softmax(2,0,1) = {np.round(p3, 3)};  "
      f"CE for true class 3 = -ln p_3 = {-np.log(p3[2]):.4f}")
