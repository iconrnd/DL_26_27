"""Shared figure style for the DL lecture series (imported by every code/ script).

Colors: Okabe-Ito palette (colorblind-safe), fixed assignment order.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# fixed categorical order — never cycled differently across figures
BLUE = "#0072B2"       # primary series / model
VERMILLION = "#D55E00" # contrast series / validation, error
GREEN = "#009E73"      # third series / truth curves
ORANGE = "#E69F00"     # fourth series
PURPLE = "#CC79A7"
GRAY = "#999999"       # reference lines, noise floors

GFX = Path(__file__).resolve().parent.parent / "gfx"
GFX.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
})


def save(fig, name):
    out = GFX / name
    fig.savefig(out)
    print(f"[saved] {out}")
