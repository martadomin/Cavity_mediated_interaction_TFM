"""
Figure of the two-orbital split energies E(n_A)/N at N = 40 (data from julia/gp/split_energy.jl),
with E_GP(N)/N (dashed) and E_DMC/N (dotted) for each V0 as horizontal lines in the colour of the curve.
On this scale E_DMC and E_GP coincide (|E_DMC - E_GP|/N <= 0.01); the DMC run at -80 is an older tau = 8 run.

Usage (from the repository root, after julia julia/gp/split_energy.jl):
    python python/plots/plot_split.py
Output: final_imag/split_energy_N40.pdf
"""

import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 24,
    "text.usetex": True,
})

V0_SHOW = [-40.0, -80.0, -150.0]

with open("numpy_arrays_GP/split_energy_curves_N40.csv") as f:
    cols = f.readline().strip().split(",")
data = np.loadtxt("numpy_arrays_GP/split_energy_curves_N40.csv", delimiter=",", skiprows=1)
curves = {c: data[:, i] for i, c in enumerate(cols)}
table = np.genfromtxt("numpy_arrays_GP/split_energy.csv", delimiter=",", names=True)
x = curves["nA_over_N"]

fig, ax = plt.subplots(figsize=(13, 10))
for i, V in enumerate(V0_SHOW):
    col = f"C{i}"
    ax.plot(x, curves[f"E_V0{V:.0f}"], "o-", color=col, ms=6, lw=2, label=fr"$V_0={V:.0f}\,\varepsilon$: Fock $|n_A,N-n_A\rangle$")
    row = table[(table["V0"] == V) & (table["N"] == 40)][0]
    ax.axhline(row["E_GP_per_N"], color=col, ls="--", lw=2)
    if np.isfinite(row["E_DMC_per_N"]):
        ax.axhline(row["E_DMC_per_N"], color=col, ls=":", lw=3.5)
ax.plot([], [], "k--", lw=2, label=r"$E_{\rm GP}(N)/N$ (same colour)")
ax.plot([], [], "k:", lw=3.5, label=r"$E_{\rm DMC}/N$ (same colour)")
ax.set_xlabel(r"$n_A/N$")
ax.set_ylabel(r"$E/N\,[\varepsilon]$")
ax.set_title(r"$N=40$")
ax.grid(True, ls="--", alpha=0.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), fontsize=20)
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "split_energy_N40.pdf"), bbox_inches="tight")
print("Saved final_imag/split_energy_N40.pdf")
