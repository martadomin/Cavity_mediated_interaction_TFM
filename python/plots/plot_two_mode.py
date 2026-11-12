"""
Figures of the two-mode ground state (data from julia/gp/two_mode.jl), N = 40.

  two_mode_energy_N40.pdf    E/N vs V0 in the crossover (-20 to -60): GP, two-mode, DMC;
                             lower panel (E - E_GP)/N
  two_mode_delta_N40.pdf     the lower panel alone: (E - E_GP)/N vs V0, two-mode and DMC
  two_mode_weights_N40.pdf   |psi(n_A)|^2 vs n_A/N for V0 = -25, -40, -60, -150

Usage (from the repository root, after julia julia/gp/two_mode.jl):
    python python/plots/plot_two_mode.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pump_axis import add_pump_axis  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 26,
    "text.usetex": True,
})


def read_csv(path):
    with open(path) as f:
        cols = f.readline().strip().split(",")
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return {c: data[:, i] for i, c in enumerate(cols)}


dense = read_csv("numpy_arrays_GP/two_mode_dense_N40.csv")
tab = read_csv("numpy_arrays_GP/two_mode.csv")
wts = read_csv("numpy_arrays_GP/two_mode_weights_N40.csv")
os.makedirs("final_imag", exist_ok=True)
VC = 2 * np.pi ** 2 * 40 / 39                          # GP threshold |V0|_c at N = 40
THR = dict(color="0.4", ls="--", lw=2)

# ---- figure 1: energies in the crossover
sel = (dense["V0"] <= -20) & (dense["V0"] >= -60)
m = (tab["N"] == 40) & (tab["V0"] >= -60) & np.isfinite(tab["E_DMC_per_N"])
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 15), height_ratios=[2, 1.3], sharex=True)
ax.plot(dense["V0"][sel], dense["E_GP_per_N"][sel], "k-", lw=2.5, label="GP")
ax.plot(dense["V0"][sel], dense["E2_per_N"][sel], "C3--", lw=2.5, label="two-mode")
conv = m & (tab["DMC_tau"] >= 32 - 1e-9)                 # converged runs (tau = 32) vs older tau = 8 runs
old = m & ~conv
ax.errorbar(tab["V0"][conv], tab["E_DMC_per_N"][conv], yerr=tab["E_DMC_err"][conv], fmt="o", color="C0", ms=11,
            capsize=5, label=r"DMC ($\tau=32$)")
ax.errorbar(tab["V0"][old], tab["E_DMC_per_N"][old], yerr=tab["E_DMC_err"][old], fmt="o", color="C0", ms=11,
            mfc="none", capsize=5, label=r"DMC ($\tau=8$, older runs)")
ax.set_ylabel(r"$E/N\,[\varepsilon]$")
ax.set_title(r"$N=40$")
ax.grid(True, ls="--", alpha=0.6)
ax.legend()

ax2.axhline(0, color="k", lw=2.5)
ax2.axvline(-VC, **THR)
ax2.text(-VC - 0.6, 0.88, r"$|V_0|_c$", transform=ax2.get_xaxis_transform(), ha="right", va="top", fontsize=30, color="0.3")
ax2.plot(dense["V0"][sel], dense["E2_per_N"][sel] - dense["E_GP_per_N"][sel], "C3--", lw=2.5)
for sel_, mfc in ((conv, "C0"), (old, "none")):
    ax2.errorbar(tab["V0"][sel_], tab["E_DMC_per_N"][sel_] - tab["E_GP_per_N"][sel_], yerr=tab["E_DMC_err"][sel_],
                 fmt="o", color="C0", mfc=mfc, ms=11, capsize=5)
ax2.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax2.set_ylabel(r"$(E-E_{\rm GP})/N\,[\varepsilon]$")
ax2.grid(True, ls="--", alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join("final_imag", "two_mode_energy_N40.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- figure 1b: the lower panel alone
fig, ax2 = plt.subplots(figsize=(13, 10))
ax2.axhline(0, color="k", lw=2.5, label="GP")
ax2.axvline(-VC, **THR)
ax2.text(-VC - 0.6, 0.88, r"$|V_0|_c$", transform=ax2.get_xaxis_transform(), ha="right", va="top", fontsize=30, color="0.3")
ax2.plot(dense["V0"][sel], dense["E2_per_N"][sel] - dense["E_GP_per_N"][sel], "C3--", lw=2.5, label="two-mode")
for sel_, mfc, lab in ((conv, "C0", r"DMC ($\tau=32$)"), (old, "none", r"DMC ($\tau=8$, older runs)")):
    if sel_.any():
        ax2.errorbar(tab["V0"][sel_], tab["E_DMC_per_N"][sel_] - tab["E_GP_per_N"][sel_], yerr=tab["E_DMC_err"][sel_],
                     fmt="o", color="C0", mfc=mfc, ms=11, capsize=5, label=lab)
ax2.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax2)
ax2.set_ylabel(r"$(E-E_{\rm GP})/N\,[\varepsilon]$")
ax2.set_title(r"$N=40$")
ax2.grid(True, ls="--", alpha=0.6)
ax2.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join("final_imag", "two_mode_delta_N40.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- figure 2: weights
fig, ax = plt.subplots(figsize=(13, 9))
for i, V in enumerate([-25, -40, -60, -150]):
    ax.plot(wts["nA_over_N"], wts[f"w_V0{V}"], "o-", color=f"C{i}", ms=7, lw=2, label=fr"$V_0={V}\,\varepsilon$")
ax.set_xlabel(r"$n_A/N$")
ax.set_ylabel(r"$|\psi(n_A)|^2$")
ax.set_title(r"$N=40$")
ax.set_yscale("log")
ax.set_ylim(1e-6, 1)
ax.grid(True, ls="--", alpha=0.6)
ax.legend(loc="lower center", fontsize=24)
fig.tight_layout()
fig.savefig(os.path.join("final_imag", "two_mode_weights_N40.pdf"), bbox_inches="tight")
print("Saved final_imag/two_mode_energy_N40.pdf and final_imag/two_mode_weights_N40.pdf")
