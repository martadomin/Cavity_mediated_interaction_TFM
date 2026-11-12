"""
Two-mode energy figure with the orbital-optimised two-mode curve added (data from julia/gp/two_mode.jl and
julia/gp/two_mode_opt.jl), N = 40. Same layout as final_imag/two_mode_energy_N40.pdf (python/plots/plot_two_mode.py),
which is left untouched.

  two_mode_opt_energy_N40.pdf   E/N vs V0 (-20 to -60): GP, two-mode (GP orbitals), DMC;
                                lower panel (E - E_GP)/N with the optimised two-mode curve added

Usage (from the repository root, after julia julia/gp/two_mode.jl and julia julia/gp/two_mode_opt.jl):
    python python/plots/plot_two_mode_opt.py
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
opt = read_csv("numpy_arrays_GP/two_mode_opt_dense_N40.csv")
tab = read_csv("numpy_arrays_GP/two_mode.csv")
os.makedirs("final_imag", exist_ok=True)
VC = 2 * np.pi ** 2 * 40 / 39                          # GP threshold |V0|_c at N = 40
THR = dict(color="0.4", ls="--", lw=2)

sel = (dense["V0"] <= -20) & (dense["V0"] >= -60)
so = (opt["V0"] <= -20) & (opt["V0"] >= -60)
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
ax2.plot(dense["V0"][sel], dense["E2_per_N"][sel] - dense["E_GP_per_N"][sel], "C3--", lw=2.5,
         label="two-mode (GP orbitals)")
ax2.plot(opt["V0"][so], opt["E2_opt_per_N"][so] - opt["E_GP_per_N"][so], "-", color="C2", lw=2,
         label="optimised two-mode")
for sel_, mfc in ((conv, "C0"), (old, "none")):
    ax2.errorbar(tab["V0"][sel_], tab["E_DMC_per_N"][sel_] - tab["E_GP_per_N"][sel_], yerr=tab["E_DMC_err"][sel_],
                 fmt="o", color="C0", mfc=mfc, ms=11, capsize=5)
ax2.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax2.set_ylabel(r"$(E-E_{\rm GP})/N\,[\varepsilon]$")
ax2.grid(True, ls="--", alpha=0.6)
ax2.legend(loc="lower left", fontsize=24)
fig.tight_layout()
fig.savefig(os.path.join("final_imag", "two_mode_opt_energy_N40.pdf"), bbox_inches="tight")
plt.close(fig)

d = opt["E2_opt_per_N"] - opt["E2_GPorb_per_N"]
i = np.argmin(d)
print(f"max (E2_opt - E2_GPorb)/N on the dense grid: {d[i]:.2e} at V0 = {opt['V0'][i]:.1f}")
print("Saved final_imag/two_mode_opt_energy_N40.pdf")
