"""
QMC natural orbitals vs GP orbitals as a function of N (style of final_imag/gp_delta_vs_N_all.pdf).

For each (N, V0) the condensate block of rho^(1) is analysed exactly as in python/obdm_momentum.py
(symmetrised, translation-averaged, |n| <= 8, even / odd natural orbitals phi_+, phi_-), and compared with
the GP orbital at the same N (same equations as julia/gp/gp_scf.jl, with the (N-1)/N factor; solved here
because gp_cn_*.csv exists only for N = 40): its normalised even / odd parts c^+, c^-.
Plotted: the infidelity 1 - |<phi_pm^QMC | c^pm>|^2 vs N (log scale), one curve per V0, for the DMC mixed
and the extrapolated (2 DMC - VMC) estimators. Filled: tau = 32 runs, open: older tau = 8 runs (the run is
the most recent OBDM file at each point, as in obdm_analysis.load_raw).
Only couplings above the GP threshold at all N shown (the GP odd part phi_- does not exist below it).

Usage (from the repository root):  python python/plots/plot_orbital_overlap_vs_N.py
Output: final_imag/orbital_overlap_vs_N.pdf, numpy_arrays_OBDM/orbital_overlap_vs_N.csv
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import _ground, NMAX  # noqa: E402
from obdm_momentum import analyse, estimator_matrix  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 22,
    "text.usetex": True,
})

BASE = os.path.join("numpy_arrays_DMC", "constant_density_1.0")
V0S = [-25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0]
NS = [15, 20, 25, 30, 35, 40, 50]
EXCLUDE = {(-150.0, 35)}            # 1200-walker run with population-control bias


def gp_orbital(V0, N):
    """GP c_n (n = -NMAX..NMAX) at finite N, or None below threshold."""
    r = (N - 1) / N
    f = lambda A: A - V0 * r * _ground(A)[1]
    lo, hi = V0 * r, -1e-8
    if f(lo) * f(hi) > 0:
        return None
    return _ground(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))[0]


def dmc_tau(V0, N):
    """imaginary time of the most recent DMC OBDM file (the one load_raw uses)."""
    v = V0 / N
    files = [f for f in glob.glob(os.path.join(BASE, "obdm", f"V0={v:.3f}", f"obdm_*_{N}_*_{v:.3f}_{float(N):.1f}_IS.npy"))
             if "extrap" not in os.path.basename(f)]
    if not files:
        return np.nan
    m = re.match(r"obdm_(\d+)_([\d.]+)_\d+_(\d+)_", os.path.basename(max(files, key=os.path.getmtime)))
    return float(m[2]) * int(m[3])


rows = []
print("   V0    N   tau  | 1-ov+ DMC   1-ov- DMC  | 1-ov+ lin   1-ov- lin")
for V in V0S:
    for N in NS:
        if (V, N) in EXCLUDE:
            continue
        c = gp_orbital(V, N)
        if c is None:
            continue
        gp = (c, np.arange(-NMAX, NMAX + 1))
        out = {}
        for est in ("DMC", "lin"):
            rho, L = estimator_matrix(BASE, N, V, est)
            if rho is None:
                break
            r = analyse(rho, N, L, 8, gp)
            out[est] = (r["overlap_plus"], r["overlap_minus"])
        if len(out) < 2:
            continue
        tau = dmc_tau(V, N)
        rows.append((V, N, tau, *out["DMC"], *out["lin"]))
        print(f"  {V:5.0f}  {N:3d}  {tau:4g}  |  {1 - out['DMC'][0]:.2e}   {1 - out['DMC'][1]:.2e}  |"
              f"  {1 - out['lin'][0]:.2e}   {1 - out['lin'][1]:.2e}")
R = np.array(rows)
np.savetxt(os.path.join("numpy_arrays_OBDM", "orbital_overlap_vs_N.csv"), R, delimiter=",", comments="", fmt="%.10g",
           header="V0,N,DMC_tau,overlap_plus_DMC,overlap_minus_DMC,overlap_plus_lin,overlap_minus_lin")

fig, axes = plt.subplots(2, 2, figsize=(26, 19), sharex=True, sharey=True)
cmap = plt.get_cmap("plasma")
cols = {V: cmap(0.88 * i / (len(V0S) - 1)) for i, V in enumerate(V0S)}
panels = [(0, 0, 3, r"DMC mixed, $\phi_+$"), (0, 1, 4, r"DMC mixed, $\phi_-$"),
          (1, 0, 5, r"$2\,$DMC$-$VMC, $\phi_+$"), (1, 1, 6, r"$2\,$DMC$-$VMC, $\phi_-$")]
for i, j, k, title in panels:
    ax = axes[i, j]
    for V in V0S:
        sel = R[:, 0] == V
        if not sel.any():
            continue
        Nv, y, conv = R[sel, 1], np.maximum(1 - R[sel, k], 1e-6), R[sel, 2] >= 32 - 1e-9
        ax.plot(Nv, y, "-", color=cols[V], lw=1.5, alpha=0.7)
        ax.plot(Nv[conv], y[conv], "o", color=cols[V], ms=11,
                label=fr"$V_0={V:.0f}\,\varepsilon$" if conv.any() else None)
        ax.plot(Nv[~conv], y[~conv], "o", color=cols[V], ms=11, mfc="none", mew=2,
                label=None if conv.any() else fr"$V_0={V:.0f}\,\varepsilon$")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.5)
for ax in axes[1]:
    ax.set_xlabel(r"$N$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$1-|\langle\phi^{\rm QMC}|\phi^{\rm GP}\rangle|^2$")
h, lab = axes[0, 0].get_legend_handles_labels()
h += [plt.Line2D([], [], marker="o", ls="", color="0.4", ms=11),
      plt.Line2D([], [], marker="o", ls="", color="0.4", mfc="none", mew=2, ms=11)]
lab += [r"$\tau=32$", r"$\tau=8$ (older runs)"]
fig.tight_layout(rect=(0, 0, 0.86, 1))
fig.legend(h, lab, loc="center right", fontsize=24)
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "orbital_overlap_vs_N.pdf"), bbox_inches="tight")
print("Saved final_imag/orbital_overlap_vs_N.pdf and numpy_arrays_OBDM/orbital_overlap_vs_N.csv")
