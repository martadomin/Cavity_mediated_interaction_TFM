"""
Energy per particle against V0 at several N: DMC (dots joined by lines), the analytic N = 2 energy and the thermodynamic-limit GP curve only (r = 1, |V0|_c = 2 pi^2). Left: full range.
Right: zoom on the threshold region. For N = 2 the Hamiltonian separates in x1 + x2 and x1 - x2 into two Mathieu problems, so E/N = pi^2 a_0(V0 / (8 pi^2)).

DMC: filled dots are the final trials (cat or single branch, 1200 walkers, tau = 8; N = 40 from numpy_arrays_GP/energy_final_N40.csv, written by python/plots/plot_energy_final.py). Open dots are the
original trial (numpy_arrays_DMC; longest, then largest, run), used only where no final-trial run exists yet. Error bars are statistical only.

Usage (from the repository root):  python python/plots/plot_energy_several_N.py [preview.png]
Output: final_imag/energy_vs_V0_several_N.pdf, numpy_arrays_GP/energy_vs_V0_several_N.csv
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import mathieu_a

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import gp_energy  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 25,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amssymb}",
})

VC_INF = 2 * np.pi ** 2
ORIGINAL_N = (15, 20, 25, 30, 50)
FINAL_N = (2, 3, 5, 7, 10, 15, 20, 30, 50)
COUPLINGS = (-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0)
ALL_SIZES = (2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50)     # fixed, so that a size keeps its colour when others are added
COLORS = {Np: plt.cm.jet(i / (len(ALL_SIZES) - 1)) for i, Np in enumerate(ALL_SIZES)}


def final_runs(Np):
    """{V0: (E/N, err/N)} of the finished cat and single-branch runs with the final settings."""
    out = {}
    for base in ("numpy_arrays_DMC_cat", "numpy_arrays_DMC_branch"):
        for f in glob.glob(os.path.join(base, "constant_density_1.0", "energy", "V0=*", f"energy_1200_0.0010_{Np}_8000_*_IS.npy")):
            m = re.match(rf"energy_1200_0\.0010_{Np}_8000_(-[\d.]+)_", os.path.basename(f))
            e = np.load(f)
            out[round(float(m[1]) * Np * 2) / 2] = (e[0] / Np, e[1] / Np)
    return out


def original_run(V, Np):
    """(E/N, err/N) of the longest, then largest, original-trial run at (V, Np), or None."""
    d = os.path.join("numpy_arrays_DMC", "constant_density_1.0", "energy", f"V0={V / Np:.3f}")
    best = None
    for f in glob.glob(os.path.join(d, "energy_*_IS.npy")):
        m = re.match(rf"energy_(\d+)_([\d.]+)_{Np}_(\d+)_", os.path.basename(f))
        if m and (best is None or (float(m[2]) * int(m[3]), int(m[1])) > best[0]):
            best = ((float(m[2]) * int(m[3]), int(m[1])), f)
    if best is None:
        return None
    e = np.load(best[1])
    return e[0] / Np, e[1] / Np


rows = []
for Np in sorted(set(FINAL_N) | set(ORIGINAL_N)):
    fin = final_runs(Np) if Np in FINAL_N else {}
    rows += [(Np, V, e, err, 1) for V, (e, err) in fin.items()]
    if Np in ORIGINAL_N:
        rows += [(Np, V, *r, 0) for V in COUPLINGS if V not in fin and (r := original_run(V, Np)) is not None]
F = np.loadtxt(os.path.join("numpy_arrays_GP", "energy_final_N40.csv"), delimiter=",", skiprows=1)
rows += [(40, V, e / 40, err / 40, 1) for V, _, _, e, err, _, _, _ in F]
D = np.array(sorted(rows, key=lambda r: (r[0], -r[1])))
np.savetxt(os.path.join("numpy_arrays_GP", "energy_vs_V0_several_N.csv"), D, delimiter=",", comments="", fmt="%.10g", header="N,V0,E_over_N,err,trial_0original_1final")

Vd = np.unique(np.concatenate([np.linspace(-85, -VC_INF * 1.05, 120), np.linspace(-VC_INF * 1.05, -VC_INF, 120), np.linspace(-VC_INF, 0, 40)]))
Eg = np.array([gp_energy(v, np.inf) for v in Vd])
E2 = np.pi ** 2 * mathieu_a(0, np.abs(Vd) / (8 * np.pi ** 2))     # a_0 is even in q

fig, (ax, az) = plt.subplots(1, 2, figsize=(28, 11.5))
for a in (ax, az):
    a.axvline(-VC_INF, color="0.4", ls="--", lw=2)
    a.plot(Vd, E2, "-", color=COLORS[2], lw=2.5, zorder=2)
    a.plot(Vd, Eg, "k-", lw=3.5, zorder=3)
    for Np in np.unique(D[:, 0]):
        if Np != 2:                                                      # N = 2 already has its analytic line
            S = D[D[:, 0] == Np]
            a.plot(S[:, 1], S[:, 2], "-", color=COLORS[int(Np)], lw=2.2, zorder=2)
        for final, face in ((1, COLORS[int(Np)]), (0, "white")):
            S = D[(D[:, 0] == Np) & (D[:, 4] == final)]
            if len(S):
                a.errorbar(S[:, 1], S[:, 2], yerr=S[:, 3], fmt="o", color=COLORS[int(Np)], mfc=face, mec="k", mew=1.2, ms=10, capsize=3, zorder=4)
    a.set_xlabel(r"$V_0/\varepsilon$")
    a.set_ylabel(r"$E/N\,[\varepsilon]$")
    a.grid(True, ls="--", alpha=0.6)
handles = [Line2D([], [], color="k", lw=3.5), Line2D([], [], color="0.4", ls="--", lw=2), Line2D([], [], marker="o", color="0.5", mec="k", ls="", ms=10),
           Line2D([], [], marker="o", color="0.5", mfc="white", mec="k", ls="", ms=10)]
labels = [r"GP, $N\to\infty$", r"$|V_0|_c=2\pi^2$", "DMC, final trials", "DMC, original trial"]
shown = [Np for Np in ALL_SIZES if Np in D[:, 0]]
handles += [Line2D([], [], color=COLORS[Np], lw=3.5) for Np in shown]
labels += [r"$N=2$ (analytic)" if Np == 2 else fr"$N={Np}$" for Np in shown]
ax.legend(handles, labels, loc="lower right", ncol=3, columnspacing=0.9, handletextpad=0.4, handlelength=1.4)
ax.set_xlim(-85, 0)
az.set_xlim(-31, -8)
az.set_ylim(-1.0, 0.05)
fig.tight_layout()

os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "energy_vs_V0_several_N.pdf"), bbox_inches="tight")
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], bbox_inches="tight", dpi=60)      # optional PNG preview
print("Saved final_imag/energy_vs_V0_several_N.pdf, numpy_arrays_GP/energy_vs_V0_several_N.csv")
