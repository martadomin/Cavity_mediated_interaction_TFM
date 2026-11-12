"""
Quantum depletion in particles, N_dep = N - N_0 - N_1 (N_0,1 = lambda_0,1, the two leading eigenvalues of the
full rho^(1)), from the spectra of python/obdm_depletion.py (N = 40, all V0) and python/obdm_depletion_vs_N.py
(N = 15 ... 50). Error bars: the noise level of rho^(1) in particles, N |lambda_min|/N (the most negative
eigenvalue of a matrix that should be positive semi-definite). The trace of each matrix is rescaled to N
(0.3 - 4 % bin-smearing deficit), so N_dep is also affected by that normalisation. GP (single branch or cat):
N_dep = 0.

Usage (from the repository root, after python python/obdm_depletion.py and python python/obdm_depletion_vs_N.py):
    python python/plots/plot_depletion_particles.py
Output: final_imag/depletion_particles.pdf
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pump_axis import add_pump_axis  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 22,
    "text.usetex": True,
})

A = np.loadtxt(os.path.join("numpy_arrays_OBDM", "depletion_N40.csv"), delimiter=",", skiprows=1)
B = np.loadtxt(os.path.join("numpy_arrays_OBDM", "depletion_vs_N.csv"), delimiter=",", skiprows=1)
N40 = 40

fig, (ax, axn, axv) = plt.subplots(1, 3, figsize=(42, 11))

# left: N = 40 vs V0 (columns: V0, est, lambda0..5, D, noise)
ax.axhline(0, color="k", lw=2.5, label=r"GP: $0$")
for k, (mk, col, lab, dx) in enumerate((("o", "0.6", "VMC", -1.2), ("s", "C2", "DMC mixed", 0.0),
                                        ("^", "C3", r"$2\,$DMC$-$VMC", 1.2))):
    r = A[A[:, 1] == k]
    dep = N40 * (1 - r[:, 2] - r[:, 3])
    ax.errorbar(r[:, 0] + dx, dep, yerr=N40 * r[:, -1], fmt=mk, color=col, ms=11, capsize=4, mew=2, label=lab)
ax.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax.set_ylabel(r"$N-N_0-N_1$")
ax.set_title(r"$N=40$", fontsize=30)
ax.set_ylim(-1.5, 1.5)
ax.grid(True, ls="--", alpha=0.6)
ax.legend(loc="upper left", fontsize=22)

# right: vs N, DMC mixed (columns: V0, N, est, l0, l1, l2, noise)
V0S = sorted(set(B[:, 0]), reverse=True)
cmap = plt.get_cmap("plasma")
axn.axhline(0, color="k", lw=2.5, label=r"GP: $0$")
for i, V in enumerate(V0S):
    r = B[(B[:, 0] == V) & (B[:, 2] == 0)]
    Ns = r[:, 1]
    dep = Ns * (1 - r[:, 3] - r[:, 4])
    axn.errorbar(Ns + (i - 2) * 0.35, dep, yerr=Ns * r[:, 6], fmt="o-", color=cmap(0.85 * i / (len(V0S) - 1)),
                 ms=11, capsize=4, lw=1.5, label=fr"$V_0={V:.0f}\,\varepsilon$")
axn.set_xlabel(r"$N$")
axn.set_ylabel(r"$N-N_0-N_1$")
axn.set_title("DMC mixed", fontsize=30)
axn.set_ylim(-1.0, 1.0)
axn.grid(True, ls="--", alpha=0.6)
axn.legend(loc="lower left", fontsize=22, ncol=2)

# third: vs V0, one curve per N (DMC mixed)
NS = sorted(set(B[:, 1]))
cmapn = plt.get_cmap("viridis")
axv.axhline(0, color="k", lw=2.5, label=r"GP: $0$")
for i, Nn in enumerate(NS):
    r = B[(B[:, 1] == Nn) & (B[:, 2] == 0)]
    r = r[np.argsort(r[:, 0])]
    axv.errorbar(r[:, 0] + (i - 3) * 0.8, Nn * (1 - r[:, 3] - r[:, 4]), yerr=Nn * r[:, 6], fmt="o-",
                 color=cmapn(0.9 * i / (len(NS) - 1)), ms=11, capsize=4, lw=1.5, label=fr"$N={int(Nn)}$")
axv.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(axv)
axv.set_ylabel(r"$N-N_0-N_1$")
axv.set_title("DMC mixed", fontsize=30)
axv.set_ylim(-1.0, 1.0)
axv.grid(True, ls="--", alpha=0.6)
axv.legend(loc="lower right", fontsize=22, ncol=2)
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "depletion_particles.pdf"), bbox_inches="tight")

print("N = 40, DMC mixed: N - N0 - N1 (noise)")
r = A[A[:, 1] == 1]
for V, l0, l1, nz in zip(r[:, 0], r[:, 2], r[:, 3], r[:, -1]):
    print(f"  {V:6.0f}   {N40 * (1 - l0 - l1):+.3f}  ({N40 * nz:.3f})")
print("vs N, DMC mixed: N - N0 - N1")
for V in V0S:
    r = B[(B[:, 0] == V) & (B[:, 2] == 0)]
    print(f"  {V:6.0f}: " + "  ".join(f"N={int(n)}: {n * (1 - a - b):+.3f}" for n, a, b in zip(r[:, 1], r[:, 3], r[:, 4])))
print("Saved final_imag/depletion_particles.pdf")
