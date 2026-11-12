"""
Quantum depletion vs N: full-spectrum analysis of rho^(1) (as python/obdm_depletion.py) for N = 15 ... 50.

For each (N, V0) the QMC rho^(1) (most recent run = converged tau = 32, 1200 walkers; at -150 the N = 40 run
has 2400 walkers and N = 35 is skipped, population-control bias) is symmetrised, translation-averaged, rescaled
to trace N and diagonalised on the full real-space grid. Reported: (lambda_0 + lambda_1)/N, lambda_2/N and the
noise level |lambda_min|/N, for DMC mixed and 2 DMC - VMC. GP (single branch or cat): lambda_0 + lambda_1 = N,
lambda_2 = 0 at every N. A real depletion (fluctuations out of the two density-wave modes) would grow at small N,
roughly as 1/N; noise does not have that trend.

Usage (from the repository root):  python python/obdm_depletion_vs_N.py
Output: final_imag/obdm_depletion_vs_N.pdf, numpy_arrays_OBDM/depletion_vs_N.csv
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obdm_analysis import load_raw, symmetrize  # noqa: E402

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

BASE = os.path.join("numpy_arrays_DMC", "constant_density_1.0")
V0S = [-25.0, -30.0, -40.0, -60.0, -150.0]
NS = [15, 20, 25, 30, 35, 40, 50]
EXCLUDE = {(-150.0, 35)}


def spectrum(rho, N, L):
    rho = symmetrize(rho, L)
    dx = L / rho.shape[0]
    rho = rho * (N / (np.trace(rho) * dx))
    w = np.linalg.eigvalsh(rho * dx)[::-1] / N
    return w[0], w[1], w[2], -w[-1]


rows = []
print("   V0    N   est  (l0+l1)/N   l2/N     noise")
for V in V0S:
    for N in NS:
        if (V, N) in EXCLUDE:
            continue
        v, L = load_raw(BASE, N, V, "VMC")
        d, _ = load_raw(BASE, N, V, "DMC")
        if v is None or d is None:
            continue
        for k, (est, M) in enumerate((("DMC", d), ("lin", 2 * d - v))):
            l0, l1, l2, nz = spectrum(M, N, L)
            rows.append((V, N, k, l0, l1, l2, nz))
            print(f"  {V:5.0f}  {N:3d}  {est:4s}  {l0 + l1:.4f}    {l2:.4f}   {nz:.4f}")
R = np.array(rows)
np.savetxt(os.path.join("numpy_arrays_OBDM", "depletion_vs_N.csv"), R, delimiter=",", comments="", fmt="%.8g",
           header="V0,N,estimator(0=DMC,1=lin),lambda0_over_N,lambda1_over_N,lambda2_over_N,noise_absmin_over_N")

cmap = plt.get_cmap("plasma")
cols = {V: cmap(0.85 * i / (len(V0S) - 1)) for i, V in enumerate(V0S)}
fig, axes = plt.subplots(2, 2, figsize=(28, 20), sharex=True)
for k, est in enumerate(("DMC mixed", r"$2\,$DMC$-$VMC")):
    ax, axl = axes[k]
    ax.axhline(1, color="k", lw=2.5, label="GP: 1" if k == 0 else None)
    for i, V in enumerate(V0S):
        r = R[(R[:, 0] == V) & (R[:, 2] == k)]
        if not len(r):
            continue
        ax.errorbar(r[:, 1] + (i - 2) * 0.35, r[:, 3] + r[:, 4], yerr=r[:, 6], fmt="o-", color=cols[V], ms=11,
                    capsize=4, lw=1.5, label=fr"$V_0={V:.0f}\,\varepsilon$")
        axl.semilogy(r[:, 1], r[:, 5], "o-", color=cols[V], ms=11, lw=1.5)
        axl.semilogy(r[:, 1], r[:, 6], "s:", color=cols[V], ms=8, mfc="none", mew=2, lw=1.2)
    ax.set_ylabel(r"$(\lambda_0+\lambda_1)/N$")
    ax.set_title(est + r": $(\lambda_0+\lambda_1)/N$ (bars: noise)", fontsize=30)
    ax.grid(True, ls="--", alpha=0.6)
    axl.set_ylabel(r"$\lambda_2/N$ (filled), noise (open)")
    axl.set_title(est + r": $\lambda_2/N$ vs noise $|\lambda_{\min}|/N$", fontsize=30)
    axl.grid(True, which="both", ls="--", alpha=0.5)
axes[0, 0].legend(loc="lower right", fontsize=22)
NN = np.linspace(14, 52, 50)
for axl in axes[:, 1]:
    axl.plot(NN, 0.02 / NN, "k--", lw=1.5, label=r"$\propto 1/N$ (guide)")
    axl.legend(loc="upper right", fontsize=22)
for ax in axes[1]:
    ax.set_xlabel(r"$N$")
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "obdm_depletion_vs_N.pdf"), bbox_inches="tight")
print("Saved final_imag/obdm_depletion_vs_N.pdf and numpy_arrays_OBDM/depletion_vs_N.csv")
