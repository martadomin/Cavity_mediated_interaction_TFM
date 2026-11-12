"""
Quantum depletion from the full one-body density matrix, N = 40: are lambda_2, lambda_3, ... zero?

For every V0 the QMC rho^(1)(x, x') (python/obdm_analysis.load_raw: most recent run, i.e. the converged
tau = 32 runs where they exist) is symmetrised, averaged over translations by lambda, rescaled to trace N
(as in python/obdm_momentum.py) and diagonalised on the full real-space grid (1000 x 1000, all momenta, not
only the condensate block). Reported: the leading eigenvalues lambda_0 >= lambda_1 >= lambda_2 >= ... / N,
the depletion D = 1 - (lambda_0 + lambda_1)/N (sum rule: everything outside the two density-wave modes),
and the noise level |lambda_min|/N (a true rho^(1) is positive semi-definite, so the most negative
eigenvalue measures the statistical noise of the matrix). Estimators: VMC (reference of the DMC run),
DMC mixed, 2 DMC - VMC.

GP for comparison (no fluctuations at all, so no depletion in either form):
  * single branch phi_A^N (broken symmetry, no cat):  rho = N |phi_A><phi_A|  ->  lambda_0 = N, all others 0;
  * symmetric cat phi_A^N + phi_B^N:  lambda_pm = N (1 +- s)(1 +- s^(N-1)) / (2 (1 + s^N)), all others 0.

Usage (from the repository root):  python python/obdm_depletion.py
Output: final_imag/obdm_depletion_N40.pdf, numpy_arrays_OBDM/depletion_N40.csv
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obdm_analysis import load_raw, symmetrize  # noqa: E402
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

N = 40
BASE = os.path.join("numpy_arrays_DMC", "constant_density_1.0")
V0S = [-15.0, -20.0, -21.0, -22.0, -23.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0]
EST = {"VMC": ("o", "0.6", "VMC"), "DMC": ("s", "C2", "DMC mixed"), "lin": ("^", "C3", r"$2\,$DMC$-$VMC")}
K = 6                                         # eigenvalues kept


def spectrum(rho, L):
    rho = symmetrize(rho, L)
    dx = L / rho.shape[0]
    rho = rho * (N / (np.trace(rho) * dx))
    w = np.linalg.eigvalsh(rho * dx)[::-1] / N
    return w[:K], -w[-1]


def gp_lambdas():
    d = np.loadtxt(os.path.join("numpy_arrays_GP", "gp_dense.csv"), delimiter=",", skiprows=1)
    with open(os.path.join("numpy_arrays_GP", "gp_dense.csv")) as f:
        cols = f.readline().strip().split(",")
    return d[:, 0], d[:, cols.index("lp_N40")], d[:, cols.index("lm_N40")]


rows = []
print("   V0   est   lambda_0..5 / N                                   D = 1-(l0+l1)/N   noise |l_min|/N")
for V in V0S:
    v, L = load_raw(BASE, N, V, "VMC")
    d, _ = load_raw(BASE, N, V, "DMC")
    if v is None or d is None:
        continue
    for est, M in (("VMC", v), ("DMC", d), ("lin", 2 * d - v)):
        w, noise = spectrum(M, L)
        D = 1 - w[0] - w[1]
        rows.append((V, list(EST).index(est), *w, D, noise))
        print(f"  {V:5.0f}  {est:4s} " + " ".join(f"{x:7.4f}" for x in w) + f"    {D:+.4f}          {noise:.4f}")
R = np.array(rows)
np.savetxt(os.path.join("numpy_arrays_OBDM", f"depletion_N{N}.csv"), R, delimiter=",", comments="", fmt="%.8g",
           header="V0,estimator(0=VMC,1=DMC,2=lin)," + ",".join(f"lambda{k}_over_N" for k in range(K)) +
                  ",depletion_1-l0-l1,noise_absmin_over_N")

Vg, lp, lm = gp_lambdas()
sel = (Vg >= -155)
fig, (ax, axs, axl) = plt.subplots(1, 3, figsize=(42, 11))
# left: lambda_0, lambda_1, lambda_2, lambda_3 on a linear scale
ax.plot(Vg[sel], np.ones(sel.sum()), "-", color="k", lw=2, alpha=0.5, label=r"GP single branch: $\lambda_0=N$, rest $0$")
ax.plot(Vg[sel], lp[sel], "k-", lw=2.5, label=r"GP cat: $\lambda_\pm$, rest $0$")
ax.plot(Vg[sel], lm[sel], "k--", lw=2.5)
for k, est in enumerate(EST):
    mk, col, lab = EST[est]
    r = R[R[:, 1] == k]
    for j, mfc in ((2, col), (3, "none"), (4, col), (5, "none")):
        ax.plot(r[:, 0], r[:, j], mk, color=col, ms=11 if j < 4 else 8, mfc=mfc, mew=2,
                label=lab if j == 2 else None, alpha=1 if j < 4 else 0.8)
ax.axhline(0, color="k", lw=1)
ax.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax.set_ylabel(r"$\lambda_k/N$")
ax.set_title(r"$\lambda_0,\lambda_1$ (large) and $\lambda_2,\lambda_3$ (small markers, at $\approx 0$)", fontsize=28)
ax.grid(True, ls="--", alpha=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.93), fontsize=22)

# middle: lambda_0 + lambda_1 (= 1 - depletion), error bar = noise level |lambda_min|/N
axs.axhline(1, color="k", lw=2.5, label=r"GP (single branch and cat): $1$")
for k, est in enumerate(EST):
    mk, col, lab = EST[est]
    r = R[R[:, 1] == k]
    axs.errorbar(r[:, 0] + (k - 1) * 1.2, r[:, 2] + r[:, 3], yerr=r[:, 2 + K + 1], fmt=mk, color=col, ms=11,
                 capsize=4, mew=2, label=lab)
axs.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(axs)
axs.set_ylabel(r"$(\lambda_0+\lambda_1)/N$")
axs.set_title(r"two leading modes: $(\lambda_0+\lambda_1)/N$ (bars: noise $|\lambda_{\min}|/N$)", fontsize=28)
axs.set_ylim(0.95, 1.07)
axs.grid(True, ls="--", alpha=0.6)
axs.legend(loc="upper left", fontsize=22)

# right: lambda_2 / N, depletion and noise floor on a log scale
for k, est in enumerate(EST):
    if est == "VMC":
        continue
    mk, col, lab = EST[est]
    r = R[R[:, 1] == k]
    axl.semilogy(r[:, 0], np.abs(r[:, 4]), mk + "-", color=col, ms=11, lw=1.5, label=lab + r": $\lambda_2/N$")
    axl.semilogy(r[:, 0], r[:, 2 + K + 1], mk + ":", color=col, ms=9, mfc="none", mew=2, lw=1.5,
                 label=lab + r": noise $|\lambda_{\min}|/N$")
axl.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(axl)
axl.set_ylabel(r"$\lambda_2/N$, noise")
axl.set_title(r"$\lambda_2$ vs the noise of $\rho^{(1)}$ (GP: $\lambda_2=0$)", fontsize=28)
axl.grid(True, which="both", ls="--", alpha=0.5)
axl.legend(loc="upper left", fontsize=22)
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", f"obdm_depletion_N{N}.pdf"), bbox_inches="tight")
print(f"Saved final_imag/obdm_depletion_N{N}.pdf and numpy_arrays_OBDM/depletion_N{N}.csv")
