"""
Superfluid fraction of the GP state from the Leggett formula, compared with VMC / DMC.

    rho_s/rho = 1 / (<n> <1/n>),   averages over one unit cell (lambda = 1)

Two GP densities:
  * single branch   n(x) = |phi(x)|^2,  phi(x) = sum_n c_n exp(i 2 pi n x)   (period lambda)
  * Z2-symmetric    n(x) = [|phi(x)|^2 + |phi(x + 1/2)|^2] / 2                 (period lambda/2)
The QMC values (python/leggett_analysis.py) use the density symmetrized under x -> x + lambda/2,
i.e. the two-branch (cat) density, so the Z2-symmetric GP curve is the like-for-like comparison.
For a single-orbital state in 1D the Leggett expression is the exact phase-twist superfluid
fraction of that state (not only a bound).

GP orbital: same equations as julia/gp/gp_scf.jl (solver from python/dmc_gp_delta.py),
checked against numpy_arrays_GP/gp_cn_N40.csv.

Usage (from the repository root):
    python python/gp_superfluid.py
Output: numpy_arrays_GP/gp_superfluid.csv, final_imag/gp_superfluid.pdf
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc_gp_delta import _ground, NMAX  # noqa: E402

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

X = (np.arange(8000) + 0.5) / 8000                  # one unit cell
NS = np.arange(-NMAX, NMAX + 1)
LEGGETT = os.path.join("numpy_arrays_DMC", "leggett_analysis", "constant_density_1.0")


def gp_orbital(V0, N):
    r = 1.0 if np.isinf(N) else (N - 1) / N
    f = lambda A: A - V0 * r * _ground(A)[1]
    lo, hi = V0 * r, -1e-8
    if V0 >= 0 or f(lo) * f(hi) > 0:
        return (NS == 0).astype(float)
    return _ground(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))[0]


def leggett(n):
    n = n / n.mean()
    return 1.0 / np.mean(1.0 / n)


def rho_s(V0, N):
    c = gp_orbital(V0, N)
    phi = np.exp(2j * np.pi * np.outer(X, NS)) @ c
    n1 = np.abs(phi) ** 2
    n2 = 0.5 * (n1 + np.abs(np.exp(2j * np.pi * np.outer((X + 0.5) % 1, NS)) @ c) ** 2)
    return leggett(n1), leggett(n2)


def load_qmc(fname):
    f = os.path.join(LEGGETT, fname)
    if not os.path.isfile(f):
        return {}
    d = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
    return d


def main():
    # check the orbitals against gp_scf.jl output (n = -30..30)
    cn = np.loadtxt("numpy_arrays_GP/gp_cn_N40.csv", delimiter=",", skiprows=1)
    dmax = max(np.max(np.abs(gp_orbital(r[0], 40)[NMAX - 30:NMAX + 31] - r[1:])) for r in cn)
    print(f"check: GP orbitals vs gp_cn_N40.csv, max |dc| = {dmax:.1e}")
    print(f"check: uniform state rho_s/rho = {rho_s(0.0, 40)[0]:.12f}")

    V = np.arange(0.0, -200.5, -1.0)
    res = {N: np.array([rho_s(v, N) for v in V]) for N in (40, np.inf)}
    os.makedirs("numpy_arrays_GP", exist_ok=True)
    np.savetxt("numpy_arrays_GP/gp_superfluid.csv",
               np.column_stack([V, res[40][:, 0], res[40][:, 1], res[np.inf][:, 0], res[np.inf][:, 1]]),
               delimiter=",", comments="", fmt="%.10g",
               header="V0,rho_s_single_N40,rho_s_Z2sym_N40,rho_s_single_Ninf,rho_s_Z2sym_Ninf")

    print("\n    V0   GP single (N=40)  GP Z2-sym (N=40)   GP single (inf)  GP Z2-sym (inf)")
    for v in (-20, -25, -30, -40, -50, -60, -80, -100, -150, -200):
        i = np.where(V == v)[0][0]
        print(f"  {v:5d}   {res[40][i, 0]:14.4e}   {res[40][i, 1]:14.4f}   {res[np.inf][i, 0]:14.4e}   {res[np.inf][i, 1]:14.4f}")

    qN = load_qmc("leggett.csv")
    qinf = load_qmc("leggett_Ninf.csv")
    est = [("VMC", "o", "C1", "VMC"), ("DMC", "s", "C2", "DMC mixed"), ("lin", "^", "C3", r"$2\,{\rm DMC}-{\rm VMC}$")]

    fig, axes = plt.subplots(1, 2, figsize=(24, 9.5), sharey=True)
    for ax, N, q, title in ((axes[0], 40, qN, r"$N=40$"), (axes[1], np.inf, qinf, r"$N\to\infty$")):
        ax.plot(V, res[N][:, 1], "k-", lw=2.5, label=r"GP, $Z_2$-symmetric density")
        ax.plot(V, res[N][:, 0], "k--", lw=2.5, label=r"GP, single branch $|\varphi|^2$")
        if len(q):
            for e, mk, col, lab in est:
                if np.isinf(N):
                    m = q["estimator"] == e
                    ax.errorbar(q["V0_kac"][m], q["rho_s_inf"][m], yerr=q["rho_s_inf_err"][m], fmt=mk, color=col,
                                ms=11, mfc="none", capsize=4, label=lab)
                else:
                    m = (q["estimator"] == e) & (q["N"] == 40)
                    ax.plot(q["V0_kac"][m], q["rho_s"][m], mk, color=col, ms=11, mfc="none", label=lab)
        ax.set_title(title)
        ax.set_xlabel(r"$V_0/\varepsilon$")
        ax.set_xlim(-205, 0)
        ax.grid(True, ls="--", alpha=0.6)
    axes[0].set_ylabel(r"$\rho_s/\rho$")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    os.makedirs("final_imag", exist_ok=True)
    fig.savefig(os.path.join("final_imag", "gp_superfluid.pdf"), bbox_inches="tight")
    print("\nSaved numpy_arrays_GP/gp_superfluid.csv and final_imag/gp_superfluid.pdf")


if __name__ == "__main__":
    main()
