"""
DMC vs single-orbital GP energies at finite N:  Delta(N) = E_DMC/N - E_GP(N)/N.

GP at finite N (same equations as julia/gp/gp_scf.jl):
    E/N = epsk sum_n n^2 c_n^2 + (V0/2) (N-1)/N Theta1^2,  Theta1 = sum_n c_n c_{n+1},
    c = lowest eigenvector of h[A] (diag epsk n^2, off-diag A/2), A = V0 (N-1)/N Theta1 (Brent).
E_GP(N) is a rigorous upper bound to the exact ground-state energy, so Delta <= 0 up to
statistical error. Mean field is exact for N -> inf under Kac scaling, Delta -> 0 as O(1/N).

DMC energy error: blocking (Flyvbjerg-Petersen) of the per-step energy series after the 20 %
burn-in used by run_dmc.jl; err = largest blocked error over levels with >= MIN_BLOCKS blocks.

Usage (from the repository root):
    python python/dmc_gp_delta.py --V0kac -150                        # 1200 walkers, 32000 steps
    python python/dmc_gp_delta.py --V0kac -25 -30 -40 -60 --N 15 20 25 30 40
Output: numpy_arrays_DMC/gp_delta/constant_density_1.0/delta_V0kac<V>.csv/.md,
        final_imag/gp_delta_vs_invN_V0<V...>.pdf, final_imag/blocking_V0<V>_N<N>.pdf
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

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

EPSK = 2 * np.pi ** 2
NMAX = 40
BURN_IN = 0.2
MIN_BLOCKS = 25
BASE = os.path.join("numpy_arrays_DMC", "constant_density_1.0")
OUT = os.path.join("numpy_arrays_DMC", "gp_delta", "constant_density_1.0")


# ------------------------------------------------------------------ GP
def _ground(A):
    n = np.arange(-NMAX, NMAX + 1)
    _, v = eigh_tridiagonal(EPSK * n.astype(float) ** 2, np.full(2 * NMAX, A / 2), select="i", select_range=(0, 0))
    c = v[:, 0] * np.sign(v[NMAX, 0])
    return c, np.sum(c[:-1] * c[1:])


def gp_energy(V0, N):
    """E_GP/N at finite N (N = np.inf for the thermodynamic limit)."""
    r = 1.0 if np.isinf(N) else (N - 1) / N
    f = lambda A: A - V0 * r * _ground(A)[1]
    lo, hi = V0 * r, -1e-8
    if V0 >= 0 or f(lo) * f(hi) > 0:
        return 0.0                                     # uniform state
    A = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)
    c, th = _ground(A)
    n = np.arange(-NMAX, NMAX + 1)
    return EPSK * np.sum(n ** 2 * c ** 2) + V0 / 2 * r * th ** 2


# ------------------------------------------------------------------ DMC
def blocking_curve(y):
    """[(n_blocks, block_length, error of the mean)] for successive pairwise blockings."""
    y = np.asarray(y, float)
    out, size = [], 1
    while len(y) >= 4:
        out.append((len(y), size, y.std(ddof=1) / np.sqrt(len(y))))
        n = len(y) // 2
        y = 0.5 * (y[0:2 * n:2] + y[1:2 * n:2])
        size *= 2
    return out


def dmc_series(N, V0kac, walkers, steps, dtau):
    V0 = V0kac / N
    f = os.path.join(BASE, "energies", f"V0={V0:.3f}",
                     f"energies_{walkers}_{dtau:.4f}_{N}_{steps}_{V0:.3f}_{float(N):.1f}_IS.npy")
    if not os.path.isfile(f):
        return None
    e = np.load(f) / N
    return e[int(BURN_IN * len(e)):]


def analyse(V0kac, Ns, walkers, steps, dtau):
    rows, curves = [], {}
    for N in Ns:
        y = dmc_series(N, V0kac, walkers, steps, dtau)
        if y is None:
            print(f"  N = {N}: no run with {walkers} walkers, {steps} steps — skipped")
            continue
        curve = blocking_curve(y)
        err = max(e for nb, _, e in curve if nb >= MIN_BLOCKS)
        E, Egp = y.mean(), gp_energy(V0kac, N)
        rows.append((N, E, err, Egp, E - Egp, err, (E - Egp) / abs(Egp)))
        curves[N] = curve
    return rows, curves


def fit_delta(rows):
    """Weighted fit Delta = a + b/N; returns (a, err_a, b, err_b, chi2/dof)."""
    N = np.array([r[0] for r in rows], float)
    d = np.array([r[4] for r in rows])
    s = np.array([r[5] for r in rows])
    A = np.column_stack([np.ones_like(N), 1 / N]) / s[:, None]
    cov = np.linalg.inv(A.T @ A)
    p = cov @ A.T @ (d / s)
    chi2 = np.sum(((d - p[0] - p[1] / N) / s) ** 2)
    dof = max(len(N) - 2, 1)
    return p[0], np.sqrt(cov[0, 0]), p[1], np.sqrt(cov[1, 1]), chi2 / dof


def write_table(V0kac, rows, fit):
    os.makedirs(OUT, exist_ok=True)
    tag = f"V0kac{V0kac:.0f}"
    np.savetxt(os.path.join(OUT, f"delta_{tag}.csv"), np.array(rows), delimiter=",", comments="",
               header="N,E_DMC_per_N,err,E_GP_per_N,Delta,Delta_err,Delta_over_absEGP")
    with open(os.path.join(OUT, f"delta_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(f"V0 = {V0kac:.0f} eps\n\n")
        f.write("| N | E_DMC/N | E_GP(N)/N | Delta = E_DMC/N - E_GP/N | Delta/|E_GP| |\n|---|---|---|---|---|\n")
        for N, E, e, Eg, d, de, rel in rows:
            f.write(f"| {N} | {E:.4f} ± {e:.4f} | {Eg:.4f} | {d:+.4f} ± {de:.4f} | {rel:+.1e} |\n")
        a, ea, b, eb, chi = fit
        f.write(f"\nFit Delta = a + b/N: a = {a:+.4f} ± {ea:.4f}, b = {b:+.3f} ± {eb:.3f}, chi2/dof = {chi:.2f}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--V0kac", type=float, nargs="+", default=[-150.0])
    p.add_argument("--N", type=int, nargs="+", default=[15, 20, 25, 30, 35, 40])
    p.add_argument("--walkers", type=int, default=1200)
    p.add_argument("--steps", type=int, default=32000)
    p.add_argument("--dtau", type=float, default=1e-3)
    p.add_argument("--plateau-N", type=int, default=40, help="N whose blocking curve is plotted")
    args = p.parse_args()
    os.makedirs("final_imag", exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, V in enumerate(args.V0kac):
        rows, curves = analyse(V, args.N, args.walkers, args.steps, args.dtau)
        if len(rows) < 2:
            continue
        fit = fit_delta(rows)
        write_table(V, rows, fit)
        print(f"\nV0 = {V:.0f}")
        print("   N     E_DMC/N        E_GP/N      Delta              Delta/|E_GP|")
        for N, E, e, Eg, d, de, rel in rows:
            print(f"  {N:2d}  {E:9.4f}±{e:.4f}  {Eg:9.4f}  {d:+.4f}±{de:.4f}  {rel:+.1e}")
        a, ea, b, eb, chi = fit
        print(f"  fit Delta = a + b/N: a = {a:+.4f} ± {ea:.4f}  b = {b:+.3f} ± {eb:.3f}  chi2/dof = {chi:.2f}")

        x = np.array([1 / r[0] for r in rows]); d = np.array([r[4] for r in rows]); s = np.array([r[5] for r in rows])
        ax.errorbar(x, d, yerr=s, fmt="o", ms=10, capsize=5, color=f"C{i}", label=fr"$V_0={V:.0f}\,\varepsilon$")
        # fit line only where a + b/N describes the data (chi2/dof < 3) AND the intercept is
        # physical (Delta <= 0, i.e. a not more than 2 sigma above 0); e.g. -40 has a = +3 sigma
        if chi < 3 and a <= 2 * ea:
            xx = np.linspace(0, x.max() * 1.05, 50)
            ax.plot(xx, a + b * xx, "--", color=f"C{i}", lw=2)
        else:                                          # otherwise just join the points
            ax.plot(x, d, ":", color=f"C{i}", lw=1.5)

        # blocking plateau for one N
        Np = args.plateau_N if args.plateau_N in curves else max(curves)
        c = np.array(curves[Np])
        f2, a2 = plt.subplots(figsize=(12, 8))
        a2.semilogx(c[:, 1] * args.dtau, c[:, 2], "o-", ms=9, lw=2)
        a2.axhline(max(e for nb, _, e in curves[Np] if nb >= MIN_BLOCKS), color="k", ls=":", lw=1.5)
        a2.set_xlabel(r"block length $\tau_b\,[\hbar/\varepsilon]$")
        a2.set_ylabel(r"$\sigma(E/N)\,[\varepsilon]$")
        a2.set_title(fr"$N={Np},\ V_0={V:.0f}\,\varepsilon$")
        a2.grid(True, ls="--", alpha=0.6)
        f2.tight_layout()
        f2.savefig(os.path.join("final_imag", f"blocking_V0{V:.0f}_N{Np}.pdf"), bbox_inches="tight")
        plt.close(f2)

    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(r"$(E_{\rm DMC}-E_{\rm GP})/N\,[\varepsilon]$")
    ax.set_xlim(left=0)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    tag = "_".join(f"{V:.0f}" for V in args.V0kac)
    fig.savefig(os.path.join("final_imag", f"gp_delta_vs_invN_V0{tag}.pdf"), bbox_inches="tight")
    print(f"\nSaved tables in {OUT}/ and figures in final_imag/")


if __name__ == "__main__":
    main()
