"""
Bragg peak S(k_c)/N vs N at fixed V0: QMC (VMC, DMC mixed, 2 DMC - VMC) vs GP(N).

Same conventions as python/plots/plot_skc_gp.py: S(k) = <|rho_k|^2>/N, plotted S(k_c)/N = <|rho_k|^2>/N^2.
VMC arrays store <|rho_k|^2>, DMC arrays <|rho_k|^2>/N; both are converted before 2 DMC - VMC is formed.
GP at each N: S_GP(k_c)/N = [1 + (N-1) Theta(N)^2]/N, Theta(N) from the GP SCF at that N (same
equations as julia/gp/gp_scf.jl); the dotted line is the N -> inf limit Theta(inf)^2.
DMC run per point: longest imaginary time, then most walkers (filled = tau >= 32, open = older runs).

Usage (from the repository root):
    python python/plots/plot_skc_vs_N.py                      # V0 = -25 -30 -40 -60 -150
    python python/plots/plot_skc_vs_N.py --V0kac -40
Output: final_imag/Skc_vs_N_GP_V0<V>.pdf, numpy_arrays_GP/skc_vs_N_V0kac<V>.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import _ground  # noqa: E402

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

BV = os.path.join("numpy_arrays_VMC", "constant_density_1.0", "SSF")
BD = os.path.join("numpy_arrays_DMC", "constant_density_1.0", "SSF")


def gp_theta(V0, N):
    r = 1.0 if np.isinf(N) else (N - 1) / N
    f = lambda A: A - V0 * r * _ground(A)[1]
    lo, hi = V0 * r, -1e-8
    if V0 >= 0 or f(lo) * f(hi) > 0:
        return 0.0
    return _ground(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))[1]


def vmc_skc(Vk, N):
    v = Vk / N + 0.0
    f = os.path.join(BV, f"V0={v:.3f}", f"SSF_{N}_{v:.3f}_{float(N):.1f}.npy")
    return np.load(f).real[N - 1] / N ** 2 if os.path.isfile(f) else np.nan


def dmc_skc(Vk, N):
    """(S_DMC/N, S_lin/N, walkers, tau) for the longest / largest DMC run, or NaNs."""
    v = Vk / N + 0.0
    d = os.path.join(BD, f"V0={v:.3f}")
    pat = re.compile(rf"^SSF_(\d+)_([\d.]+)_{N}_(\d+)_{v:.3f}_{float(N):.1f}_IS\.npy$")
    best = None
    for f in (os.listdir(d) if os.path.isdir(d) else []):
        m = pat.match(f)
        if m:
            w, dt, st = int(m[1]), float(m[2]), int(m[3])
            key = (round(st * dt, 6), w)
            if best is None or key > best[0]:
                best = (key, f, w, st * dt)
    fv = os.path.join(d, f"SSF_vmc_{N}_{v:.3f}_{float(N):.1f}.npy")
    if best is None or not os.path.isfile(fv):
        return np.nan, np.nan, 0, np.nan
    sd = np.load(os.path.join(d, best[1])).real[N - 1]          # <|rho|^2>/N
    sv = np.load(fv).real[N - 1] / N                              # -> <|rho|^2>/N
    return sd / N, (2 * sd - sv) / N, best[2], best[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=[-25.0, -30.0, -40.0, -60.0, -150.0])
    args = p.parse_args()
    os.makedirs("final_imag", exist_ok=True)

    for Vk in args.V0kac:
        Ng = np.arange(10, 51)
        gp = np.array([(1 + (n - 1) * gp_theta(Vk, n) ** 2) / n for n in Ng])
        ginf = gp_theta(Vk, np.inf) ** 2
        Nv = np.arange(10, 51, 5)
        sv = np.array([vmc_skc(Vk, n) for n in Nv])
        Nd = [15, 20, 25, 30, 35, 40]
        D = [(n, *dmc_skc(Vk, n)) for n in Nd]

        fig, ax = plt.subplots(figsize=(13, 10))
        ax.plot(Ng, gp, "k-", lw=2.5, label=r"GP: $[1+(N-1)\Theta_N^2]/N$")
        ax.axhline(ginf, color="k", ls=":", lw=2, label=r"GP, $N\to\infty$: $\Theta_\infty^2$")
        m = np.isfinite(sv)
        if m.any():
            ax.plot(Nv[m], sv[m], "o", color="C1", ms=11, mfc="none", mew=2, label="VMC")
        # D rows: (N, S_DMC/N, S_lin/N, walkers, tau); filled = converged (tau >= 32), open = older runs
        for col_idx, (lab, mk, col) in ((1, (r"DMC mixed", "s", "C2")), (2, (r"$2\,$DMC$-$VMC", "^", "C3"))):
            for conv in (True, False):
                pts = [(d[0], d[col_idx]) for d in D
                       if np.isfinite(d[col_idx]) and ((d[4] >= 32 - 1e-9) == conv)]
                if pts:
                    x, y = zip(*pts)
                    ax.plot(x, y, mk, color=col, ms=12, mfc=col if conv else "none", mew=2,
                            label=lab + (r" ($\tau=32$)" if conv else r" (older runs)"))
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$S(k_c)/N$")
        ax.set_title(fr"$V_0={Vk:.0f}\,\varepsilon$")
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend(fontsize=20, loc="best")
        fig.tight_layout()
        out = os.path.join("final_imag", f"Skc_vs_N_GP_V0{Vk:.0f}.pdf")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        plt.close(fig)

        with open(os.path.join("numpy_arrays_GP", f"skc_vs_N_V0kac{Vk:.0f}.csv"), "w") as f:
            f.write("N,S_GP_over_N,S_VMC_over_N,S_DMC_over_N,S_lin_over_N,DMC_walkers,DMC_tau\n")
            for n, sd, sl, w, tau in D:
                g = (1 + (n - 1) * gp_theta(Vk, n) ** 2) / n
                vv = vmc_skc(Vk, n)
                f.write(f"{n},{g:.6g},{vv:.6g},{sd:.6g},{sl:.6g},{w},{tau:g}\n")
        print(f"\nV0 = {Vk:.0f}   (GP N->inf: {ginf:.4f})")
        print("   N    GP      VMC     DMC mix  2DMC-VMC   lin-GP   [run]")
        for n, sd, sl, w, tau in D:
            g = (1 + (n - 1) * gp_theta(Vk, n) ** 2) / n
            print(f"  {n:2d}  {g:.4f}  {vmc_skc(Vk, n):.4f}  {sd:.4f}   {sl:.4f}   {sl - g:+.4f}   [{w}w, tau={tau:g}]")
        print("Saved", out)



if __name__ == "__main__":
    main()
