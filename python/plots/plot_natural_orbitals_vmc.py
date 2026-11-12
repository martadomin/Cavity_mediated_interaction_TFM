"""
Natural orbitals of the standalone VMC runs, N = 40, with GP for comparison (layout of the natural-orbital
cell of plots.ipynb: left densities on one period, right the orbitals).

Difference from the notebook: the orbitals are parity-projected. The condensate block rho(n kL, n' kL),
|n| <= 8, of the translation-averaged rho^(1) (python/obdm_momentum.analyse) is split into its even-n and
odd-n sectors and each sector is diagonalised: phi_+ (even n, period lambda/2) and phi_- (odd n) with their
own occupations lambda_+, lambda_-. (The notebook takes the two leading eigenvectors of rho^(1) directly;
when lambda_0 ~ lambda_1, e.g. at -200, they are an arbitrary rotation of phi_+, phi_-.)
Orbitals normalised as in the notebook, int |phi|^2 dx = 1 over the box L = N lambda (uniform: 1/sqrt(N)).
GP (dashed black): natural orbitals of the GP cat = normalised even / odd parts of the GP orbital at the same
N (same equations as julia/gp/gp_scf.jl), occupations lambda_pm/N = (1 +- s)(1 +- s^(N-1))/(2(1 + s^N)).
Left: n(x) (VMC histogram, mean 1), lambda_+ |phi_+|^2, lambda_- |phi_-|^2 and their sum, GP cat density.
Right: phi_+, phi_- (VMC, solid) and GP (dashed black); grey dotted (phi_+ +- phi_-)/sqrt(2).

Usage (from the repository root):
    python python/plots/plot_natural_orbitals_vmc.py                   # V0 = 0 -60 -90 -200
    python python/plots/plot_natural_orbitals_vmc.py --V0kac 0 -40 -150
Output: final_imag/natural_orbitals_vmc_N40.pdf (or ..._V0<...>.pdf for a custom list)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import _ground, NMAX  # noqa: E402
from obdm_momentum import analyse  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 22,
    "text.usetex": True,
})

N = 40
L = float(N)
BASE = os.path.join("numpy_arrays_VMC", "constant_density_1.0")
DEFAULT = [0.0, -60.0, -90.0, -200.0]


def load(kind, V0):
    v = V0 / N + 0.0
    return np.load(os.path.join(BASE, kind, f"V0={v:.3f}", f"{kind}_{N}_{v:.3f}_{L:.1f}.npy")).astype(float)


def gp_orbital(V0):
    """GP c_n (n = -NMAX..NMAX) at N; uniform (c_0 = 1) below threshold."""
    r = (N - 1) / N
    f = lambda A: A - V0 * r * _ground(A)[1]
    lo, hi = V0 * r, -1e-8
    if V0 >= 0 or f(lo) * f(hi) > 0:
        c = np.zeros(2 * NMAX + 1); c[NMAX] = 1.0
        return c
    return _ground(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))[0]


def field(c, n, x):
    return np.real(np.exp(2j * np.pi * np.outer(x, n)) @ c)


def sector_orbitals(R, ns):
    """leading eigenpair of the even-n and of the odd-n sector of the condensate block R (real part)."""
    out = {}
    for key, mask in (("plus", ns % 2 == 0), ("minus", ns % 2 != 0)):
        w, v = np.linalg.eigh(R.real[np.ix_(mask, mask)])
        d = np.zeros(len(ns)); d[mask] = v[:, -1]
        out[key] = (w[-1], d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=DEFAULT)
    args = p.parse_args()

    x = np.linspace(-0.5, 0.5, 401)
    ngp = np.arange(-NMAX, NMAX + 1)
    fig, axes = plt.subplots(len(args.V0kac), 2, figsize=(22, 6 * len(args.V0kac)), squeeze=False)
    print(f"N = {N}:  V0    VMC lambda+/N lambda-/N | GP lambda+/N lambda-/N | |<phi+|GP>|^2 |<phi-|GP>|^2")
    for r, V in enumerate(args.V0kac):
        ax_d, ax_o = axes[r]
        res = analyse(load("obdm", V), N, L, 8, (None, None))
        ns = res["ns"]
        so = sector_orbitals(res["block"], ns)
        lp, dp = so["plus"]
        lm, dm = so["minus"]

        # GP natural orbitals and cat occupations
        c = gp_orbital(V)
        cp, cm = c * (ngp % 2 == 0), c * (ngp % 2 != 0)
        s = float(np.sum(c * c * (-1.0) ** ngp))
        cp /= np.linalg.norm(cp)
        has_minus = np.linalg.norm(cm) > 1e-12
        cm = cm / np.linalg.norm(cm) if has_minus else np.where(np.abs(ngp) == 1, 1 / np.sqrt(2), 0.0)
        gl_p = (1 + s) * (1 + s ** (N - 1)) / (2 * (1 + s ** N))
        gl_m = 1 - gl_p
        # signs: QMC orbitals aligned with GP
        gq = lambda cc: np.array([cc[ngp == k][0] for k in ns])
        if np.dot(dp, gq(cp)) < 0: dp = -dp
        if np.dot(dm, gq(cm)) < 0: dm = -dm
        ovp, ovm = np.dot(dp, gq(cp)) ** 2, np.dot(dm, gq(cm)) ** 2

        box = 1 / np.sqrt(N)                        # one-period -> box normalisation
        fp, fm = field(dp, ns, x) * box, field(dm, ns, x) * box
        gpp, gpm = field(cp, ngp, x) * box, field(cm, ngp, x) * box

        # left: densities on one period (in units of the mean density 1/lambda)
        nx = load("hist_1D", V)
        xh = -0.5 + (np.arange(len(nx)) + 0.5) / len(nx)
        ax_d.plot(xh, nx / nx.mean(), "k-", lw=3, label=r"$n(x)$ (VMC)")
        ax_d.plot(x, lp * fp ** 2, "--", color="C0", lw=2.5, label=r"$\lambda_+|\phi_+|^2$")
        ax_d.plot(x, lm * fm ** 2, ":", color="C3", lw=3, label=r"$\lambda_-|\phi_-|^2$")
        ax_d.plot(x, lp * fp ** 2 + lm * fm ** 2, "-", color="0.6", lw=1.5, label=r"$\lambda_+|\phi_+|^2+\lambda_-|\phi_-|^2$")
        ax_d.plot(x, N * (gl_p * gpp ** 2 + gl_m * gpm ** 2), "--", color="k", lw=1.5, label="GP cat")
        ax_d.set_title(fr"$V_0={V:.0f}$, $\lambda_+/N={lp / N:.2f}$, $\lambda_-/N={lm / N:.2f}$ (GP: ${gl_p:.2f}$, ${gl_m:.2f}$)",
                       fontsize=26)
        ax_d.set_ylabel("density")
        ax_d.set_xlim(-0.5, 0.5)
        ax_d.grid(True, ls="--", alpha=0.6)

        # right: orbitals
        ax_o.plot(x, fp, "-", color="C0", lw=2.5, label=r"$\phi_+$ (even)")
        ax_o.plot(x, fm, "-", color="C3", lw=2.5, label=r"$\phi_-$ (odd)")
        ax_o.plot(x, gpp, "--", color="k", lw=1.8, label="GP")
        if has_minus or V == 0:
            ax_o.plot(x, gpm, "--", color="k", lw=1.8)
        ax_o.plot(x, (fp + fm) / np.sqrt(2), ":", color="0.45", lw=2, label=r"$(\phi_+\pm\phi_-)/\sqrt{2}$")
        ax_o.plot(x, (fp - fm) / np.sqrt(2), ":", color="0.45", lw=2)
        ax_o.axhline(0, color="k", lw=0.8)
        ax_o.set_xlim(-0.5, 0.5)
        ax_o.grid(True, ls="--", alpha=0.6)
        print(f"        {V:6.0f}   {lp / N:.4f}   {lm / N:.4f}  |   {gl_p:.4f}   {gl_m:.4f}  |   {ovp:.5f}    {ovm:.5f}")
    axes[0, 0].legend(loc="lower right", fontsize=20)
    axes[0, 1].legend(loc="lower right", fontsize=20)
    for ax in axes[-1]:
        ax.set_xlabel(r"$x/\lambda$")
    fig.tight_layout()
    os.makedirs("final_imag", exist_ok=True)
    tag = "" if args.V0kac == DEFAULT else "_V0" + "_".join(f"{V:.0f}" for V in args.V0kac)
    out = os.path.join("final_imag", f"natural_orbitals_vmc_N{N}{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
