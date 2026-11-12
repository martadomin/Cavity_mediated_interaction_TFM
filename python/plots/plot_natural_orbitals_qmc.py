"""
Natural orbitals phi_+ (even) and phi_- (odd) from DMC, N = 40, compared with GP (layout of the natural-orbital
cell of plots.ipynb: left densities on one period, right the orbitals).

QMC: converged DMC runs (most recent OBDM file = 1200 walkers, tau = 32; 2400 walkers at -150), estimators
DMC mixed and extrapolated 2 DMC - VMC (python/obdm_momentum.estimator_matrix). The condensate block
rho(n kL, n' kL), |n| <= 8, of the translation-averaged rho^(1) (obdm_momentum.analyse) is split into its
even-n and odd-n sectors and each sector is diagonalised: phi_+ and phi_- with occupations lambda_+, lambda_-
(parity projection: well defined also when lambda_+ ~ lambda_-).
GP (dashed black): natural orbitals of the GP cat = normalised even / odd parts of the GP orbital at N = 40
(same equations as julia/gp/gp_scf.jl), occupations lambda_pm/N = (1 +- s)(1 +- s^(N-1))/(2(1 + s^N)).
Orbitals normalised to int |phi|^2 dx = 1 over the box L = N lambda (uniform: 1/sqrt(N)), as in the notebook.
Left: densities from the DMC mixed block, n(x) = lambda_+ |phi_+|^2 + lambda_- |phi_-|^2 (+ depletion,
< 0.5 %), the extrapolated density n(x) = 2 n_DMC - n_VMC (diagonal of the extrapolated block, markers;
linear in the data, so unbiased to first order) and the GP cat density. The extrapolated occupations are not used (unphysical at -150,
lambda_+ + lambda_- = 1.06 N); its orbitals are.
Right: phi_+, phi_- (DMC mixed, solid; extrapolated, markers) and GP (dashed black).

Usage (from the repository root):
    python python/plots/plot_natural_orbitals_qmc.py                    # V0 = -25 -40 -60 -150
Output: final_imag/natural_orbitals_dmc_N40.pdf (or ..._V0<...>.pdf for a custom list)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import _ground, NMAX  # noqa: E402
from obdm_momentum import analyse, estimator_matrix  # noqa: E402

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
BASE = os.path.join("numpy_arrays_DMC", "constant_density_1.0")
DEFAULT = [-25.0, -40.0, -60.0, -150.0]


def gp_orbital(V0):
    r = (N - 1) / N
    f = lambda A: A - V0 * r * _ground(A)[1]
    lo, hi = V0 * r, -1e-8
    if V0 >= 0 or f(lo) * f(hi) > 0:
        return None
    return _ground(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))[0]


def field(c, n, x):
    return np.real(np.exp(2j * np.pi * np.outer(x, n)) @ c)


def sectors(est, V):
    """{plus, minus: (lambda, d)} from the parity sectors of the condensate block, and ns."""
    rho, _ = estimator_matrix(BASE, N, V, est)
    res = analyse(rho, N, L, 8, (None, None))
    R, ns = res["block"].real, res["ns"]
    out = {}
    for key, mask in (("plus", ns % 2 == 0), ("minus", ns % 2 != 0)):
        w, v = np.linalg.eigh(R[np.ix_(mask, mask)])
        d = np.zeros(len(ns)); d[mask] = v[:, -1]
        out[key] = (w[-1], d)
    return out, ns, res["block"]


def block_density(R, ns, x):
    """n(x) [1/lambda] from the condensate block: sum_nn' R_nn' exp(2 pi i (n - n') x) / N."""
    F = np.exp(2j * np.pi * np.outer(x, ns))
    return np.real(np.einsum("in,nm,im->i", F, R, F.conj())) / N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=DEFAULT)
    args = p.parse_args()

    x = np.linspace(-0.5, 0.5, 401)
    xm = np.linspace(-0.5, 0.5, 21)
    ngp = np.arange(-NMAX, NMAX + 1)
    box = 1 / np.sqrt(N)
    Vs = [V for V in args.V0kac if gp_orbital(V) is not None]
    fig, axes = plt.subplots(len(Vs), 2, figsize=(22, 6 * len(Vs)), squeeze=False)
    print(f"N = {N}:  V0   lambda+/N lambda-/N: DMC  |  GP   | |<phi+|GP>|^2 DMC  lin | |<phi-|GP>|^2 DMC  lin")
    for r, V in enumerate(Vs):
        ax_d, ax_o = axes[r]
        c = gp_orbital(V)
        cp, cm = c * (ngp % 2 == 0), c * (ngp % 2 != 0)
        s = float(np.sum(c * c * (-1.0) ** ngp))
        cp, cm = cp / np.linalg.norm(cp), cm / np.linalg.norm(cm)
        gl_p = (1 + s) * (1 + s ** (N - 1)) / (2 * (1 + s ** N))
        gl_m = 1 - gl_p
        gpp, gpm = field(cp, ngp, x) * box, field(cm, ngp, x) * box

        orb, ov, blk = {}, {}, {}
        for est in ("DMC", "lin"):
            sec, ns, blk[est] = sectors(est, V)
            gq = {"plus": np.array([cp[ngp == k][0] for k in ns]), "minus": np.array([cm[ngp == k][0] for k in ns])}
            for key in ("plus", "minus"):
                lam, d = sec[key]
                if np.dot(d, gq[key]) < 0:
                    d = -d
                orb[(est, key)] = (lam, d, ns)
                ov[(est, key)] = np.dot(d, gq[key]) ** 2

        lp, dp, ns = orb[("DMC", "plus")]
        lm, dm, _ = orb[("DMC", "minus")]
        fp, fm = field(dp, ns, x) * box, field(dm, ns, x) * box
        ax_d.plot(x, lp * fp ** 2 + lm * fm ** 2, "-", color="k", lw=3, label=r"$n(x)=\lambda_+|\phi_+|^2+\lambda_-|\phi_-|^2$")
        ax_d.plot(x, lp * fp ** 2, "--", color="C0", lw=2.5, label=r"$\lambda_+|\phi_+|^2$")
        ax_d.plot(x, lm * fm ** 2, ":", color="C3", lw=3, label=r"$\lambda_-|\phi_-|^2$")
        ax_d.plot(xm, block_density(blk["lin"], ns, xm), "^", color="k", ms=10, mfc="none", mew=2,
                  label=r"$n(x)$, $2\,$DMC$-$VMC")
        ax_d.plot(x, N * (gl_p * gpp ** 2 + gl_m * gpm ** 2), "--", color="0.45", lw=2, label="GP cat")
        ax_d.set_title(fr"$V_0={V:.0f}$: DMC $\lambda_\pm/N={lp / N:.2f},\,{lm / N:.2f}$ (GP ${gl_p:.2f},\,{gl_m:.2f}$)",
                       fontsize=26)
        ax_d.set_ylabel("density")
        ax_d.set_xlim(-0.5, 0.5)
        ax_d.set_ylim(bottom=0)
        ax_d.grid(True, ls="--", alpha=0.6)

        for key, col, lab in (("plus", "C0", r"$\phi_+$"), ("minus", "C3", r"$\phi_-$")):
            _, d, _ = orb[("DMC", key)]
            ax_o.plot(x, field(d, ns, x) * box, "-", color=col, lw=2.5, label=lab + " DMC mixed")
            _, dl, nsl = orb[("lin", key)]
            ax_o.plot(xm, field(dl, nsl, xm) * box, "^", color=col, ms=10, mfc="none", mew=2,
                      label=lab + r" $2\,$DMC$-$VMC")
        ax_o.plot(x, gpp, "--", color="k", lw=1.8, label="GP")
        ax_o.plot(x, gpm, "--", color="k", lw=1.8)
        ax_o.axhline(0, color="k", lw=0.8)
        ax_o.set_title(fr"overlap with GP: DMC ${ov[('DMC', 'plus')]:.4f},\,{ov[('DMC', 'minus')]:.4f}$; "
                       fr"extrap. ${ov[('lin', 'plus')]:.4f},\,{ov[('lin', 'minus')]:.4f}$", fontsize=24)
        ax_o.set_xlim(-0.5, 0.5)
        ax_o.grid(True, ls="--", alpha=0.6)
        print(f"        {V:5.0f}   {lp / N:.4f} {lm / N:.4f}  | {gl_p:.4f} {gl_m:.4f} |   {ov[('DMC', 'plus')]:.5f} "
              f"{ov[('lin', 'plus')]:.5f} |   {ov[('DMC', 'minus')]:.5f} {ov[('lin', 'minus')]:.5f}")
    axes[0, 0].legend(loc="lower right", fontsize=19)
    axes[0, 1].legend(loc="lower right", fontsize=19, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel(r"$x/\lambda$")
    fig.tight_layout()
    os.makedirs("final_imag", exist_ok=True)
    tag = "" if args.V0kac == DEFAULT else "_V0" + "_".join(f"{V:.0f}" for V in Vs)
    out = os.path.join("final_imag", f"natural_orbitals_dmc_N{N}{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
