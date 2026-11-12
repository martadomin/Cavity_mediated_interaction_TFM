"""
Lowdin orbitals chi_{A,B} = (phi_+ +- phi_-)/sqrt(2) in real space: QMC natural orbitals vs GP
(layout of python/plots/plot_gp_branches.py), N = 40. Like is compared with like:

QMC (markers): phi_+ = sum_{n even} d+_n exp(2 pi i n x), phi_- = sum_{n odd} d-_n ..., the even and odd
natural orbitals of the condensate block of rho^(1) (python/obdm_momentum.py,
numpy_arrays_OBDM/momentum_N40_V0kac<V>.npz, |n| <= 8, real, d_{-n} = d_n, normalised).
GP (lines): phi_+ and phi_- = the even / odd parts of the GP orbital phi_A (julia/gp/gp_scf.jl), each
normalised (the orbitals of julia/gp/two_mode.jl).
chi_B(x) = chi_A(x + lambda/2). The sign of each QMC phi_pm is fixed by a positive overlap with the GP one.
(These are the two-mode orbitals, not the GP branch phi_A itself: phi_A weights phi_pm by sqrt((1 +- s)/2).)
QMC estimators: VMC, DMC mixed and the natural orbitals of the extrapolated OBDM 2 rho_DMC - rho_VMC
(at -150 its eigenvalues are unphysical, lambda_0 + lambda_1 = 1.06 N, but its orbitals are not: they
remove the first-order trial-function bias of the DMC orbitals and match GP to 0.99997).

Usage (from the repository root, after julia julia/gp/gp_scf.jl and python python/obdm_momentum.py):
    python python/plots/plot_qmc_branches.py                    # V0 = -25 -40 -60 -150
    python python/plots/plot_qmc_branches.py --V0kac -30 -80
Output: final_imag/qmc_branches_N40.pdf (or qmc_branches_N40_V0<...>.pdf for a custom list)
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

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

N = 40
DEFAULT = [-25.0, -40.0, -60.0, -150.0]
EST = {"VMC": ("o", "C1", "VMC"), "DMC": ("s", "C2", "DMC mixed"), "lin": ("^", "C4", r"$2\,$DMC$-$VMC")}


def field(c, n, x):
    return np.real(np.exp(2j * np.pi * np.outer(x, n)) @ c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=DEFAULT)
    args = p.parse_args()

    path = os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv")
    with open(path) as f:
        ngp = np.array([int(v) for v in f.readline().strip().split(",")[1:]])
    cn = np.loadtxt(path, delimiter=",", skiprows=1)
    x = np.linspace(0, 2, 801)
    xm = np.linspace(0, 2, 41)                                    # marker positions

    Vs = [V for V in args.V0kac
          if np.isclose(cn[:, 0], V).any() and os.path.isfile(f"numpy_arrays_OBDM/momentum_N{N}_V0kac{V:.0f}.npz")]
    for V in [V for V in args.V0kac if V not in Vs]:
        print(f"V0 = {V:.0f}: no GP or OBDM data, skipped")
    ncol = 2 if len(Vs) > 1 else 1
    nrow = int(np.ceil(len(Vs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13 * ncol, 9 * nrow), sharex=True, squeeze=False)

    print(f"N = {N}:   V0   estimator   max|chi_A|^2 (GP)     |<chi_A^QMC|chi_A^GP>|^2")
    for ax, V in zip(axes.flat, Vs):
        c = cn[np.isclose(cn[:, 0], V)][0, 1:]
        s = np.sum(c * c * (-1.0) ** ngp)
        gpp, gpm = c * (ngp % 2 == 0), c * (ngp % 2 != 0)
        gpp, gpm = gpp / np.linalg.norm(gpp), gpm / np.linalg.norm(gpm)
        gA, gB = (gpp + gpm) / np.sqrt(2), (gpp - gpm) / np.sqrt(2)
        rA, rB = field(gA, ngp, x) ** 2, field(gB, ngp, x) ** 2
        ax.plot(x, rA, "-", color="C0", lw=2.5, label=r"GP $|\chi_A|^2$")
        ax.plot(x, rB, "--", color="C3", lw=2.5, label=r"GP $|\chi_B|^2$")
        ax.axhline(1, color="0.6", lw=1)

        d = np.load(f"numpy_arrays_OBDM/momentum_N{N}_V0kac{V:.0f}.npz")
        for est, (mk, col, lab) in EST.items():
            ns = d[f"{est}_ns"]
            pp, pm = d[f"{est}_phi_plus"], d[f"{est}_phi_minus"]
            cgp = np.array([c[ngp == k][0] if (ngp == k).any() else 0.0 for k in ns])
            if np.dot(pm, np.where(ns % 2 != 0, cgp, 0.0)) < 0:
                pm = -pm
            if np.dot(pp, np.where(ns % 2 == 0, cgp, 0.0)) < 0:
                pp = -pp
            cA, cBq = (pp + pm) / np.sqrt(2), (pp - pm) / np.sqrt(2)
            ax.plot(xm, field(cA, ns, xm) ** 2, mk, color=col, ms=10, mfc="none", mew=2,
                    label=fr"{lab} $|\chi_A|^2$")
            ax.plot(xm, field(cBq, ns, xm) ** 2, mk, color=col, ms=10, mew=2, alpha=0.7,
                    label=fr"{lab} $|\chi_B|^2$")
            gAq = np.array([gA[ngp == k][0] if (ngp == k).any() else 0.0 for k in ns])
            ov = np.dot(cA, gAq) ** 2
            print(f"        {V:6.0f}   {est:4s}       {np.max(field(cA, ns, x) ** 2):.4f} ({rA.max():.4f})"
                  f"       {ov:.4f}")
        ax.set_title(fr"$V_0={V:.0f}\,\varepsilon$,\ $s={s:.3f}$")
        ax.grid(True, ls="--", alpha=0.6)
        ax.set_ylim(bottom=0)
    for ax in axes.flat[len(Vs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"$x/\lambda$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"density $\times\,\lambda$")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.90 if nrow > 1 else 0.80))
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=26, frameon=False)

    os.makedirs("final_imag", exist_ok=True)
    tag = "" if args.V0kac == DEFAULT else "_V0" + "_".join(f"{V:.0f}" for V in Vs)
    out = os.path.join("final_imag", f"qmc_branches_N{N}{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
