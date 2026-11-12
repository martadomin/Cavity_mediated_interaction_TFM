"""
The two GP branches in real space: |phi_A(x)|^2 and |phi_B(x)|^2 vs x, N = 40.

phi_A(x) = sum_n c_n exp(2 pi i n x) = c_0 + 2 sum_{n>0} c_n cos(2 pi n x)   (c_n from julia/gp/gp_scf.jl,
real and c_{-n} = c_n), and phi_B has c^B_n = (-1)^n c_n, i.e. phi_B(x) = phi_A(x + 1/2): the branch
shifted by half a cavity wavelength (cos 2 pi x -> -cos 2 pi x). Both are normalised to one on a
period, so |phi|^2 = 1 is the uniform density. The dotted line is the cat-state density
(|phi_A|^2 + |phi_B|^2)/2, exact up to terms ~ s^N (s = <phi_A|phi_B>).

Usage (from the repository root, after julia julia/gp/gp_scf.jl):
    python python/plots/plot_gp_branches.py                       # V0 = -25 -40 -60 -150
    python python/plots/plot_gp_branches.py --V0kac -30 -80
Output: final_imag/gp_branches_N40.pdf (or gp_branches_N40_V0<...>.pdf for a custom list)
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


def branch(c, n, x):
    return np.real(np.exp(2j * np.pi * np.outer(x, n)) @ c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=DEFAULT)
    args = p.parse_args()

    path = os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv")
    with open(path) as f:
        n = np.array([int(v) for v in f.readline().strip().split(",")[1:]])
    cn = np.loadtxt(path, delimiter=",", skiprows=1)
    x = np.linspace(0, 2, 801)                                   # two cavity wavelengths

    Vs = [V for V in args.V0kac if np.isclose(cn[:, 0], V).any()]
    for V in set(args.V0kac) - set(Vs):
        print(f"V0 = {V:.0f}: not in {path}, skipped")
    ncol = 2 if len(Vs) > 1 else 1
    nrow = int(np.ceil(len(Vs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13 * ncol, 9 * nrow), sharex=True, squeeze=False)

    print(f"N = {N}:   V0    s = <A|B>   max|phi_A|^2   min|phi_A|^2")
    for ax, V in zip(axes.flat, Vs):
        c = cn[np.isclose(cn[:, 0], V)][0, 1:]
        cB = c * (-1.0) ** n
        rA, rB = branch(c, n, x) ** 2, branch(cB, n, x) ** 2
        s = np.sum(c * cB)
        ax.plot(x, rA, "-", color="C0", lw=3, label=r"$|\varphi_A(x)|^2$")
        ax.plot(x, rB, "--", color="C3", lw=3, label=r"$|\varphi_B(x)|^2$")
        ax.plot(x, (rA + rB) / 2, ":", color="k", lw=2.5, label=r"cat: $(|\varphi_A|^2+|\varphi_B|^2)/2$")
        ax.axhline(1, color="0.6", lw=1)
        ax.set_title(fr"$V_0={V:.0f}\,\varepsilon$,\ $s={s:.3f}$")
        ax.grid(True, ls="--", alpha=0.6)
        ax.set_ylim(bottom=0)
        print(f"        {V:6.0f}   {s:.4f}     {rA.max():.4f}        {rA.min():.4f}")
    for ax in axes.flat[len(Vs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"$x/\lambda$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"density $\times\,\lambda$")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94 if nrow > 1 else 0.88))
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=30, frameon=False)

    os.makedirs("final_imag", exist_ok=True)
    tag = "" if args.V0kac == DEFAULT else "_V0" + "_".join(f"{V:.0f}" for V in Vs)
    out = os.path.join("final_imag", f"gp_branches_N{N}{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
