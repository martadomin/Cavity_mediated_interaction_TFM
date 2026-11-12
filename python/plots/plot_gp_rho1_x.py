"""
GP one-body density matrix rho^(1)(x,x') in real space for the symmetric two-branch (cat) state, N = 40,
over one cavity wavelength, at the couplings of python/plots/plot_gp_branches.py.

With phi_A(x) = sum_n c_n exp(2 pi i n x) (c_n from julia/gp/gp_scf.jl, normalised on one period) and
phi_B(x) = phi_A(x + lambda/2) (c_n -> (-1)^n c_n), s = <phi_A|phi_B>, density 1:
    rho(x,x') = [ phiA phiA' + phiB phiB' + s^(N-1) (phiA phiB' + phiB phiA') ] / (2 (1 + s^N))
(the same expression as python/plots/plot_gp_g1.py). The diagonal rho(x,x) is the cat density of
gp_branches_N40.pdf. rho is periodic in x and x' separately (no decay with |x - x'|: GP has full
off-diagonal long-range order), so one period in each variable shows all of it.
Bottom row: cuts rho(x, 0) and rho(x, lambda/2) (x' on a peak of branch A or B) and the density n(x) = rho(x, x).

Usage (from the repository root, after julia julia/gp/gp_scf.jl):
    python python/plots/plot_gp_rho1_x.py                       # V0 = -25 -40 -60 -150
    python python/plots/plot_gp_rho1_x.py --V0kac -30 -100
Output: final_imag/gp_rho1_x_N40.pdf (or gp_rho1_x_N40_V0<...>.pdf for a custom list)
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 24,
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
    x = np.linspace(0, 1, 401)
    i0, iq, ih = 0, 100, 200                                      # x' = 0, lambda/4, lambda/2

    Vs = [V for V in args.V0kac if np.isclose(cn[:, 0], V).any()]
    for V in set(args.V0kac) - set(Vs):
        print(f"V0 = {V:.0f}: not in {path}, skipped")
    fig, axes = plt.subplots(2, len(Vs), figsize=(9 * len(Vs), 17), height_ratios=[1, 0.75], squeeze=False)

    print(f"N = {N}:   V0     s       rho(0,0)  rho(0,l/2)  rho(l/4,l/4)  rho(0,l/2)/sqrt(n(0)n(l/2))")
    for j, V in enumerate(Vs):
        c = cn[np.isclose(cn[:, 0], V)][0, 1:]
        cB = c * (-1.0) ** n
        s = np.sum(c * cB)
        a, b = branch(c, n, x), branch(cB, n, x)
        rho = (np.outer(a, a) + np.outer(b, b) + s ** (N - 1) * (np.outer(a, b) + np.outer(b, a))) / (2 * (1 + s ** N))

        ax = axes[0, j]
        im = ax.imshow(rho, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=0, aspect="equal")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=24)
        cb.set_label(r"$\rho^{(1)}(x,x')\ [\lambda^{-1}]$", fontsize=28)
        ax.set_title(fr"$V_0={V:.0f}\,\varepsilon$,\ $s={s:.3f}$")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$x'/\lambda$")

        ax = axes[1, j]
        ax.plot(x, rho[:, i0], "-", color="C0", lw=3, label=r"$\rho^{(1)}(x,0)$")
        ax.plot(x, rho[:, ih], "--", color="C3", lw=3, label=r"$\rho^{(1)}(x,\lambda/2)$")
        ax.plot(x, np.diag(rho), ":", color="k", lw=2.5, label=r"$n(x)=\rho^{(1)}(x,x)$")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.6)
        print(f"        {V:6.0f}  {s:.4f}   {rho[0, 0]:.4f}    {rho[0, ih]:.4f}     {rho[iq, iq]:.4f}"
              f"        {rho[0, ih] / np.sqrt(rho[0, 0] * rho[ih, ih]):.4f}")
    axes[1, 0].set_ylabel(r"$\rho^{(1)}\ [\lambda^{-1}]$")
    h, lab = axes[1, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=30, frameon=False)

    os.makedirs("final_imag", exist_ok=True)
    tag = "" if args.V0kac == DEFAULT else "_V0" + "_".join(f"{V:.0f}" for V in Vs)
    out = os.path.join("final_imag", f"gp_rho1_x_N{N}{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
