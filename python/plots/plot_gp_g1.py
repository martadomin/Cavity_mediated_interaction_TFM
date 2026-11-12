"""
GP prediction for |g1(x,x')|^2 = rho(x,x')^2 / [n(x) n(x')] of the symmetric two-branch (cat) state,
in the layout of the g1sq map of plots.ipynb (n(x) on top, map below, one cavity period).

With phi_A(x) = sum_n c_n exp(i n kL x)/sqrt(L) (branch A) and phi_B(x) = phi_A(x + lambda/2) (branch B,
c_n -> (-1)^n c_n), s = <phi_A|phi_B>:
    rho(x,x') = N / (2 (1 + s^N)) [ phiA phiA' + phiB phiB' + s^(N-1) (phiA phiB' + phiB phiA') ]
A single branch (broken symmetry) would give |g1|^2 = 1 everywhere; the cat gives a checkerboard.

Usage (from the repository root, after julia julia/gp/gp_scf.jl):
    python python/plots/plot_gp_g1.py                        # N = 40, V0 = -100
    python python/plots/plot_gp_g1.py --V0 -60 -100 -200
Output: final_imag/g1sq_map_GP_N<N>_V0<V0>.pdf
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
    "figure.titlesize": 36,
    "text.usetex": True,
})

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=40)
p.add_argument("--V0", type=float, nargs="+", default=[-100.0])
p.add_argument("--nx", type=int, default=400, help="grid points per period")
args = p.parse_args()
N = args.N

d = np.loadtxt(os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv"), delimiter=",", skiprows=1)
ns = np.arange(-30, 31)
x = -0.5 + (np.arange(args.nx) + 0.5) / args.nx            # one period, lambda = 1
os.makedirs("final_imag", exist_ok=True)

for V in args.V0:
    row = d[np.isclose(d[:, 0], V)]
    if len(row) == 0:
        print(f"V0 = {V} not in the GP grid, skipped")
        continue
    c = row[0, 1:]
    s = np.sum((-1.0) ** ns * c ** 2)
    phiA = (c[None, :] * np.cos(2 * np.pi * np.outer(x, ns))).sum(axis=1)   # real: c_n = c_-n
    phiB = (c[None, :] * (-1.0) ** ns * np.cos(2 * np.pi * np.outer(x, ns))).sum(axis=1)
    pre = N / (2 * (1 + s ** N))
    rho = pre * (np.outer(phiA, phiA) + np.outer(phiB, phiB)
                 + s ** (N - 1) * (np.outer(phiA, phiB) + np.outer(phiB, phiA)))
    n_x = np.diag(rho)
    g1sq = rho ** 2 / np.outer(n_x, n_x)
    ia, ib = np.argmin(abs(x - 0.0)), np.argmin(abs(x - 0.5 + 1e-9)) % len(x)
    print(f"V0 = {V:.0f}: s = {s:.4f}, g1^2(well A, well A) = {g1sq[ia, ia]:.3f}, "
          f"g1^2(A, B) = {g1sq[ia, ib - 0]:.3f}")

    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(11, 13), height_ratios=[1, 3], sharex=True)
    ax0.plot(x, n_x / (N / 1.0), "-", color="C0", linewidth=2)           # n(x)/(N/L) = n(x)/rho
    ax0.set_ylabel(r"$n(x)$")
    ax0.set_title(fr"GP, $N={N},\ V_0={V:.0f}$")
    ax0.grid(True, linestyle="--", alpha=0.6)
    im = ax.imshow(g1sq, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel(r"$x/\lambda$", labelpad=20)
    ax.set_ylabel(r"$x'/\lambda$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|g^{(1)}(x,x')|^2$")
    plt.tight_layout()
    out = os.path.join("final_imag", f"g1sq_map_GP_N{N}_V0{V:.0f}.pdf")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)
