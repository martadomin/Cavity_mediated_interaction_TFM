"""
Map of |rho(n kL, n' kL)| / N from the files written by obdm_momentum.py, in the style of
plots.ipynb (LaTeX serif fonts, Blues colormap, PDF in final_imag/). One figure per V0_kac.

Usage (from the repository root):
    python python/plots/plot_rho_k.py                                  # N = 40, V0_kac = -40 -100 -200, lin
    python python/plots/plot_rho_k.py --estimator DMC --V0kac -60 -100
    python python/plots/plot_rho_k.py --nshow 5                        # show |n|, |n'| <= 5
Output: final_imag/rho_kk_N<N>_V0<V0kac>_<estimator>.pdf
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
    "legend.fontsize": 30,
    "figure.titlesize": 36,
    "text.usetex": True,
})

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=40)
p.add_argument("--V0kac", type=float, nargs="+", default=[-40.0, -100.0, -200.0])
p.add_argument("--estimator", choices=["VMC", "DMC", "lin"], default="lin")
p.add_argument("--nshow", type=int, default=6, help="show |n|, |n'| <= nshow")
args = p.parse_args()

os.makedirs("final_imag", exist_ok=True)
for V in args.V0kac:
    f = os.path.join("numpy_arrays_OBDM", f"momentum_N{args.N}_V0kac{V:.0f}.npz")
    if not os.path.isfile(f) or f"{args.estimator}_block" not in np.load(f):
        print(f"skipping V0_kac = {V:.0f}: no {args.estimator} data in {f} (run python/obdm_momentum.py)")
        continue
    d = np.load(f)
    ns = d[f"{args.estimator}_ns"]
    sel = np.abs(ns) <= args.nshow
    G = np.abs(d[f"{args.estimator}_block"])[np.ix_(sel, sel)] / args.N
    n = ns[sel]

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(G, origin="lower", extent=[n[0] - 0.5, n[-1] + 0.5, n[0] - 0.5, n[-1] + 0.5],
                   cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel(r"$n'$")
    ax.set_ylabel(r"$n$")
    ax.set_xticks(np.arange(-args.nshow, args.nshow + 1, 2))
    ax.set_yticks(np.arange(-args.nshow, args.nshow + 1, 2))
    ax.set_title(fr"$N={args.N},\ V_0={V:.0f}$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|\rho(nk_L,n'k_L)|/N$")
    plt.tight_layout()
    out = os.path.join("final_imag", f"rho_kk_N{args.N}_V0{V:.0f}_{args.estimator}.pdf")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)
