"""
Side-by-side |rho(n kL, n' kL)|/N: GP cat prediction vs QMC, same colour scale.

GP cat (symmetric two-branch state, s^N -> 0):  rho_nn'/N = c_n c_n' (1 + (-1)^(n-n'))/2
QMC: condensate block from python/obdm_momentum.py (numpy_arrays_OBDM/momentum_N<N>_V0kac<V>.npz).

Usage (from the repository root, after julia julia/gp/gp_scf.jl and python/obdm_momentum.py):
    python python/plots/plot_rho_cat_compare.py                       # N = 40, V0 = -100, 2 DMC - VMC
    python python/plots/plot_rho_cat_compare.py --V0kac -80 --estimator DMC
Output: final_imag/rho_kk_vs_cat_N<N>_V0<V>_<estimator>.pdf
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
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "text.usetex": True,
})

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=40)
p.add_argument("--V0kac", type=float, default=-100.0)
p.add_argument("--estimator", choices=["VMC", "DMC", "lin"], default="lin")
p.add_argument("--nshow", type=int, default=6)
args = p.parse_args()
N, V = args.N, args.V0kac
LABEL = {"VMC": "VMC", "DMC": "DMC mixed", "lin": r"$2\,$DMC$-$VMC"}

# GP cat
d = np.loadtxt(os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv"), delimiter=",", skiprows=1)
c = d[np.isclose(d[:, 0], V)][0, 1:]
nn = np.arange(-30, 31)
sel = np.abs(nn) <= args.nshow
n = nn[sel]
cat = np.abs(np.outer(c[sel], c[sel]) * (1 + (-1.0) ** (n[:, None] - n[None, :])) / 2)

# QMC
q = np.load(os.path.join("numpy_arrays_OBDM", f"momentum_N{N}_V0kac{V:.0f}.npz"))
ns = q[f"{args.estimator}_ns"]
m = np.abs(ns) <= args.nshow
G = np.abs(q[f"{args.estimator}_block"])[np.ix_(m, m)] / N
assert np.array_equal(ns[m], n)

fig, axes = plt.subplots(1, 2, figsize=(21, 9.5), constrained_layout=True)
ext = [n[0] - 0.5, n[-1] + 0.5, n[0] - 0.5, n[-1] + 0.5]
vmax = max(cat.max(), G.max())                       # one common colour scale for both panels
for ax, M, title in ((axes[0], cat, r"GP cat"), (axes[1], G, LABEL[args.estimator])):
    im = ax.imshow(M, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xlabel(r"$n'$")
    ax.set_xticks(np.arange(-args.nshow, args.nshow + 1, 2))
    ax.set_yticks(np.arange(-args.nshow, args.nshow + 1, 2))
    ax.set_title(title)
axes[0].set_ylabel(r"$n$")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cb.set_label(r"$|\rho(nk_L,n'k_L)|/N$")
fig.suptitle(fr"$N={N},\ V_0={V:.0f}\,\varepsilon$")
os.makedirs("final_imag", exist_ok=True)
out = os.path.join("final_imag", f"rho_kk_vs_cat_N{N}_V0{V:.0f}_{args.estimator}.pdf")
fig.savefig(out, dpi=600, bbox_inches="tight")

odd = (n[:, None] - n[None, :]) % 2 == 1
print(f"max |rho_nn'|/N on odd n-n': GP cat {cat[odd].max():.2e}, QMC {G[odd].max():.2e}")
print(f"max | QMC - cat | on even n-n': {np.abs(G - cat)[~odd].max():.3f}")
print("Saved", out)
