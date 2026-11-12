"""
Plots of the GP (SCF) results of julia/gp/gp_scf.jl, with the QMC data overlaid where it exists.
Same style as plots.ipynb (LaTeX serif fonts, PDF in final_imag/).

  gp_energy.pdf              E/N vs V0: GP (N = 40, N -> inf), VMC and DMC (N -> inf)
  gp_nk_N<N>_V0<V0>.pdf      n(n kL) = N c_n^2 (GP) vs QMC block diagonal
  gp_rho_cat_N<N>_V0<V0>.pdf  GP cat prediction |rho_nn'|/N = c_n c_n' (1 + (-1)^(n-n'))/2
  gp_lambda_N<N>.pdf         lambda_pm/N (GP cat) vs QMC lambda_0/N, lambda_1/N
  gp_overlap_N<N>.pdf        |sum_n c_n^pm d_n^pm|^2 per parity sector (QMC natural orbitals)

Usage (from the repository root, after julia julia/gp/gp_scf.jl [and python/obdm_momentum.py]):
    python python/plots/plot_gp.py
    python python/plots/plot_gp.py --N 40 --V0kac -40 -100 -200
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pump_axis import add_pump_axis  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 26,
    "figure.titlesize": 36,
    "text.usetex": True,
})

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=40)
p.add_argument("--V0kac", type=float, nargs="+", default=[-40.0, -100.0, -200.0])
p.add_argument("--nshow", type=int, default=6)
args = p.parse_args()
N = args.N
GP = "numpy_arrays_GP"
EST = {"VMC": ("o", "C1"), "DMC": ("s", "C2"), "lin": ("^", "C3")}
EST_LABEL = {"VMC": "VMC", "DMC": "DMC mixed", "lin": r"$2\,$DMC$-$VMC"}
os.makedirs("final_imag", exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join("final_imag", name), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved", os.path.join("final_imag", name))


def gp_orbitals(tag):
    d = np.loadtxt(os.path.join(GP, f"gp_cn_{tag}.csv"), delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1:], np.arange(-30, 31)


def qmc_npz(V):
    f = os.path.join("numpy_arrays_OBDM", f"momentum_N{N}_V0kac{V:.0f}.npz")
    return np.load(f) if os.path.isfile(f) else None


# ---------------------------------------------------------------- 1. energy
dense = np.loadtxt(os.path.join(GP, "gp_dense.csv"), delimiter=",", skiprows=1)
fig, ax = plt.subplots(figsize=(13, 10))
ax.plot(dense[:, 0], dense[:, 6], "k-", lw=2.5, label=r"GP, $N\to\infty$")
for method, (mk, col) in (("VMC", ("o", "C1")), ("DMC", ("s", "C2"))):
    f = f"numpy_arrays_{method}/thermodynamic_limit/constant_density_1.0/e_inf_order1_nmin0.csv"
    if os.path.isfile(f):
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        ax.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2], fmt=mk, color=col, ms=9, capsize=3,
                    label=fr"{method}, $N\to\infty$")
ax.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax.set_ylabel(r"$E/N\ [\varepsilon]$")
ax.set_xlim(-210, 0)
ax.legend()
ax.grid(True, ls="--", alpha=0.6)
save(fig, "gp_energy.pdf")

# ---------------------------------------------------------------- 2-3. n(k) and cat rho_nn'
V0s, C, nn = gp_orbitals(f"N{N}")
for V in args.V0kac:
    row = np.where(np.isclose(V0s, V))[0]
    if len(row) == 0:
        print(f"V0 = {V} not in the GP grid, skipped")
        continue
    c = C[row[0]]
    sel = np.abs(nn) <= args.nshow
    n = nn[sel]

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.bar(n, N * c[sel] ** 2, width=0.8, color="0.75", edgecolor="k", label=r"GP: $Nc_n^2$")
    q = qmc_npz(V)
    if q is not None:
        for est, (mk, col) in EST.items():
            if f"{est}_block_diag" in q:
                ns = q[f"{est}_ns"]
                m = np.abs(ns) <= args.nshow
                ax.plot(ns[m], q[f"{est}_block_diag"][m], mk, color=col, ms=11, mfc="none", mew=2,
                        label=EST_LABEL[est])
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2 * N)
    ax.set_xticks(np.arange(-args.nshow, args.nshow + 1, 2))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$n(nk_L)$")
    ax.set_title(fr"$N={N},\ V_0={V:.0f}$")
    ax.legend(loc="upper right")
    save(fig, f"gp_nk_N{N}_V0{V:.0f}.pdf")

    cat = np.outer(c[sel], c[sel]) * (1 + (-1.0) ** (n[:, None] - n[None, :])) / 2
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(np.abs(cat), origin="lower", extent=[n[0] - .5, n[-1] + .5, n[0] - .5, n[-1] + .5],
                   cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel(r"$n'$")
    ax.set_ylabel(r"$n$")
    ax.set_xticks(np.arange(-args.nshow, args.nshow + 1, 2))
    ax.set_yticks(np.arange(-args.nshow, args.nshow + 1, 2))
    ax.set_title(fr"GP (cat), $V_0={V:.0f}$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$|\rho_{nn'}|/N$")
    save(fig, f"gp_rho_cat_N{N}_V0{V:.0f}.pdf")

# ---------------------------------------------------------------- 4-5. lambda and overlaps
gp = np.loadtxt(os.path.join(GP, f"gp_scf_N{N}.csv"), delimiter=",", skiprows=1)
summ = os.path.join("numpy_arrays_OBDM", f"summary_N{N}.csv")
S = np.loadtxt(summ, delimiter=",", skiprows=1, ndmin=2) if os.path.isfile(summ) else None
names = ["VMC", "DMC", "lin"]

# QMC points only where the OBDM is usable: VMC and DMC mixed down to -150 (converged 2400-walker run;
# V0 = -200 dropped), the extrapolated estimator only down to -100 (at -150 it is unphysical,
# lambda0 + lambda1 > N). The GP cat curves cover the full range.
QMC_VMIN = {"VMC": -150.0, "DMC": -150.0, "lin": -100.0}
VC = 2 * np.pi ** 2 * N / (N - 1)                 # GP threshold |V0|_c at this N

fig, ax = plt.subplots(figsize=(13, 10))
ax.plot(dense[:, 0], dense[:, 9], "k-", lw=2.5, label=r"GP cat: $\lambda_+/N$")
ax.plot(dense[:, 0], dense[:, 10], "k--", lw=2.5, label=r"GP cat: $\lambda_-/N$")
if S is not None:
    for k, est in enumerate(names):
        r = S[(S[:, 1] == k) & (S[:, 0] >= QMC_VMIN[est])]
        mk, col = EST[est]
        ax.plot(r[:, 0], r[:, 5], mk, color=col, ms=11, mfc="none", mew=2, label=EST_LABEL[est] + r": $\lambda_0/N$")
        ax.plot(r[:, 0], r[:, 6], mk, color=col, ms=11, alpha=0.6, label=EST_LABEL[est] + r": $\lambda_1/N$")
ax.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax.set_ylabel(r"$\lambda/N$")
ax.legend(fontsize=20, ncol=2)
ax.grid(True, ls="--", alpha=0.6)
save(fig, f"gp_lambda_N{N}.pdf")

if S is not None:
    # overlaps only above threshold (below it the GP orbital has no odd part, so phi_- and the
    # "-" overlap are undefined), down to -150 for all estimators: at -150 the extrapolated OBDM has
    # unphysical eigenvalues (excluded from the lambda plot) but its orbitals are fine (overlap 0.99997)
    fig, ax = plt.subplots(figsize=(13, 10))
    for k, est in enumerate(names):
        r = S[(S[:, 1] == k) & (S[:, 0] >= -150.0) & (S[:, 0] < -VC)]
        mk, col = EST[est]
        ax.plot(r[:, 0], r[:, 9], mk + "-", color=col, ms=11, mfc="none", mew=2, label=EST_LABEL[est] + r": $+$")
        ax.plot(r[:, 0], r[:, 10], mk + "--", color=col, ms=11, label=EST_LABEL[est] + r": $-$")
    ax.set_xlabel(r"$V_0/\varepsilon$")
    add_pump_axis(ax)
    ax.set_ylabel(r"$|\sum_n c_n^{\pm} d_n^{\pm}|^2$")
    ax.set_ylim(0.94, 1.005)
    ax.legend(fontsize=20, ncol=2, loc="lower right")
    ax.grid(True, ls="--", alpha=0.6)
    save(fig, f"gp_overlap_N{N}.pdf")
