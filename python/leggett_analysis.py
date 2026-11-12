"""
Superfluid fraction from the Leggett bound for the VMC / DMC production runs, compared
between estimators, together with the condensate occupations λ₀/N, λ₁/N.

Leggett upper bound (1D, density periodic with the cavity period λ):
    ρ_S/ρ ≤ 1 / (⟨n⟩ ⟨1/n⟩),   averages over one cell.

Density n(x): runs made after the n_cell output was added store the density folded into one
cavity period (x mod λ, 200 bins, every step and every particle): that is used when present.
For the older runs (n(x) only over the whole box with 100 bins, too coarse for the Leggett
bound, which is dominated by the density minima) n(x) = ρ₁(x, x) is taken instead from the
diagonal of the one-body density matrix (25 bins per λ; few samples, hence noisy minima),
folded into one cell and averaged over the exact symmetries of the ground state:
translations by λ, reflection x → −x and the Z₂ shift x → x + λ/2 (linear interpolation,
12.5 bins). Checked against the VMC hist_1D (100 bins/cell): agreement within ~0.01–0.02.

Estimators of n(x):
    VMC          n_VMC
    DMC mixed    n_DMC
    linear       2 n_DMC − n_VMC
    quadratic    n_DMC² / n_VMC
n(x) is a diagonal observable, so both extrapolations are applied bin by bin.

ρ_S/ρ is extrapolated to N → ∞ with an unweighted linear fit in 1/N (error from the
residuals). λ₀/N and λ₁/N at N → ∞ are read from python/obdm_analysis.py output.

Usage (from the repository root, after python/obdm_analysis.py):
    python python/leggett_analysis.py
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obdm_analysis import load_raw, symmetrize, fit_inverse_N  # noqa: E402

METHODS = ["VMC", "DMC", "lin", "quad"]
SHORT = {"VMC": "VMC", "DMC": "DMC mixed", "lin": "linear extr.", "quad": "quadratic extr."}
TITLE = {"VMC": "VMC", "DMC": "DMC (mixed)",
         "lin": r"linear extrapolation $2\langle O\rangle_{\rm DMC}-\langle O\rangle_{\rm VMC}$",
         "quad": r"quadratic extrapolation $\langle O\rangle_{\rm DMC}^2/\langle O\rangle_{\rm VMC}$"}
FLOOR = 1e-8


def cell_density(rho, L):
    """n(x) over one cell from the diagonal of a (symmetrized, λ-averaged) ρ₁, normalized to ⟨n⟩ = 1."""
    nb = rho.shape[0]
    per = nb // int(round(L))
    c = np.diag(rho)[: per * int(round(L))].reshape(int(round(L)), per).mean(0)
    c = 0.5 * (c + c[::-1])                                         # x → −x
    xs = np.arange(per)
    c = 0.5 * (c + np.interp((xs + per / 2) % per, xs, c, period=per))   # x → x + λ/2
    return c / c.mean()


def load_cell(base, N, V0kac, which, density=1.0):
    """Directly accumulated density folded into one cell (n_cell/ncell_*), symmetrized under
    x → −x and x → x + λ/2, normalized to mean 1. None if the run did not store it."""
    V0 = V0kac / N
    L = N / density
    d = os.path.join(base, "n_cell", f"V0={V0:.3f}")
    if which == "VMC":
        f = os.path.join(d, f"ncell_vmc_{N}_{V0:.3f}_{L:.1f}.npy")
        files = [f] if os.path.isfile(f) else []
    else:
        files = glob.glob(os.path.join(d, f"ncell_*_{N}_*_{V0:.3f}_{L:.1f}_IS.npy"))
    if not files:
        return None
    c = np.load(max(files, key=os.path.getmtime)).astype(float)
    nb = len(c)
    # bin centres (j + ½)/nb − ½ → reflection maps j → nb−1−j; λ/2 shift = nb/2 bins
    c = 0.5 * (c + c[::-1])
    if nb % 2 == 0:
        c = 0.5 * (c + np.roll(c, nb // 2))
    return c / c.mean()


def leggett(n):
    """1/(⟨n⟩⟨1/n⟩) and the number of bins clipped at FLOOR (n ≤ 0)."""
    clipped = int(np.sum(n <= FLOOR))
    n = np.maximum(n, FLOOR)
    return 1.0 / (n.mean() * (1.0 / n).mean()), clipped


def density_estimators(n_v, n_d):
    small = n_v <= FLOOR
    quad = np.where(small, 2 * n_d - n_v, n_d ** 2 / np.where(small, 1.0, n_v))
    return {"VMC": n_v, "DMC": n_d, "lin": 2 * n_d - n_v, "quad": quad}


def load_occupations(path):
    """{(estimator, V0_kac): (λ0/N, err, λ1/N, err)} at N → ∞ from obdm_analysis.py."""
    occ = {}
    if os.path.isfile(path):
        for r in csv.DictReader(open(path)):
            occ[(r["estimator"], float(r["V0_kac"]))] = tuple(float(r[k]) for k in
                                                              ("n0_inf", "n0_inf_err", "n1_inf", "n1_inf_err"))
    return occ


def load_occupations_N(path):
    """{(estimator, V0_kac, N): (λ0/N, λ1/N)} from obdm_analysis.py."""
    occ = {}
    if os.path.isfile(path):
        for r in csv.DictReader(open(path)):
            occ[(r["estimator"], float(r["V0_kac"]), int(r["N"]))] = (float(r["n0"]), float(r["n1"]))
    return occ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--density", type=float, default=1.0)
    p.add_argument("--N", type=int, nargs="+", default=[15, 20, 25, 30, 35, 40])
    p.add_argument("--V0kac", type=float, nargs="+",
                   default=[-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0, -200.0])
    args = p.parse_args()

    folder = f"constant_density_{args.density:.1f}"
    base = os.path.join("numpy_arrays_DMC", folder)
    obdm_dir = os.path.join("numpy_arrays_DMC", "obdm_analysis", folder)
    out_dir = os.path.join("numpy_arrays_DMC", "leggett_analysis", folder)
    img_dir = os.path.join("imag_DMC", "leggett_analysis", folder)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    rows = []                                   # estimator, V0_kac, N, rho_s, clipped bins
    densities = {}
    source = {}                                 # (V0_kac, N) -> density source
    for Vk in args.V0kac:
        for N in args.N:
            rv, L = load_raw(base, N, Vk, "VMC", args.density)
            rd, _ = load_raw(base, N, Vk, "DMC", args.density)
            if rv is None or rd is None:
                continue
            n_v, n_d = load_cell(base, N, Vk, "VMC", args.density), load_cell(base, N, Vk, "DMC", args.density)
            if n_v is None or n_d is None:                       # older runs: OBDM diagonal
                n_v = cell_density(symmetrize(rv, L), L)
                n_d = cell_density(symmetrize(rd, L), L)
                source[(Vk, N)] = "OBDM diag"
            else:
                source[(Vk, N)] = "n_cell"
            for m, n in density_estimators(n_v, n_d).items():
                rs, clipped = leggett(n)
                rows.append((m, Vk, N, rs, clipped))
                densities[(m, Vk, N)] = n
    if not rows:
        print("No data found under", base)
        return
    direct = sorted({Vk for (Vk, N), s in source.items() if s == "n_cell"})
    print("density source: directly accumulated n(x mod λ) for V0_kac =", direct or "none",
          "; OBDM diagonal for the rest")

    # --- N → ∞ ---
    Nmax = max(args.N)
    extrap = {}
    print(f"{'V0_kac':>7} " + "".join(f"{SHORT[m]:>24}" for m in METHODS) + "      (ρ_S/ρ, N → ∞;  clipped bins at N=%d)" % Nmax)
    for Vk in args.V0kac:
        line = f"{Vk:7.0f} "
        for m in METHODS:
            sel = sorted([r for r in rows if r[0] == m and r[1] == Vk], key=lambda r: r[2])
            if len(sel) >= 3:
                a, e, _ = fit_inverse_N([r[2] for r in sel], np.array([r[3] for r in sel]))
            else:
                a = e = np.nan
            extrap[(m, Vk)] = (a, e)
            clip = [r[4] for r in sel if r[2] == Nmax]
            line += f"{a:13.4f} ± {e:.4f}" + (f" [{clip[0]}]" if clip and clip[0] else "    ")
        print(line)

    with open(os.path.join(out_dir, "leggett.csv"), "w") as f:
        f.write("estimator,V0_kac,N,rho_s,clipped_bins\n")
        for m, Vk, N, rs, c in rows:
            f.write(f"{m},{Vk},{N},{rs:.6f},{c}\n")
    with open(os.path.join(out_dir, "leggett_Ninf.csv"), "w") as f:
        f.write("estimator,V0_kac,rho_s_inf,rho_s_inf_err\n")
        for (m, Vk), (a, e) in extrap.items():
            f.write(f"{m},{Vk},{a:.6f},{e:.6f}\n")

    occ = load_occupations(os.path.join(obdm_dir, "obdm_occupations_Ninf.csv"))
    occN = load_occupations_N(os.path.join(obdm_dir, "obdm_occupations.csv"))
    if not occ:
        print("(run python/obdm_analysis.py first to add λ0/N, λ1/N to the plots)")

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 16})
    cmap = plt.cm.viridis
    Ns = sorted(args.N)

    # --- plot 1: per estimator, ρ_S/ρ + λ0/N + λ1/N, thin line per N, N → ∞ in black ---
    for m in METHODS:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        for i, N in enumerate(Ns):
            c = cmap(i / max(1, len(Ns) - 1))
            pts = sorted([(r[1], r[3]) for r in rows if r[0] == m and r[2] == N])
            if pts:
                V, rs = map(np.array, zip(*pts))
                ax.plot(V, rs, "-", color=c, lw=1, alpha=0.55)
            pts = sorted([(Vk, *occN[(m, Vk, N)]) for Vk in args.V0kac if (m, Vk, N) in occN])
            if pts:
                V, l0, l1 = map(np.array, zip(*pts))
                ax.plot(V, l0, "--", color=c, lw=1, alpha=0.55)
                ax.plot(V, l1, ":", color=c, lw=1, alpha=0.55)
        pts = sorted([(Vk, *extrap[(m, Vk)]) for Vk in args.V0kac if (m, Vk) in extrap])
        V, a, e = map(np.array, zip(*pts))
        ax.errorbar(V, a, yerr=e, fmt="o-", color="black", ms=7, lw=2.2, zorder=5)
        pts = sorted([(Vk,) + occ[(m, Vk)] for Vk in args.V0kac if (m, Vk) in occ])
        if pts:
            V, l0, e0, l1, e1 = map(np.array, zip(*pts))
            ax.errorbar(V, l0, yerr=e0, fmt="s--", color="black", ms=7, lw=2.2, zorder=5)
            ax.errorbar(V, l1, yerr=e1, fmt="^:", color="black", ms=7, lw=2.2, zorder=5)
        style = [Line2D([0], [0], color="gray", ls="-", marker="o", ms=7, label=r"$\rho_S/\rho$ (Leggett)"),
                 Line2D([0], [0], color="gray", ls="--", marker="s", ms=7, label=r"$\lambda_0/N$"),
                 Line2D([0], [0], color="gray", ls=":", marker="^", ms=7, label=r"$\lambda_1/N$")]
        sizes = [Line2D([0], [0], color=cmap(i / max(1, len(Ns) - 1)), lw=2.5, label=f"$N = {N}$")
                 for i, N in enumerate(Ns)] + [Line2D([0], [0], color="black", lw=2.5, label=r"$N \to \infty$")]
        leg1 = ax.legend(handles=style, loc="upper left", fontsize=13, framealpha=0.9)
        ax.add_artist(leg1)
        ax.legend(handles=sizes, loc="center left", ncol=1, fontsize=11, framealpha=0.9)
        ax.set_xlabel(r"$V_0/\varepsilon$", fontsize=22)
        ax.set_ylabel(r"$\rho_S/\rho,\ \lambda_\ell/N$", fontsize=22)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, ls="--", lw=0.6, alpha=0.6)
        ax.set_title(TITLE[m], fontsize=14)
        fig.tight_layout()
        fname = os.path.join(img_dir, f"leggett_lambda_{m}.png")
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print("plot:", fname)

    # --- plot 2: ρ_S/ρ at N → ∞ for all estimators ---
    colors = {"VMC": "C0", "DMC": "C1", "lin": "C3", "quad": "C2"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in METHODS:
        pts = sorted([(Vk, *extrap[(m, Vk)]) for Vk in args.V0kac if (m, Vk) in extrap])
        V, a, e = map(np.array, zip(*pts))
        ax.errorbar(V, a, yerr=e, fmt="o-", color=colors[m], ms=6, lw=1.6, label=SHORT[m])
    ax.text(0.02, 0.95, r"Leggett bound, $N\to\infty$", transform=ax.transAxes, va="top", fontsize=15)
    ax.set_xlabel(r"$V_0/\varepsilon$", fontsize=22)
    ax.set_ylabel(r"$\rho_S/\rho$", fontsize=22)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, ls="--", lw=0.6, alpha=0.6)
    ax.legend(fontsize=13, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "leggett_all_estimators.png"), dpi=200)
    plt.close(fig)

    # --- plot 3: cell densities at the largest N for a few couplings ---
    show = [v for v in (-20.0, -40.0, -60.0, -100.0, -150.0) if v in args.V0kac]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6), sharey=False)
    for ax, Vk in zip(np.atleast_1d(axes), show):
        for m in METHODS:
            if (m, Vk, Nmax) in densities:
                n = densities[(m, Vk, Nmax)]
                x = (np.arange(len(n)) + 0.5) / len(n) - 0.5
                ax.plot(x, n, "-", color=colors[m], lw=1.3, label=SHORT[m])
        ax.set_title(rf"$V_0/\varepsilon = {Vk:g}$, $N={Nmax}$", fontsize=12)
        ax.set_xlabel(r"$x/\lambda$")
    np.atleast_1d(axes)[0].set_ylabel(r"$n(x)/\bar n$")
    np.atleast_1d(axes)[-1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "leggett_densities.png"), dpi=150)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"\nSaved {out_dir}/leggett*.csv and plots in {img_dir}")


if __name__ == "__main__":
    main()
