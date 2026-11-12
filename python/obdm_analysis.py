"""
One-body density matrix analysis of the VMC / DMC production runs (julia/dmc/run_dmc.jl).

Estimators of ρ₁(x, x') for every (N, V0_kac):
  * VMC                           ⟨ρ₁⟩_VMC
  * DMC (mixed)                   ⟨ρ₁⟩_DMC
  * linear extrapolation          2 ⟨O⟩_DMC − ⟨O⟩_VMC        (applied to the ρ₁ matrix)
  * quadratic extrapolation       ⟨O⟩_DMC² / ⟨O⟩_VMC          (applied to λ₀, λ₁ and g₁(r))

Each matrix is symmetrized and averaged over translations by one cavity period λ (an exact
symmetry of H; one cell = nb/L bins), which reduces the noise by ~√N without losing
resolution inside the cell. The linear estimator is linear in the data, so it is applied to
the matrix itself. The quadratic one is not: applied element by element to the (noisy)
matrix it acquires a bias ~ var(ρ₁)/ρ₁, which is large where ρ₁ is small or noisy (e.g. it
gives λ₀/N = 1.01 at V0_kac = −10 and 0.67 instead of ~0.55 at −100). It is therefore applied
to the well-determined derived quantities: λ₀^quad = (λ₀^DMC)² / λ₀^VMC and
g₁^quad(r) = g₁^DMC(r)² / g₁^VMC(r). For λ₁ the formula is ill-conditioned where
λ₁^VMC ≪ λ₁^DMC (V0_kac ≈ −30 … −40), so λ₁^quad = N − λ₀^quad is used instead: VMC, DMC
and the linear estimator all satisfy λ₀ + λ₁ = N to better than 1.5 % (no depletion
beyond the two density-wave modes).

From each matrix:
  * n₀ ≥ n₁ = two largest eigenvalues of ρ₁ / N (natural-orbital occupations λ₀/N, λ₁/N);
  * g₁(r) = (1/N) ∫ dx ρ₁(x, x + r)   (g₁(0) = 1);
  * noise level = most negative eigenvalue / N (ρ₁ is positive semi-definite).
n₀ and n₁ are extrapolated to N → ∞ with a linear fit in 1/N.

Usage (from the repository root):
    python python/obdm_analysis.py              # all estimators, plots in imag_DMC/obdm_analysis/
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

METHODS = ["VMC", "DMC", "lin", "quad"]
LABELS = {"VMC": "VMC", "DMC": "DMC (mixed)",
          "lin": r"linear extrapolation $2\langle\rho_1\rangle_{\rm DMC}-\langle\rho_1\rangle_{\rm VMC}$",
          "quad": r"quadratic extrapolation $\langle\rho_1\rangle_{\rm DMC}^2/\langle\rho_1\rangle_{\rm VMC}$"}
SHORT = {"VMC": "VMC", "DMC": "DMC mixed", "lin": "linear extr.", "quad": "quadratic extr."}


# ----------------------------------------------------------------------------- loading
def load_raw(base, N, V0kac, which, density=1.0):
    """Raw ρ₁ matrix: which = "VMC" or "DMC" (mixed). Returns (matrix or None, L)."""
    V0 = V0kac / N
    L = N / density
    d = os.path.join(base, "obdm", f"V0={V0:.3f}")
    if which == "VMC":
        f = os.path.join(d, f"obdm_vmc_{N}_{V0:.3f}_{L:.1f}.npy")
        files = [f] if os.path.isfile(f) else []
    else:
        files = [f for f in glob.glob(os.path.join(d, f"obdm_*_{N}_*_{V0:.3f}_{L:.1f}_IS.npy"))
                 if "extrap" not in os.path.basename(f)]
    if not files:
        return None, L
    return np.load(max(files, key=os.path.getmtime)), L


def symmetrize(rho, L):
    """ρ₁ → ½(ρ₁ + ρ₁ᵀ), averaged over translations by λ = 1."""
    nb = rho.shape[0]
    rho = 0.5 * (rho + rho.T)
    n_cells = int(round(L))
    shift = nb // n_cells
    if shift * n_cells == nb:
        rho = np.mean([np.roll(np.roll(rho, k * shift, 0), k * shift, 1) for k in range(n_cells)], axis=0)
    return rho


def quadratic(o_dmc, o_vmc):
    """⟨O⟩_DMC² / ⟨O⟩_VMC (elementwise for arrays; 2·DMC − VMC where |⟨O⟩_VMC| ≈ 0)."""
    o_dmc, o_vmc = np.asarray(o_dmc, float), np.asarray(o_vmc, float)
    small = np.abs(o_vmc) < 1e-12
    return np.where(small, 2 * o_dmc - o_vmc, o_dmc ** 2 / np.where(small, 1.0, o_vmc))


def analyse(rho, N, L):
    """n0, n1, noise, r, g1 of an (already symmetrized) ρ₁ matrix."""
    nb = rho.shape[0]
    dx = L / nb
    w = np.linalg.eigvalsh(rho * dx)[::-1]
    idx = np.arange(nb)
    g1 = np.array([np.sum(rho[idx, (idx + j) % nb]) * dx for j in range(nb // 2 + 1)]) / N
    return w[0] / N, w[1] / N, max(0.0, -w[-1] / N), np.arange(nb // 2 + 1) * dx, g1


def fit_inverse_N(N, y):
    """Unweighted fit y = y_inf + a/N; returns (y_inf, error from residuals, slope)."""
    x = 1.0 / np.asarray(N, float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = len(y) - 2
    s2 = (np.sum((y - A @ coef) ** 2) / dof) if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(A.T @ A)
    return coef[0], np.sqrt(cov[0, 0]), coef[1]


# ----------------------------------------------------------------------------- plots
def plot_paper_style(rows, extrap, method, Ns, img_dir):
    """λ₀/N (solid) and λ₁/N (dashed) vs V0: one thin line per N, N → ∞ (linear in 1/N) in black."""
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 16})
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, N in enumerate(sorted(Ns)):
        pts = sorted([(r[1], r[3], r[4]) for r in rows if r[0] == method and r[2] == N])
        if not pts:
            continue
        V, n0, n1 = map(np.array, zip(*pts))
        c = plt.cm.viridis(i / max(1, len(Ns) - 1))
        ax.plot(V, n0, "-", color=c, lw=1, alpha=0.55, label=f"$N = {N}$")
        ax.plot(V, n1, "--", color=c, lw=1, alpha=0.55)
    pts = sorted([(Vk,) + v for (m, Vk), v in extrap.items() if m == method])
    V, a0, e0, a1, e1 = map(np.array, zip(*pts))
    ax.errorbar(V, a0, yerr=e0, fmt="o-", color="black", ms=7, lw=2, capsize=0, label=r"$N \to \infty$", zorder=5)
    ax.errorbar(V, a1, yerr=e1, fmt="s--", color="black", ms=7, lw=2, capsize=0, zorder=5)
    ax.text(0.02, 0.95, r"solid: $\lambda_0/N$, dashed: $\lambda_1/N$", transform=ax.transAxes, va="top", fontsize=15)
    ax.set_xlabel(r"$V_0/\varepsilon$", fontsize=22)
    ax.set_ylabel(r"$\lambda_\ell/N$", fontsize=22)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, ls="--", lw=0.6, alpha=0.6)
    ax.legend(ncol=2, fontsize=13, loc="lower left", framealpha=0.9)
    ax.set_title(LABELS[method], fontsize=14)
    fig.tight_layout()
    fname = os.path.join(img_dir, f"obdm_lambda_{method}.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    return fname


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--density", type=float, default=1.0)
    p.add_argument("--N", type=int, nargs="+", default=[15, 20, 25, 30, 35, 40])
    p.add_argument("--V0kac", type=float, nargs="+",
                   default=[-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0, -200.0])
    p.add_argument("--g1-V0kac", type=float, nargs="+", default=[-10.0, -20.0, -30.0, -40.0, -60.0, -100.0],
                   help="couplings whose g1(r) is plotted (largest N)")
    args = p.parse_args()

    folder = f"constant_density_{args.density:.1f}"
    base = os.path.join("numpy_arrays_DMC", folder)
    out_dir = os.path.join("numpy_arrays_DMC", "obdm_analysis", folder)
    img_dir = os.path.join("imag_DMC", "obdm_analysis", folder)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    rows = []                       # method, V0kac, N, n0, n1, noise
    g1_store = {}
    for Vk in args.V0kac:
        for N in args.N:
            rv, L = load_raw(base, N, Vk, "VMC", args.density)
            rd, _ = load_raw(base, N, Vk, "DMC", args.density)
            if rv is None or rd is None:
                continue
            V, D = symmetrize(rv, L), symmetrize(rd, L)
            res = {m: analyse(M, N, L) for m, M in (("VMC", V), ("DMC", D), ("lin", 2 * D - V))}
            (n0v, n1v, _, r, g1v), (n0d, n1d, _, _, g1d) = res["VMC"], res["DMC"]
            # λ₁: D²/V is ill-conditioned where λ₁_VMC ≪ λ₁_DMC (−30 … −40: 0.01–0.05 vs 0.08–0.19);
            # VMC, DMC and the linear estimator all satisfy λ₀ + λ₁ = N to < 1.5 %, so λ₁ = N − λ₀.
            n0q = float(quadratic(n0d, n0v))
            res["quad"] = (n0q, 1.0 - n0q, np.nan, r, quadratic(g1d, g1v))
            for m in METHODS:
                n0, n1, noise, r, g1 = res[m]
                rows.append((m, Vk, N, n0, n1, noise))
                g1_store[(m, Vk, N)] = (r, g1)
    if not rows:
        print("No OBDM data found under", base)
        return

    # --- table at the largest N and N → ∞ (linear in 1/N) ---
    Nmax = max(args.N)
    extrap = {}                     # (method, Vk) -> (n0_inf, err0, n1_inf, err1)
    print(f"{'V0_kac':>7} {'estimator':>16} | {'n0(N=%d)' % Nmax:>9} {'n1':>7} {'noise':>6} | "
          f"{'n0(N→∞)':>16} {'n1(N→∞)':>16}")
    for Vk in args.V0kac:
        for m in METHODS:
            sel = sorted([r for r in rows if r[0] == m and r[1] == Vk], key=lambda r: r[2])
            if not sel:
                continue
            Ns = np.array([r[2] for r in sel]); n0s = np.array([r[3] for r in sel]); n1s = np.array([r[4] for r in sel])
            if len(sel) >= 3:
                a0, e0, _ = fit_inverse_N(Ns, n0s)
                a1, e1, _ = fit_inverse_N(Ns, n1s)
            else:
                a0 = e0 = a1 = e1 = np.nan
            extrap[(m, Vk)] = (a0, e0, a1, e1)
            last = [r for r in sel if r[2] == Nmax]
            n0N, n1N, nzN = (last[0][3], last[0][4], last[0][5]) if last else (np.nan,) * 3
            print(f"{Vk:7.0f} {SHORT[m]:>16} | {n0N:9.4f} {n1N:7.4f} {nzN:6.3f} | "
                  f"{a0:7.4f} ± {e0:.4f} {a1:7.4f} ± {e1:.4f}")

    with open(os.path.join(out_dir, "obdm_occupations.csv"), "w") as f:
        f.write("estimator,V0_kac,N,n0,n1,noise\n")
        for m, Vk, N, n0, n1, nz in rows:
            f.write(f"{m},{Vk},{N},{n0:.6f},{n1:.6f},{nz:.6f}\n")
    with open(os.path.join(out_dir, "obdm_occupations_Ninf.csv"), "w") as f:
        f.write("estimator,V0_kac,n0_inf,n0_inf_err,n1_inf,n1_inf_err\n")
        for (m, Vk), (a0, e0, a1, e1) in extrap.items():
            f.write(f"{m},{Vk},{a0:.6f},{e0:.6f},{a1:.6f},{e1:.6f}\n")

    # --- plot 1 (paper style), one figure per estimator ---
    for m in METHODS:
        print("plot:", plot_paper_style(rows, extrap, m, args.N, img_dir))

    # --- plot 2: all estimators at N → ∞ in one figure ---
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 16})
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"VMC": "C0", "DMC": "C1", "lin": "C3", "quad": "C2"}
    for m in METHODS:
        pts = sorted([(Vk,) + v for (mm, Vk), v in extrap.items() if mm == m])
        V, a0, e0, a1, e1 = map(np.array, zip(*pts))
        ax.errorbar(V, a0, yerr=e0, fmt="o-", color=colors[m], ms=6, lw=1.5, label=SHORT[m])
        ax.errorbar(V, a1, yerr=e1, fmt="s--", color=colors[m], ms=5, lw=1.5)
    ax.text(0.02, 0.95, r"$N\to\infty$;  solid: $\lambda_0/N$, dashed: $\lambda_1/N$", transform=ax.transAxes,
            va="top", fontsize=15)
    ax.set_xlabel(r"$V_0/\varepsilon$", fontsize=22); ax.set_ylabel(r"$\lambda_\ell/N$", fontsize=22)
    ax.set_ylim(-0.03, 1.05); ax.grid(True, ls="--", lw=0.6, alpha=0.6)
    ax.legend(fontsize=13, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "obdm_lambda_all_estimators.png"), dpi=200)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    # --- plot 3: n0 vs 1/N for each estimator ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, m in zip(axes, METHODS):
        for i, Vk in enumerate(args.V0kac):
            sel = sorted([r for r in rows if r[0] == m and r[1] == Vk], key=lambda r: r[2])
            if not sel:
                continue
            Nsel = np.array([r[2] for r in sel]); y = np.array([r[3] for r in sel]); x = 1 / Nsel
            line, = ax.plot(x, y, "o", ms=4, color=plt.cm.viridis(i / max(1, len(args.V0kac) - 1)), label=f"{Vk:g}")
            if len(sel) >= 3:
                a, _, b = fit_inverse_N(Nsel, y)
                xs = np.linspace(0, x.max(), 20)
                ax.plot(xs, a + b * xs, "--", lw=0.8, color=line.get_color())
        ax.set_title(SHORT[m]); ax.set_xlabel("1/N"); ax.set_xlim(left=0)
    axes[0].set_ylabel("$n_0$")
    axes[-1].legend(title="$V_0^{Kac}$", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "obdm_n0_vs_invN.png"), dpi=150)
    plt.close(fig)

    # --- plot 4: g1(r) at the largest N ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, m in zip(axes, METHODS):
        for i, Vk in enumerate(args.g1_V0kac):
            if (m, Vk, Nmax) in g1_store:
                r, g1 = g1_store[(m, Vk, Nmax)]
                ax.plot(r, g1, lw=1, color=plt.cm.plasma(i / max(1, len(args.g1_V0kac) - 1)), label=f"{Vk:g}")
        ax.set_title(f"{SHORT[m]}, N = {Nmax}"); ax.set_xlabel("r / λ")
    axes[0].set_ylabel("$g_1(r)$")
    axes[-1].legend(title="$V_0^{Kac}$", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "obdm_g1.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved {out_dir}/obdm_occupations*.csv and plots in {img_dir}")


if __name__ == "__main__":
    main()
