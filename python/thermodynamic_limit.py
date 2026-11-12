"""
Thermodynamic-limit extrapolation of the energy per particle.

With Kac scaling (bare V0 = V0_kac / N) the energy per particle e(N) = E/N has a finite
limit. For each V0_kac we fit

    e(N) = e_inf + a / N            (order 1, default)
    e(N) = e_inf + a / N + b / N^2  (order 2)

by weighted least squares in 1/N and report e_inf = e(1/N -> 0) with its error.

Errors on e(N):
  VMC: blocking analysis of energy_array (E_L / N at each measurement).
  DMC: the blocking error saved by run_dmc.jl (energy file = [E, err]).

Usage (from the repository root):
    python python/thermodynamic_limit.py                     # VMC
    python python/thermodynamic_limit.py --method DMC        # DMC (latest run per point)
    python python/thermodynamic_limit.py --method DMC --dtau 0.002    # DMC at one Δτ
    python python/thermodynamic_limit.py --order 2 --nmin 20
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def blocking_error(x, min_blocks=32):
    """Standard error of the mean of a correlated series (Flyvbjerg-Petersen plateau)."""
    y = np.asarray(x, dtype=float)
    err = y.std(ddof=1) / np.sqrt(len(y))
    while len(y) // 2 >= min_blocks:
        n = len(y) // 2
        y = 0.5 * (y[0:2 * n:2] + y[1:2 * n:2])
        err = max(err, y.std(ddof=1) / np.sqrt(n))
    return err


def load_vmc(base, N, V0, L):
    """(E/N, error) from run_vmc.jl / run_vmc_energy.jl output, or None."""
    tag = f"{N}_{V0:.3f}_{L:.1f}.npy"
    f_E = os.path.join(base, "energy", f"V0={V0:.3f}", f"energy_{tag}")
    f_arr = os.path.join(base, "energy_array", f"V0={V0:.3f}", f"energy_array_{tag}")
    if not os.path.isfile(f_E):
        return None
    e = float(np.load(f_E)) / N
    err = blocking_error(np.load(f_arr)) if os.path.isfile(f_arr) else np.nan
    return e, err


# filename_core in run_dmc.jl: walkers_dtau_N_steps_V0_L[_IS]
DMC_RE = re.compile(r"energy_(\d+)_([\d.]+)_(\d+)_(\d+)_(-?[\d.]+)_([\d.]+)(_IS)?\.npy$")


def load_dmc(base, N, V0, L, dtau=None, walkers=None):
    """(E/N, error) from run_dmc.jl output (most recent matching file), or None."""
    files = []
    for f in glob.glob(os.path.join(base, "energy", f"V0={V0:.3f}", "energy_*.npy")):
        m = DMC_RE.search(os.path.basename(f))
        if not m:
            continue
        w, dt, n, _, v0, l, _ = m.groups()
        if int(n) != N or abs(float(v0) - V0) > 5e-4 or abs(float(l) - L) > 0.05:
            continue
        if dtau is not None and abs(float(dt) - dtau) > 1e-9:
            continue
        if walkers is not None and int(w) != walkers:
            continue
        files.append(f)
    if not files:
        return None
    E, err = np.atleast_1d(np.load(max(files, key=os.path.getmtime)))[:2]
    return E / N, err / N


def fit_inverse_N(N, e, err, order):
    """Weighted polynomial fit in x = 1/N. Returns coefficients, their errors, chi2/dof."""
    x = 1.0 / np.asarray(N, dtype=float)
    A = np.vander(x, order + 1, increasing=True)            # [1, x, x^2, ...]
    w = 1.0 / np.asarray(err) ** 2
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (w * e))
    dof = len(e) - (order + 1)
    chi2 = np.sum(w * (e - A @ coef) ** 2)
    chi2_dof = chi2 / dof if dof > 0 else np.nan
    # inflate errors when the fit is poor (chi2/dof > 1)
    scale = np.sqrt(max(chi2_dof, 1.0)) if dof > 0 else 1.0
    return coef, np.sqrt(np.diag(cov)) * scale, chi2_dof


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["VMC", "DMC"], default="VMC")
    p.add_argument("--density", type=float, default=1.0)
    p.add_argument("--N", type=int, nargs="+", default=[15, 20, 25, 30, 35, 40])
    p.add_argument("--nmin", type=int, default=0, help="only use N >= nmin in the fit")
    p.add_argument("--V0kac", type=float, nargs="+", default=None,
                   help="default: 0 to -250 in steps of 10 (VMC); the run_dmc.jl list (DMC)")
    p.add_argument("--order", type=int, choices=[1, 2], default=1)
    p.add_argument("--dtau", type=float, default=None, help="DMC: only runs with this Δτ")
    p.add_argument("--walkers", type=int, default=None, help="DMC: only runs with this many walkers")
    p.add_argument("--show-fits", type=float, nargs="*", default=[0.0, -50.0, -100.0, -150.0, -200.0, -250.0],
                   help="V0_kac values whose E/N vs 1/N fit is plotted")
    args = p.parse_args()
    if args.V0kac is None:
        args.V0kac = (list(np.arange(0.0, -251.0, -10.0)) if args.method == "VMC" else
                      [-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0, -200.0])

    folder = f"constant_density_{args.density:.1f}"
    base = os.path.join(f"numpy_arrays_{args.method}", folder)
    out_dir = os.path.join(f"numpy_arrays_{args.method}", "thermodynamic_limit", folder)
    img_dir = os.path.join(f"imag_{args.method}", "thermodynamic_limit", folder)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    Ns = [N for N in args.N if N >= args.nmin]
    rows = []
    fit_data = {}
    print(f"{args.method}: e(N) = e_inf + a/N" + (" + b/N^2" if args.order == 2 else "") + f", N = {Ns}")
    print(f"{'V0_kac':>8} {'n_pts':>5} {'e_inf':>14} {'+/- err':>10} {'a':>12} {'chi2/dof':>9}")
    for V0kac in args.V0kac:
        pts = []
        for N in Ns:
            L = N / args.density
            V0 = V0kac / N
            r = (load_vmc(base, N, V0, L) if args.method == "VMC"
                 else load_dmc(base, N, V0, L, args.dtau, args.walkers))
            if r is not None and np.isfinite(r[0]):
                pts.append((N, *r))
        if len(pts) < args.order + 2:
            print(f"{V0kac:8.1f} {len(pts):5d}   not enough points")
            continue
        N_arr, e_arr, err_arr = map(np.array, zip(*pts))
        # e = 0 exactly at V0_kac = 0 (zero error): give it a tiny error so the fit is defined
        err_arr = np.where(np.isfinite(err_arr) & (err_arr > 0), err_arr, 1e-12)
        coef, coef_err, chi2_dof = fit_inverse_N(N_arr, e_arr, err_arr, args.order)
        rows.append([V0kac, coef[0], coef_err[0], coef[1], coef_err[1], chi2_dof, len(pts)])
        fit_data[V0kac] = (N_arr, e_arr, err_arr, coef)
        print(f"{V0kac:8.1f} {len(pts):5d} {coef[0]:14.6f} {coef_err[0]:10.2e} {coef[1]:12.4f} {chi2_dof:9.2f}")

    if not rows:
        print("No data found under", base)
        return
    rows = np.array(rows)
    tag = f"order{args.order}_nmin{args.nmin}"
    np.savetxt(os.path.join(out_dir, f"e_inf_{tag}.csv"), rows, delimiter=",",
               header="V0_kac,e_inf,e_inf_err,a,a_err,chi2_dof,n_points", comments="")
    print("\nSaved", os.path.join(out_dir, f"e_inf_{tag}.csv"))

    # --- E/N vs 1/N with fits for a few couplings ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for V0kac in args.show_fits:
        if V0kac not in fit_data:
            continue
        N_arr, e_arr, err_arr, coef = fit_data[V0kac]
        line = ax.errorbar(1 / N_arr, e_arr, yerr=err_arr, fmt="o", ms=4, label=f"$V_0^{{Kac}}$ = {V0kac:g}")
        xs = np.linspace(0, 1.05 / N_arr.min(), 100)
        ax.plot(xs, np.polyval(coef[::-1], xs), "--", color=line[0].get_color(), lw=1)
    ax.set_xlabel("1/N")
    ax.set_ylabel("E/N")
    ax.set_xlim(left=0)
    ax.set_title(f"{args.method}: extrapolation to N → ∞")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, f"E_per_N_vs_invN_{tag}.png"), dpi=150)

    # --- e_inf vs V0_kac ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="o-", ms=4)
    ax.set_xlabel("$V_0^{Kac}$")
    ax.set_ylabel("$e_\\infty = \\lim_{N\\to\\infty} E/N$")
    ax.set_title(f"{args.method}: thermodynamic limit")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, f"e_inf_vs_V0kac_{tag}.png"), dpi=150)
    print("Plots saved to", img_dir)


if __name__ == "__main__":
    main()
