"""
Equilibration check of DMC energy traces: E(tau), running average, first vs second half of the
production data, and E/N for larger discards (in units of the run_dmc.jl burn-in, 20 % of the run).

Usage (from the repository root):
    python python/dmc_equilibration.py                       # V0 = -150, N = 35 40, 1200 walkers, 32000 steps
    python python/dmc_equilibration.py --V0kac -40 --N 40
Output: final_imag/equilibration_V0<V>.pdf
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc_gp_delta import blocking_curve, gp_energy, MIN_BLOCKS, BASE  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 20,
    "text.usetex": True,
})


def mean_err(y):
    curve = blocking_curve(y)
    errs = [e for nb, _, e in curve if nb >= MIN_BLOCKS]
    return y.mean(), (max(errs) if errs else curve[-1][2])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, default=-150.0)
    p.add_argument("--N", type=int, nargs="+", default=[35, 40])
    p.add_argument("--walkers", type=int, default=1200)
    p.add_argument("--steps", type=int, default=32000)
    p.add_argument("--dtau", type=float, default=1e-3)
    args = p.parse_args()

    fig, axes = plt.subplots(len(args.N), 1, figsize=(16, 7 * len(args.N)), squeeze=False)
    for ax, N in zip(axes[:, 0], args.N):
        V0 = args.V0kac / N
        f = os.path.join(BASE, "energies", f"V0={V0:.3f}",
                         f"energies_{args.walkers}_{args.dtau:.4f}_{N}_{args.steps}_{V0:.3f}_{float(N):.1f}_IS.npy")
        e = np.load(f) / N
        tau = (np.arange(len(e)) + 1) * args.dtau
        egp = gp_energy(args.V0kac, N)
        n0 = int(0.2 * len(e))                                     # run_dmc.jl discard

        print(f"\nV0 = {args.V0kac:.0f}, N = {N}  ({len(e)} steps, tau_total = {tau[-1]:.1f}),  E_GP/N = {egp:.4f}")
        prod = e[n0:]
        h = len(prod) // 2
        m1, s1 = mean_err(prod[:h]); m2, s2 = mean_err(prod[h:])
        print(f"  first half  (tau {tau[n0]:.1f}-{tau[n0 + h - 1]:.1f}): {m1:.4f} ± {s1:.4f}")
        print(f"  second half (tau {tau[n0 + h]:.1f}-{tau[-1]:.1f}): {m2:.4f} ± {s2:.4f}"
              f"   diff = {m2 - m1:+.4f} ± {np.hypot(s1, s2):.4f} ({abs(m2 - m1) / np.hypot(s1, s2):.1f} sigma)")
        for k in (1, 2, 4):
            nk = k * n0
            m, s = mean_err(e[nk:])
            print(f"  discard {k}x (tau > {tau[nk]:5.1f}, {tau[-1] - tau[nk]:4.1f} kept): E/N = {m:.4f} ± {s:.4f}"
                  f"   E - E_GP = {m - egp:+.4f} ± {s:.4f}")

        # trace: 0.1-tau block means, running average from the start and from the discard
        b = 100
        nb = len(e) // b
        tb = tau[:nb * b].reshape(nb, b).mean(1)
        eb = e[:nb * b].reshape(nb, b).mean(1)
        ax.plot(tb, eb, "-", color="0.6", lw=1, label=r"$E(\tau)/N$, $0.1$ blocks")
        ax.plot(tau, np.cumsum(e) / np.arange(1, len(e) + 1), "C0-", lw=2.5, label="running average from 0")
        ax.plot(tau[n0:], np.cumsum(prod) / np.arange(1, len(prod) + 1), "C3-", lw=2.5,
                label="running average after discard")
        ax.axhline(egp, color="k", ls="--", lw=2, label=r"$E_{\rm GP}(N)/N$")
        for k, ls in ((1, ":"), (2, "-."), (4, "--")):
            ax.axvline(tau[k * n0], color="C2", ls=ls, lw=1.5)
        m, s = mean_err(prod)
        ax.set_ylim(m - 0.4, m + 0.4)
        ax.set_xlabel(r"$\tau\,[\hbar/\varepsilon]$")
        ax.set_ylabel(r"$E/N\,[\varepsilon]$")
        ax.set_title(fr"$N={N},\ V_0={args.V0kac:.0f}\,\varepsilon$ (green: discard $\times1,2,4$)")
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    os.makedirs("final_imag", exist_ok=True)
    out = os.path.join("final_imag", f"equilibration_V0{args.V0kac:.0f}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
