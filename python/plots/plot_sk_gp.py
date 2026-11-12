"""
Static structure factor S(k) vs k at fixed V0 and N: QMC (VMC, DMC mixed, 2 DMC - VMC) vs GP.

S(k) = <|rho_k|^2>/N, rho_k = sum_a exp(i k x_a), on the QMC grid k = 2 pi n / L, n = 1 ... 5L.
VMC arrays store <|rho_k|^2>, DMC arrays <|rho_k|^2>/N; both are converted before 2 DMC - VMC is formed
(the saved SSF_extrap_* files mix the two and are not used).

GP (product state phi^N; the Z2 cat gives the same values up to terms ~ s^N): the density of phi is
periodic with lambda, so <exp(i k x)>_phi = F_j = sum_n c_n c_{n+j} at k = j k_c (k_c = 2 pi / lambda)
and 0 otherwise. Hence S_GP(k) = 1 + (N-1) F_j^2 at the Bragg peaks k = j k_c and S_GP(k) = 1 elsewhere
(GP has no correlations between particles, so no structure between the peaks).

DMC run: longest imaginary time, then most walkers.

Usage (from the repository root, after julia julia/gp/gp_scf.jl):
    python python/plots/plot_sk_gp.py                          # N = 40, V0 = -40 -150
    python python/plots/plot_sk_gp.py --V0kac -60 --N 40
Output: final_imag/Sk_vs_k_GP_N<N>_V0<V>.pdf
"""

import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 24,
    "text.usetex": True,
})

BV = os.path.join("numpy_arrays_VMC", "constant_density_1.0", "SSF")
BD = os.path.join("numpy_arrays_DMC", "constant_density_1.0", "SSF")


def dmc_arrays(Vk, N):
    """(S_DMC(k), S_VMC-ref(k), label) in the S = <|rho_k|^2>/N normalisation, or (None, None, '')."""
    v = Vk / N + 0.0
    d = os.path.join(BD, f"V0={v:.3f}")
    pat = re.compile(rf"^SSF_(\d+)_([\d.]+)_{N}_(\d+)_{v:.3f}_{float(N):.1f}_IS\.npy$")
    best = None
    for f in (os.listdir(d) if os.path.isdir(d) else []):
        m = pat.match(f)
        if m:
            w, dt, st = int(m[1]), float(m[2]), int(m[3])
            key = (round(st * dt, 6), w)
            if best is None or key > best[0]:
                best = (key, f, w, st * dt)
    fv = os.path.join(d, f"SSF_vmc_{N}_{v:.3f}_{float(N):.1f}.npy")
    if best is None or not os.path.isfile(fv):
        return None, None, ""
    sd = np.load(os.path.join(d, best[1])).real
    sv = np.load(fv).real / N
    return sd, sv, fr"{best[2]} walkers, $\tau={best[3]:g}$"


def vmc_array(Vk, N):
    v = Vk / N + 0.0
    f = os.path.join(BV, f"V0={v:.3f}", f"SSF_{N}_{v:.3f}_{float(N):.1f}.npy")
    return np.load(f).real / N if os.path.isfile(f) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--V0kac", type=float, nargs="+", default=[-40.0, -150.0])
    args = p.parse_args()
    N, L = args.N, float(args.N)
    k_over_kc = np.arange(1, int(5 * L) + 1) / L              # k / k_c on the QMC grid

    cn = np.loadtxt(os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv"), delimiter=",", skiprows=1)
    os.makedirs("final_imag", exist_ok=True)

    for Vk in args.V0kac:
        row = cn[np.isclose(cn[:, 0], Vk)]
        if not len(row):
            print(f"V0 = {Vk:.0f}: not in numpy_arrays_GP/gp_cn_N{N}.csv, skipped")
            continue
        c = row[0, 1:]
        jmax = int(k_over_kc[-1])
        F = np.array([np.sum(c[:len(c) - j] * c[j:]) for j in range(1, jmax + 1)])
        s_gp = np.ones_like(k_over_kc)
        for j in range(1, jmax + 1):
            s_gp[int(round(j * L)) - 1] = 1 + (N - 1) * F[j - 1] ** 2

        sd, sv_ref, run = dmc_arrays(Vk, N)
        sv = vmc_array(Vk, N)
        if sv is None:
            sv = sv_ref

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(k_over_kc, s_gp, "k-", lw=2.5, zorder=5,               # GP in black, on top of QMC
                label=r"GP: $1+(N-1)F_j^2$ at $k=jk_c$, else 1")
        if sv is not None:
            ax.plot(k_over_kc, sv, "-", color="C1", lw=2, label="VMC")
        if sd is not None:
            ax.plot(k_over_kc, sd, "-", color="C2", lw=2, label="DMC mixed")
            ax.plot(k_over_kc, 2 * sd - sv_ref, "--", color="C3", lw=2, label=r"$2\,$DMC$-$VMC")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k/k_c$")
        ax.set_ylabel(r"$S(k)$")
        ax.set_title(fr"$N={N},\ V_0={Vk:.0f}\,\varepsilon$" + (f"  (DMC: {run})" if run else ""), fontsize=30)
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=22)
        fig.tight_layout()
        out = os.path.join("final_imag", f"Sk_vs_k_GP_N{N}_V0{Vk:.0f}.pdf")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        plt.close(fig)

        print(f"\nN = {N}, V0 = {Vk:.0f}  (DMC: {run})")
        print("  j   S_GP     VMC      DMC mix   2DMC-VMC")
        for j in range(1, jmax + 1):
            i = int(round(j * L)) - 1
            vals = [s_gp[i], sv[i] if sv is not None else np.nan,
                    sd[i] if sd is not None else np.nan, (2 * sd - sv_ref)[i] if sd is not None else np.nan]
            print(f"  {j}  " + "  ".join(f"{x:8.3f}" for x in vals))
        if sd is not None:
            off = np.ones_like(k_over_kc, bool)
            off[[int(round(j * L)) - 1 for j in range(1, jmax + 1)]] = False
            print(f"  between peaks: GP 1, DMC mixed {sd[off].min():.3f}-{sd[off].max():.3f}, "
                  f"2DMC-VMC {(2 * sd - sv_ref)[off].min():.3f}-{(2 * sd - sv_ref)[off].max():.3f}")
        print("Saved", out)


if __name__ == "__main__":
    main()
