"""
Bragg peak of the static structure factor, S(k_c) at k_c = 2 pi / lambda: QMC vs GP.

Normalisation. Here S(k) = <|rho_k|^2> / N with rho_k = sum_a exp(i k x_a) (S -> 1 at large k), and
the plotted quantity is S(k_c)/N = <|rho_k|^2>/N^2, which tends to Theta^2 deep in the phase.
The saved arrays do NOT share one normalisation:
  * VMC (julia/src/sampler.jl, numpy_arrays_VMC/.../SSF and numpy_arrays_DMC/.../SSF_vmc_*):
    <|rho_k|^2>              (large-k tail -> N)
  * DMC mixed (julia/dmc/dmc_core.jl, numpy_arrays_DMC/.../SSF_<walkers>_*):
    <|rho_k|^2> / N          (large-k tail -> 1)
so the saved SSF_extrap_* = 2 SSF_DMC - SSF_vmc is wrong. Here both are converted to S(k) first and
the extrapolated estimator is recomputed as 2 S_DMC - S_VMC (VMC reference of the same DMC run).

GP (product state phi^N, and the cat up to terms ~ s^N):  <|rho_k|^2> = N + N(N-1) Theta^2, i.e.
S_GP(k_c) = 1 + (N-1) Theta^2, with Theta = sum_n c_n c_{n+1} from julia/gp/gp_scf.jl (gp_dense.csv).

DMC run per point: longest imaginary time, then most walkers. V0 = -200 (dropped) is not shown for DMC.
No statistical errors are saved for S(k), so the QMC points have no error bars.

Usage (from the repository root):  python python/plots/plot_skc_gp.py
Output: final_imag/Skc_vs_V0_GP_N40.pdf, numpy_arrays_GP/skc_gp_qmc_N40.csv
"""

import glob
import os
import re

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
    "legend.fontsize": 24,
    "text.usetex": True,
})

N = 40
IC = N - 1                                              # k = 2 pi n / L, n = 1 ... 5L; k_c is n = L
VC = 2 * np.pi ** 2 * N / (N - 1)
BV = os.path.join("numpy_arrays_VMC", "constant_density_1.0", "SSF")
BD = os.path.join("numpy_arrays_DMC", "constant_density_1.0", "SSF")
DMC_VMIN = -150.0                                       # -200 dropped


def vmc_skc(Vk):
    """S(k_c) from the standalone VMC runs (array = <|rho_k|^2>)."""
    v = Vk / N + 0.0
    f = os.path.join(BV, f"V0={v:.3f}", f"SSF_{N}_{v:.3f}_{float(N):.1f}.npy")
    return np.load(f).real[IC] / N if os.path.isfile(f) else np.nan


def dmc_skc(Vk):
    """(S_DMC(k_c), S_VMC-ref(k_c), run label) for the longest / largest DMC run at (N, Vk)."""
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
        return np.nan, np.nan, ""
    s_dmc = np.load(os.path.join(d, best[1])).real[IC]           # already <|rho|^2>/N
    s_vmc = np.load(fv).real[IC] / N                              # <|rho|^2> -> <|rho|^2>/N
    return s_dmc, s_vmc, f"{best[2]}w, tau={best[3]:g}"


# GP
dense = np.loadtxt(os.path.join("numpy_arrays_GP", "gp_dense.csv"), delimiter=",", skiprows=1)
Vg, th = dense[:, 0], dense[:, 2]                                  # Theta1 at N = 40
s_gp = (1 + (N - 1) * th ** 2) / N


def gp_at(Vk):
    return float(np.interp(Vk, Vg[::-1], s_gp[::-1]))


# QMC
V_vmc = np.arange(0.0, -211.0, -10.0)          # GP grid (gp_dense.csv) ends at -210
S_vmc = np.array([vmc_skc(v) for v in V_vmc])
V_dmc = [-10, -15, -20, -25, -30, -35, -40, -50, -60, -80, -100, -150]
rows = []
for v in V_dmc:
    sd, sv, run = dmc_skc(v)
    rows.append((v, gp_at(v), vmc_skc(v) / N, sv / N, sd / N, (2 * sd - sv) / N, run))

# figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(Vg, s_gp, "k-", lw=2.5, label=r"GP: $[1+(N-1)\Theta^2]/N$")
m = np.isfinite(S_vmc) & (V_vmc >= -150)
ax.plot(V_vmc[m], S_vmc[m] / N, "o", color="0.65", ms=7, mfc="none", mew=1.5, label="VMC")   # pale: reference only
R = np.array([r[:6] for r in rows], float)
ax.plot(R[:, 0], R[:, 4], "s", color="C2", ms=11, label="DMC mixed")
ax.plot(R[:, 0], R[:, 5], "^", color="C3", ms=12, label=r"$2\,$DMC$-$VMC")
ax.axvline(-VC, color="0.5", ls=":", lw=2)
ax.text(-VC + 1.5, 0.5, r"$|V_0|_c$", ha="left", fontsize=28, color="0.4")
ax.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax.set_ylabel(r"$S(k_c)/N$")
ax.set_title(fr"$N={N}$")
ax.set_xlim(-155, 0)
ax.grid(True, ls="--", alpha=0.6)
ax.legend(loc="lower left")
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", f"Skc_vs_V0_GP_N{N}.pdf"), dpi=600, bbox_inches="tight")

# table
with open(os.path.join("numpy_arrays_GP", f"skc_gp_qmc_N{N}.csv"), "w") as f:
    f.write("V0,S_GP_over_N,S_VMC_over_N,S_VMCref_over_N,S_DMC_over_N,S_lin_over_N,DMC_run\n")
    for r in rows:
        f.write(",".join(f"{x:.6g}" for x in r[:6]) + f",{r[6]}\n")
print(f"N = {N}, S(k_c)/N   (threshold |V0|_c = {VC:.3f})")
print("    V0   GP      VMC     VMC(ref) DMC mix  2DMC-VMC   VMC-GP   DMC-GP   lin-GP   [DMC run]")
for v, g, vm, vr, sd, sl, run in rows:
    print(f"{v:6.0f}  {g:.4f}  {vm:.4f}  {vr:.4f}   {sd:.4f}   {sl:.4f}   {vr - g:+.4f}  {sd - g:+.4f}  {sl - g:+.4f}   [{run}]")
print(f"Saved final_imag/Skc_vs_V0_GP_N{N}.pdf and numpy_arrays_GP/skc_gp_qmc_N{N}.csv")
