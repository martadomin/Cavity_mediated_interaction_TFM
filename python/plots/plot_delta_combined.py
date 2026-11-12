"""
Delta(N) = E_DMC/N - E_GP(N)/N vs 1/N for the DMC couplings in one panel (same analysis as
python/dmc_gp_delta.py: blocking errors of the per-step energy series after 20 % burn-in, GP at finite N).

Run per (N, V0): longest imaginary time, then most walkers, among the runs with tau >= 8.
  filled markers: converged runs (tau = 32): V0 = -25, -30, -40, -60 (N = 15 ... 50) and -150
  open markers:   older runs (tau = 8, 600 or 1200 walkers): the other couplings
Left out:
  * V0 = -150, N = 35 (1200 walkers, tau = 32): population-control bias (Delta > 0); N = 40 uses the
    2400-walker run instead of the biased 1200-walker one;
  * V0 = -10, N = 40: only a 200-walker, tau = 3 run exists;
  * V0 = -200 (unless --include-200): its runs (1200 walkers, tau = 8) violate the variational bound,
    Delta up to +0.43 (8 sigma) at N = 35, 40 (population-control bias, unconverged); with
    --include-200 it is drawn in a separate figure.
Below threshold (|V0| < 2 pi^2 N/(N-1), i.e. -10, -15, -20 at all N here) the GP state is uniform and
E_GP = 0, so Delta = E_DMC/N.
Points are joined by lines (no fits: the a + b/N fits are in python/dmc_gp_delta.py).

Usage (from the repository root):
    python python/plots/plot_delta_combined.py                  # 12 couplings (-10 ... -150)
    python python/plots/plot_delta_combined.py --include-200    # all 13 -> gp_delta_vs_invN_all_with200.pdf
    python python/plots/plot_delta_combined.py --vs-N           # Delta vs N instead of 1/N -> gp_delta_vs_N_all.pdf
Output: final_imag/gp_delta_vs_{invN,N}_all[_with200].pdf, numpy_arrays_DMC/gp_delta/constant_density_1.0/delta_combined.csv
"""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import analyse, BASE, OUT  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 22,
    "text.usetex": True,
})

WITH200 = "--include-200" in sys.argv
VS_N = "--vs-N" in sys.argv
V0S = [-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0] + ([-200.0] if WITH200 else [])
NS = [15, 20, 25, 30, 35, 40, 50]
TAU_MIN = 8.0
EXCLUDE = {(-150.0, 35)}


def best_run(V, N):
    """(walkers, steps, dtau, tau) of the longest / largest run with tau >= TAU_MIN, or None."""
    v = V / N
    d = os.path.join(BASE, "energies", f"V0={v:.3f}")
    pat = re.compile(rf"^energies_(\d+)_([\d.]+)_{N}_(\d+)_{v:.3f}_{float(N):.1f}_IS\.npy$")
    best = None
    for f in (os.listdir(d) if os.path.isdir(d) else []):
        m = pat.match(f)
        if m:
            w, dt, st = int(m[1]), float(m[2]), int(m[3])
            key = (round(st * dt, 6), w)
            if key[0] >= TAU_MIN - 1e-9 and (best is None or key > best[0]):
                best = (key, w, st, dt)
    return None if best is None else (best[1], best[2], best[3], best[0][0])


out_rows = []
fig, ax = plt.subplots(figsize=(15, 11))
cmap = plt.get_cmap("plasma")
print("   V0    N  walkers  tau     Delta")
for i, V in enumerate(V0S):
    col = cmap(0.88 * i / (len(V0S) - 1))                      # stop before the pale yellow end
    pts = []
    for N in NS:
        if (V, N) in EXCLUDE:
            continue
        r = best_run(V, N)
        if r is None:
            continue
        w, st, dt, tau = r
        rows, _ = analyse(V, [N], w, st, dt)
        if not rows:
            continue
        _, E, err, Eg, d, de, _ = rows[0]
        pts.append((N if VS_N else 1 / N, d, de, tau >= 32 - 1e-9))
        out_rows.append((V, N, E, err, Eg, d, de, w, tau))
        print(f"  {V:5.0f}  {N:3d}  {w:5d}  {tau:4g}   {d:+.4f} ± {de:.4f}")
    if not pts:
        continue
    x, d, s, conv = map(np.array, zip(*pts))
    o = np.argsort(x)
    ax.plot(x[o], d[o], "-", color=col, lw=1.5, alpha=0.7)
    for c_ in (True, False):
        m = conv == c_
        if m.any():
            ax.errorbar(x[m], d[m], yerr=s[m], fmt="o", ms=10, capsize=4, color=col, mfc=col if c_ else "none",
                        mew=2, label=fr"$V_0={V:.0f}\,\varepsilon$" if c_ or not conv.any() else None)

ax.axhline(0, color="k", lw=1)
if VS_N:
    ax.set_xlim(12, 53)
    ax.set_xlabel(r"$N$")
else:
    ax.set_xlim(0, 0.07)
    ax.set_xlabel(r"$1/N$")
ax.set_ylabel(r"$(E_{\rm DMC}-E_{\rm GP})/N\,[\varepsilon]$")
ax.grid(True, ls="--", alpha=0.6)
h, lab = ax.get_legend_handles_labels()
h += [plt.Line2D([], [], marker="o", ls="", color="0.4", ms=10), plt.Line2D([], [], marker="o", ls="", color="0.4",
                                                                            mfc="none", mew=2, ms=10)]
lab += [r"$\tau=32$", r"$\tau=8$ (older runs)"]
ax.legend(h, lab, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
FIG = f"gp_delta_vs_{'N' if VS_N else 'invN'}_all{'_with200' if WITH200 else ''}.pdf"
fig.savefig(os.path.join("final_imag", FIG), bbox_inches="tight")

os.makedirs(OUT, exist_ok=True)
np.savetxt(os.path.join(OUT, "delta_combined.csv"), np.array(out_rows), delimiter=",", comments="",
           header="V0,N,E_DMC_per_N,err,E_GP_per_N,Delta,Delta_err,walkers,tau", fmt="%.10g")
print(f"Saved final_imag/{FIG} and", os.path.join(OUT, "delta_combined.csv"))
