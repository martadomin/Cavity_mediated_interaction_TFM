"""
N dependence of the two-mode correction (E2 - E_GP)/N near threshold (data from julia/gp/two_mode_scaling.jl,
GP orbitals, no optimisation), x = V0/|V0|_c(N), |V0|_c(N) = 2 pi^2 N/(N-1).

  two_mode_scaling_vs_x.pdf   (E2 - E_GP)/N vs x, one curve per N (left: linear; right: |.| on a log scale)
  two_mode_scaling_vs_N.pdf   |E2 - E_GP|/N vs N at x = -1, -1.5, -3 and at the dip, log-log, with power-law fits

Usage (from the repository root, after julia julia/gp/two_mode_scaling.jl):
    python python/plots/plot_two_mode_scaling.py
"""

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
    "legend.fontsize": 26,
    "text.usetex": True,
})


def read_csv(path):
    with open(path) as f:
        cols = f.readline().strip().split(",")
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return {c: data[:, i] for i, c in enumerate(cols)}


cur = read_csv("numpy_arrays_GP/two_mode_scaling_curves.csv")
summ = read_csv("numpy_arrays_GP/two_mode_scaling_summary.csv")
Ns = summ["N"].astype(int)
x = cur["x"]
cmap = plt.get_cmap("viridis")
cols = [cmap(0.9 * i / (len(Ns) - 1)) for i in range(len(Ns))]
os.makedirs("final_imag", exist_ok=True)

# ---- plot 1
fig, (ax, axl) = plt.subplots(1, 2, figsize=(26, 10))
for i, N in enumerate(Ns):
    y = cur[f"Delta_N{N}"]
    ax.plot(x, y, "-", color=cols[i], lw=2.5, label=fr"$N={N}$")
    axl.semilogy(x, np.abs(y), "-", color=cols[i], lw=2.5)
    ax.plot(summ["x_min"][i], summ["depth"][i], "o", color=cols[i], ms=9)
    axl.plot(summ["x_min"][i], abs(summ["depth"][i]), "o", color=cols[i], ms=9)
for a in (ax, axl):
    a.axvline(-1, color="0.4", ls="--", lw=2)
    a.set_xlabel(r"$x=V_0/|V_0|_c(N)$")
    a.set_xlim(-3, -0.8)
    a.grid(True, ls="--", alpha=0.6)
ax.axhline(0, color="k", lw=1)
ax.set_ylabel(r"$(E_2-E_{\rm GP})/N\,[\varepsilon]$")
axl.set_ylabel(r"$|E_2-E_{\rm GP}|/N\,[\varepsilon]$")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join("final_imag", "two_mode_scaling_vs_x.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- plot 2
fig, ax = plt.subplots(figsize=(13, 10))
series = [("Delta_x-1", r"$x=-1$ (threshold)", "C0", "o"), ("depth", r"dip minimum", "k", "*"),
          ("Delta_x-1.5", r"$x=-1.5$", "C1", "s"), ("Delta_x-3", r"$x=-3$", "C2", "^")]
NN = np.logspace(np.log10(Ns.min() / 1.5), np.log10(Ns.max() * 1.5), 50)
print("log-log fits |E2 - E_GP|/N = A N^p (all N; last-two-N local slope in brackets)")
for key, lab, c, mk in series:
    y = np.abs(summ[key])
    p, lA = np.polyfit(np.log(Ns), np.log(y), 1)
    loc = np.diff(np.log(y))[-1] / np.diff(np.log(Ns))[-1]
    ax.loglog(Ns, y, mk, color=c, ms=13 if mk == "*" else 11, label=fr"{lab}: $N^{{{p:.2f}}}$")
    ax.loglog(NN, np.exp(lA) * NN ** p, "--", color=c, lw=1.5)
    print(f"  {lab:28s} p = {p:.3f}  [{loc:.3f}]")
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$|E_2-E_{\rm GP}|/N\,[\varepsilon]$")
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(loc="lower left", fontsize=24)
fig.tight_layout()
fig.savefig(os.path.join("final_imag", "two_mode_scaling_vs_N.pdf"), bbox_inches="tight")
print("Saved final_imag/two_mode_scaling_vs_x.pdf and final_imag/two_mode_scaling_vs_N.pdf")
