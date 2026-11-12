"""
Energy vs V0 at N = 40: GP, GP + Bogoliubov, VMC and DMC, with the trial of each regime: Z2-symmetric cat trial up to |V0| = 27 (numpy_arrays_DMC_cat),
single-branch trial from |V0| = 30 to 80 (numpy_arrays_DMC_branch). Total energies, 1200 walkers, tau = 8, dtau = 0.001.

The single-branch run at V0 = -25 is not used: its walkers stay in one well, so it gives the energy of an isolated well (0.24 above the even ground state).
At |V0| >= 30 the even-odd splitting is negligible (Delta/2 = 6e-4 at V0 = -30).

E_Bog = (sum_k omega_k - tr A) / 2, A = Q (h - mu + X) Q, with the BdG matrix of python/bogoliubov_depletion.py (below threshold it reduces to the analytic
[omega - e - V0 r / 2] / 2, omega = sqrt(e (e + V0 r)), r = (N - 1) / N). The threshold line and top axis use the finite-N |V0|_c = 2 pi^2 N / (N - 1), where GP turns.

Panels: E/N, full range; E/N, zoomed on the threshold region; E - E_GP against E_Bog; E_DMC - E_GP against N at fixed V0, to see that it stays O(1).
The N-dependence panel uses the original-trial runs (numpy_arrays_DMC; longest, then largest, run of each point), the only ones at several N; the open points are
the new-trial values at N = 40, drawn at N = 41 so they can be seen. Error bars are statistical only.

Usage (from the repository root):  python python/plots/plot_energy_final.py
Output: final_imag/energy_vs_V0_N40.pdf, numpy_arrays_GP/energy_final_N40.csv, numpy_arrays_GP/energy_scaling_vs_N.csv
Note: python/plots/plot_energy_vs_V0.py writes the same PDF name with the original trial.
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import _ground, gp_energy, NMAX, EPSK  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 25,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amssymb}",
})

N, L = 40, 40.0
WALKERS, STEPS, DTAU = 1200, 8000, 0.001
R_ = (N - 1) / N
VC = 2 * np.pi ** 2 / R_
TRIALS = {"cat": ("numpy_arrays_DMC_cat", (-10.0, -15.0, -18.5, -19.0, -19.5, -20.0, -20.5, -21.0, -21.5, -22.0, -23.0, -25.0, -27.0), "s"),
          "branch": ("numpy_arrays_DMC_branch", (-30.0, -35.0, -40.0, -50.0, -60.0, -80.0), "D")}
SCALING_V = (-15.0, -20.0, -25.0, -30.0, -40.0, -60.0)
SCALING_N = (15, 20, 25, 30, 35, 40, 50)
n = np.arange(-NMAX, NMAX + 1)
M = len(n)
COS = 0.5 * (np.eye(M, k=1) + np.eye(M, k=-1))


def e_bog(V0, Np=N):
    R_ = (Np - 1) / Np
    f = lambda A: A - V0 * R_ * _ground(A)[1]
    lo, hi = V0 * R_, -1e-8
    A = 0.0 if f(lo) * f(hi) > 0 else brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)
    h = np.diag(EPSK * n.astype(float) ** 2) + A * COS
    w0, v0 = np.linalg.eigh(h)
    c, mu = v0[:, 0], w0[0]
    Q = np.eye(M) - np.outer(c, c)
    g = COS @ c
    X = V0 * R_ * np.outer(g, g)
    Lp = Q @ (h - mu * np.eye(M) + X) @ Q
    Xp = Q @ X @ Q
    w, V = np.linalg.eig(np.block([[Lp, Xp], [-Xp, -Lp]]))
    w, V = w.real, V.real
    som = sum(w[k] for k in np.where(w > 1e-7)[0] if V[:M, k] @ V[:M, k] - V[M:, k] @ V[M:, k] > 1e-12)
    return 0.5 * (som - np.trace(Lp))


def load(base, V):
    v = V / N
    d = os.path.join(base, "constant_density_1.0", "energy", f"V0={v:.3f}")
    e = np.load(os.path.join(d, f"energy_{WALKERS}_{DTAU:.4f}_{N}_{STEPS}_{v:.3f}_{L:.1f}_IS.npy"))
    ev = np.load(os.path.join(d, f"energy_vmc_{N}_{v:.3f}_{L:.1f}.npy"))
    return e[0], e[1], ev[0], ev[1]


data = {t: np.array([(V, gp_energy(V, N) * N, e_bog(V), *load(b, V)) for V in Vs]) for t, (b, Vs, _) in TRIALS.items()}   # V, GP, Bog, DMC, err, VMC, err
A = np.vstack([np.c_[data[t], np.full(len(data[t]), i)] for i, t in enumerate(TRIALS)])
np.savetxt(os.path.join("numpy_arrays_GP", f"energy_final_N{N}.csv"), A, delimiter=",", comments="", fmt="%.10g",
           header="V0,E_GP,E_Bog,E_DMC,E_DMC_err,E_VMC,E_VMC_err,trial_0cat_1branch")
print("    V0      E_GP    E_Bog     E_VMC              E_DMC           DMC-GP  DMC-(GP+Bog)")
for V, g, b, d, ed, v, ev, _ in A:
    print(f"{V:7.1f} {g:9.3f} {b:8.4f} {v:9.3f}+-{ev:.3f} {d:9.3f}+-{ed:.3f} {d - g:+8.3f} {d - g - b:+8.3f}")

Vd = np.unique(np.concatenate([np.linspace(-85, -32, 80), np.linspace(-32, -VC * 1.02, 160), -VC * np.array([1.0002, 1.001, 1.005, 1.01, 0.9998, 0.999, 0.995, 0.99]),
                               np.linspace(-VC * 0.98, -8, 80), np.linspace(-8, -0.5, 20)]))
Eg = np.array([gp_energy(v, N) * N for v in Vd])
Eb = np.array([e_bog(v) for v in Vd])


def scaling_run(V, Np):
    """(E_DMC - E_GP, err, walkers, tau) of the longest, then largest, original-trial run at (V, Np), or None."""
    v = V / Np
    d = os.path.join("numpy_arrays_DMC", "constant_density_1.0", "energy", f"V0={v:.3f}")
    best = None
    for f in glob.glob(os.path.join(d, "energy_*_IS.npy")):
        m = re.match(rf"energy_(\d+)_([\d.]+)_{Np}_(\d+)_", os.path.basename(f))
        if m and (best is None or (float(m[2]) * int(m[3]), int(m[1])) > best[0]):
            best = ((float(m[2]) * int(m[3]), int(m[1])), f)
    if best is None:
        return None
    e = np.load(best[1])
    return e[0] - gp_energy(V, Np) * Np, e[1], best[0][1], best[0][0]


SC = np.array([(V, Np, *r, e_bog(V, Np)) for V in SCALING_V for Np in SCALING_N if (r := scaling_run(V, Np)) is not None])
np.savetxt(os.path.join("numpy_arrays_GP", "energy_scaling_vs_N.csv"), SC, delimiter=",", comments="", fmt="%.10g",
           header="V0,N,E_DMC_minus_E_GP,err,walkers,tau,E_Bog")
print("\nE_DMC - E_GP against N, original trial (E_Bog in brackets)")
for V in SCALING_V:
    print(f"{V:6.0f} " + "  ".join(f"N{int(r[1])}: {r[2]:+.3f}+-{r[3]:.3f} ({r[6]:+.3f})" for r in SC[SC[:, 0] == V]))

fig, axs = plt.subplots(2, 2, figsize=(30, 21))
(ax, azn), (adz, an) = axs
ax.plot(Vd, Eg / N, "k-", lw=2.5, label="GP")
ax.plot(Vd, (Eg + Eb) / N, "-", color="C3", lw=2.5, label="GP + Bogoliubov")
azn.plot(Vd, Eg / N, "k-", lw=2.5)
azn.plot(Vd, (Eg + Eb) / N, "-", color="C3", lw=2.5)
adz.axhline(0, color="k", lw=2.5)
adz.plot(Vd, Eb, "-", color="C3", lw=2.5, label=r"$E_{\rm Bog}$")
for t, (_, _, marker) in TRIALS.items():
    D = data[t]
    for y, ye, c, mf, name, z in ((3, 4, "C0", None, "DMC", 3), (5, 6, "C1", "none", "VMC", 4)):
        kw = dict(fmt=marker, color=c, ms=12, capsize=4, mew=2, zorder=z)
        if mf:
            kw["mfc"] = mf
        ax.errorbar(D[:, 0], D[:, y] / N, yerr=D[:, ye] / N, label=fr"{name}, {t} trial", **kw)
        azn.errorbar(D[:, 0], D[:, y] / N, yerr=D[:, ye] / N, **kw)
        adz.errorbar(D[:, 0], D[:, y] - D[:, 1], yerr=D[:, ye], **kw)
ax.set_ylabel(r"$E/N\,[\varepsilon]$")
ax.legend(loc="lower right")
azn.set_ylabel(r"$E/N\,[\varepsilon]$ (zoom)")
adz.set_ylabel(r"$E-E_{\rm GP}\,[\varepsilon]$")
adz.set_ylim(-6, 2.6)
adz.legend(loc="upper left")
for a in (ax, adz):
    a.set_xlim(-85, 0)
azn.set_xlim(-31, -8)
azn.set_ylim(-1.0, 0.05)
for a in (ax, azn, adz):
    a.axvline(-VC, color="0.4", ls="--", lw=2)
    a.set_xlabel(r"$V_0/\varepsilon$")
for a in axs.flat:
    a.grid(True, ls="--", alpha=0.6)
for a in (ax, azn):
    sec = a.secondary_xaxis("top", functions=(lambda v: np.sqrt(np.abs(v) / VC), lambda e: -VC * np.asarray(e) ** 2))
    sec.set_xlabel(r"$\eta/\eta_c(N)$", labelpad=10)
    if a is ax:
        sec.set_xticks([0.5, 1.0, 1.5, 2.0])
ax.text(-VC - 1, -7.6, r"$|V_0|_c(N)=20.25$", rotation=90, fontsize=23, ha="right", va="bottom", color="0.3")

Nd = np.linspace(14, 52, 40)
an.axhline(0, color="k", lw=2)
for i, V in enumerate(SCALING_V):
    c = f"C{i}"
    S = SC[SC[:, 0] == V]
    an.plot(Nd, [e_bog(V, x) for x in Nd], "-", color=c, lw=2.5)
    an.errorbar(S[:, 1], S[:, 2], yerr=S[:, 3], fmt="o", color=c, ms=12, capsize=4, mew=2, label=fr"$V_0={V:.0f}$")
    new = A[A[:, 0] == V][0]
    an.plot(N + 1.0, new[3] - new[1], "o", color=c, mfc="white", ms=12, mew=2.5, zorder=5)
h, l = an.get_legend_handles_labels()
h += [Line2D([], [], color="0.3", lw=2.5), Line2D([], [], marker="o", color="0.3", mfc="white", ms=12, mew=2.5, ls="")]
l += [r"$E_{\rm Bog}(N)$", r"new trial, $N=40$"]
an.legend(h, l, loc="lower center", ncol=4, columnspacing=1.0, handletextpad=0.4)
an.set_xlim(14, 52)
an.set_ylim(-5.4, 0.2)
an.set_xlabel(r"$N$")
an.set_ylabel(r"$E_{\rm DMC}-E_{\rm GP}\,[\varepsilon]$")
fig.suptitle(r"$g=0$; $N=40$ except lower right", fontsize=34, y=0.995)
fig.tight_layout()
fig.savefig(os.path.join("final_imag", f"energy_vs_V0_N{N}.pdf"), bbox_inches="tight")
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], bbox_inches="tight", dpi=50)       # optional PNG preview
print(f"Saved final_imag/energy_vs_V0_N{N}.pdf, numpy_arrays_GP/energy_final_N{N}.csv")
