"""
Bogoliubov quantum depletion around the single-branch GP state for several N (same calculation as
python/bogoliubov_depletion.py, which is N = 40 only).

With Kac scaling N enters only through the (N-1)/N factor of the GP mean field and of the BdG exchange term,
and through the threshold |V0|_c(N) = 2 pi^2 N/(N-1). So N' (an absolute number of atoms) is almost
N-independent and the depleted fraction N'/N falls as 1/N.
Panels: N' vs V0, N'/N vs V0 (log), and N' vs x = V0/|V0|_c(N) (same threshold for every N), with the
split into the two-mode space (N'_in) and outside (N'_out, dotted).

Usage (from the repository root):  python python/bogoliubov_depletion_vs_N.py
Output: final_imag/bogoliubov_depletion_vs_N.pdf, numpy_arrays_GP/bogoliubov_depletion_vs_N.csv
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc_gp_delta import _ground, NMAX, EPSK  # noqa: E402
from pump_axis import add_pump_axis  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 22,
    "text.usetex": True,
})

NS = [15, 20, 40, 100, 1000]
n = np.arange(-NMAX, NMAX + 1)
M = len(n)
COS = 0.5 * (np.eye(M, k=1) + np.eye(M, k=-1))


def bogoliubov(V0, N):
    """(N', N'_in, N'_out) around the GP branch at (V0, N)."""
    R = (N - 1) / N
    f = lambda A: A - V0 * R * _ground(A)[1]
    lo, hi = V0 * R, -1e-8
    A = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14) if (V0 < 0 and f(lo) * f(hi) < 0) else 0.0
    h = np.diag(EPSK * n.astype(float) ** 2) + A * COS
    w, v = np.linalg.eigh(h)
    c = v[:, 0] * np.sign(v[NMAX, 0])
    mu = w[0]
    Q = np.eye(M) - np.outer(c, c)
    g = COS @ c
    X = V0 * R * np.outer(g, g)
    Lp = Q @ (h - mu * np.eye(M) + X) @ Q
    Xp = Q @ X @ Q
    ev, V = np.linalg.eig(np.block([[Lp, Xp], [-Xp, -Lp]]))
    ev, V = ev.real, V.real
    rho = np.zeros((M, M))
    for k in np.where(ev > 1e-7)[0]:
        u, vv = V[:M, k], V[M:, k]
        nrm = u @ u - vv @ vv
        if nrm > 1e-12:
            rho += np.outer(vv, vv) / nrm
    cp, cm = c * (n % 2 == 0), c * (n % 2 != 0)
    cp /= np.linalg.norm(cp)
    cm = cm / np.linalg.norm(cm) if np.linalg.norm(cm) > 1e-12 else np.where(np.abs(n) == 1, 1 / np.sqrt(2), 0.0)
    P = np.outer(cp, cp) + np.outer(cm, cm)
    Nd, Nin = np.trace(rho), np.trace(P @ rho)
    return Nd, Nin, Nd - Nin


XG = np.unique(np.concatenate([np.linspace(0.2, 0.95, 16), 1 + np.array([-0.03, -0.015, -0.007, 0.007, 0.015, 0.03]),
                               np.linspace(1.05, 1.5, 15), np.linspace(1.6, 10.0, 40)]))
rows = []
for N in NS:
    vc = 2 * np.pi ** 2 * N / (N - 1)
    for x in XG:
        V = -x * vc
        Nd, Nin, Nout = bogoliubov(V, N)
        rows.append((N, V, x, Nd, Nin, Nout))
Rw = np.array(rows)
np.savetxt(os.path.join("numpy_arrays_GP", "bogoliubov_depletion_vs_N.csv"), Rw, delimiter=",", comments="",
           fmt="%.10g", header="N,V0,x=V0/|V0|_c(N),N_dep,N_dep_in,N_dep_out")

print("N'  (atoms out of the GP branch) at fixed V0; in brackets N'_out")
print("   V0   " + "  ".join(f"N={N:<5d}       " for N in NS))
for V in (-15.0, -21.0, -25.0, -30.0, -40.0, -60.0, -150.0):
    print(f"  {V:5.0f}  " + "  ".join(f"{bogoliubov(V, N)[0]:.4f} ({bogoliubov(V, N)[2]:.1e})" for N in NS))
print("\nN'/N at fixed V0")
for V in (-25.0, -40.0, -60.0):
    print(f"  {V:5.0f}  " + "  ".join(f"N={N}: {bogoliubov(V, N)[0] / N:.2e}" for N in NS))

cmap = plt.get_cmap("viridis")
cols = {N: cmap(0.9 * i / (len(NS) - 1)) for i, N in enumerate(NS)}
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(42, 11))
for N in NS:
    r = Rw[Rw[:, 0] == N]
    r = r[np.argsort(r[:, 1])]
    a1.semilogy(r[:, 1], r[:, 3], "-", color=cols[N], lw=2.5, label=fr"$N={N}$")
    a1.semilogy(r[:, 1], np.maximum(r[:, 5], 1e-12), ":", color=cols[N], lw=2)
    a2.semilogy(r[:, 1], r[:, 3] / N, "-", color=cols[N], lw=2.5, label=fr"$N={N}$")
    a3.semilogy(r[:, 2], r[:, 3], "-", color=cols[N], lw=2.5, label=fr"$N={N}$")
for a in (a1, a2):
    a.set_xlabel(r"$V_0/\varepsilon$")
    a.set_xlim(-205, 0)
    add_pump_axis(a)
a1.set_ylabel(r"$N'$ (solid), $N'_{\rm out}$ (dotted)")
a1.set_ylim(1e-6, 30)
a2.set_ylabel(r"$N'/N$")
a2.set_ylim(1e-8, 1)
a3.set_xlabel(r"$x=V_0/|V_0|_c(N)$")
a3.set_ylabel(r"$N'$")
a3.set_xscale("log")
a3.set_ylim(1e-4, 30)
a3.axvline(1, color="0.4", ls="--", lw=2)
for a in (a1, a2, a3):
    a.grid(True, which="both", ls="--", alpha=0.5)
    a.legend(loc="lower left", fontsize=22)
a1.set_title(r"depleted atoms", fontsize=30)
a2.set_title(r"depleted fraction", fontsize=30)
a3.set_title(r"vs distance from the finite-$N$ threshold", fontsize=30)
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "bogoliubov_depletion_vs_N.pdf"), bbox_inches="tight")
print("Saved final_imag/bogoliubov_depletion_vs_N.pdf and numpy_arrays_GP/bogoliubov_depletion_vs_N.csv")
