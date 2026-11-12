"""
N - lambda_0 across the critical window, N = 40: exact two-mode, Bogoliubov N' (below threshold, diverges), and DMC, mixed and 2 DMC - VMC.
DMC comes from the Z2-symmetric cat trial (numpy_arrays_DMC_cat, up to |V0| = 27 and at 30) and, from |V0| = 30, from the single-branch trial with the
cat rebuilt from one branch (numpy_arrays_DMC_branch). The original-trial points (critical_window_calibration_N40.csv) are shown for comparison.
The stored extrapolated matrix is used, as it was formed with the VMC of the same run.

Estimator: top eigenvalue in the even, lambda-periodic plane-wave block (|n| <= 8), lambda_1 in the odd one, no trace rescaling, and the bin averaging
divided out: element (n, m) / (s_n s_m), s_n = sinc(n dx). err = max(|N - lambda_0 - lambda_1|, noise bar). The half-lambda shift multiplies element
(n, m) by (-1)^(n - m), so the even and odd blocks of a single-branch matrix are already those of the rebuilt cat. The rebuild ignores the overlap of the
two branches and is not used at |V0| = 25.

Lower panel: DMC / two-mode - 1, with the time-step check at V0 = -21. Side panel: two-mode N - lambda_0 at V0 = -|V0|_c(N) against N, with the analytic
asymptote 0.228 N^(1/3) of the quartic oscillator.

Usage (from the repository root, after critical_window_two_mode.py and critical_window_dmc.py):  python python/plots/plot_critical_window.py
Output: final_imag/critical_window_N40.pdf, numpy_arrays_OBDM/critical_window_final_N40.csv
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from critical_window_dmc import measure  # noqa: E402
from critical_window_two_mode import two_mode  # noqa: E402
from obdm_analysis import symmetrize  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 24,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amssymb}",
})

N, L = 40, 40.0
NMAX = 8
VC = 2 * np.pi ** 2 * N / (N - 1)
WIN = N ** (-2 / 3)
A_QUARTIC = 0.22806
XLIM = (0.45, 4.3)
BRANCH_FROM = -29.0         # the rebuild needs separated branches


def stored(base):
    """(V, dtau, mixed matrix, extrapolated matrix) of every 1200-walker run."""
    out = []
    for f in glob.glob(os.path.join(base, "constant_density_1.0", "obdm", "V0=*", "obdm_extrap_1200_*_IS.npy")):
        m = re.match(r"obdm_extrap_\d+_([\d.]+)_\d+_\d+_(-[\d.]+)_", os.path.basename(f))
        out.append((float(m[2]) * N, float(m[1]), np.load(f.replace("obdm_extrap_", "obdm_")), np.load(f)))
    return sorted(out, key=lambda r: (-r[0], r[1]))


def meas(M):
    S = symmetrize(M, L)
    nb = S.shape[0]
    dx = L / nb
    x = (np.arange(nb) + 0.5) * dx
    ns = np.arange(-NMAX, NMAX + 1)
    P = np.column_stack([np.sqrt(1 / L) * np.exp(2j * np.pi * k * x) for k in ns])
    s = np.sinc(ns * dx)
    C = (P.conj().T @ S @ P).real * dx * dx / np.outer(s, s)
    even = ns % 2 == 0
    d = N - np.linalg.eigvalsh(C[np.ix_(even, even)])[-1]
    l1 = np.linalg.eigvalsh(C[np.ix_(~even, ~even)])[-1]
    return d, max(abs(d - l1), measure(M)["bar"])


cat = [(V, dt, *meas(m), *meas(x)) for V, dt, m, x in stored("numpy_arrays_DMC_cat")]
br = [(V, dt, *meas(m), *meas(x)) for V, dt, m, x in stored("numpy_arrays_DMC_branch") if V <= BRANCH_FROM]
cmain = np.array([c for c in cat if c[1] == 1e-3])
chalf = np.array([c for c in cat if c[1] != 1e-3])
B = np.array(br)
tm = lambda V: two_mode(V, N)[0]
tc, tb, th = (np.array([tm(V) for V in A[:, 0]]) for A in (cmain, B, chalf))

rows = [(V, 0, dt, m, em, x, ex, t) for (V, dt, m, em, x, ex), t in zip(cmain, tc)] + [(V, 1, dt, m, em, x, ex, t) for (V, dt, m, em, x, ex), t in zip(B, tb)] \
    + [(V, 0, dt, m, em, x, ex, t) for (V, dt, m, em, x, ex), t in zip(chalf, th)]
np.savetxt(os.path.join("numpy_arrays_OBDM", "critical_window_final_N40.csv"), np.array(rows), delimiter=",", comments="", fmt="%.8g",
           header="V0,trial_0cat_1branch,dtau,mixed,mixed_err,extrap,extrap_err,two_mode")

TM = np.loadtxt(os.path.join("numpy_arrays_GP", "critical_window_two_mode_N40.csv"), delimiter=",", skiprows=1)
TH = np.loadtxt(os.path.join("numpy_arrays_GP", "critical_window_threshold_vs_N.csv"), delimiter=",", skiprows=1)
OLD = np.loadtxt(os.path.join("numpy_arrays_OBDM", "critical_window_calibration_N40.csv"), delimiter=",", skiprows=1)
xg = np.geomspace(XLIM[0], XLIM[1], 300)
tg = np.array([tm(-a * VC) for a in xg])
xc, xb = -cmain[:, 0] / VC, -B[:, 0] / VC
CAT, BR = "k", "C2"
TICKS = [0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]

fig = plt.figure(figsize=(28, 17))
gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[2.2, 1], hspace=0.06, wspace=0.2)
ax, res, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

ax.axvspan(1 - WIN, 1 + WIN, color="C1", alpha=0.25, lw=0, label=r"critical window $|t|\lesssim N^{-2/3}$")
ax.loglog(TM[:, 0], TM[:, 3], "--", color="C3", lw=3.5, label=r"Bogoliubov $N'=\frac12(A/\omega-1)$")
ax.loglog(xg, tg, "-", color="C0", lw=4, label="exact two-mode")
ax.plot(-OLD[2:, 0] / VC, OLD[2:, 7], "x", color="0.55", ms=14, mew=3, label="original trial, extrapolated")
ax.errorbar(xc, cmain[:, 2], yerr=cmain[:, 3], fmt="o", color=CAT, ms=13, capsize=4, mew=2, label="cat trial, mixed")
ax.errorbar(xc, cmain[:, 4], yerr=cmain[:, 5], fmt="o", color=CAT, mfc="none", ms=13, capsize=4, mew=2, label="cat trial, extrapolated")
ax.errorbar(xb, B[:, 2], yerr=B[:, 3], fmt="s", color=BR, ms=13, capsize=4, mew=2, label="branch rebuild, mixed")
ax.errorbar(xb, B[:, 4], yerr=B[:, 5], fmt="s", color=BR, mfc="none", ms=17, capsize=4, mew=2, label="branch rebuild, extrapolated")
ax.axvline(1, color="0.4", ls="--", lw=2)
ax.set_ylabel(r"$N-\lambda_0$ (exact ground state)")
ax.set_ylim(0.01, 40)
ax.grid(True, which="major", ls="--", alpha=0.4)
ax.legend(loc="lower right")
ax.tick_params(labelbottom=False)
sec = ax.secondary_xaxis("top", functions=(lambda x: np.sqrt(np.abs(x)), lambda e: np.asarray(e) ** 2))
sec.xaxis.set_major_locator(FixedLocator([0.7, 0.8, 1.0, 1.2, 1.5, 2.0]))
sec.xaxis.set_minor_locator(NullLocator())
sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
sec.set_xlabel(r"$\sqrt{N}\eta/(\sqrt{N}\eta)_c$", labelpad=12)

res.axvspan(1 - WIN, 1 + WIN, color="C1", alpha=0.25, lw=0)
res.axhline(0, color="C0", lw=4)
res.axvline(1, color="0.4", ls="--", lw=2)
res.errorbar(xc, cmain[:, 2] / tc - 1, yerr=cmain[:, 3] / tc, fmt="o", color=CAT, ms=13, capsize=4, mew=2)
res.errorbar(xc, cmain[:, 4] / tc - 1, yerr=cmain[:, 5] / tc, fmt="o", color=CAT, mfc="none", ms=13, capsize=4, mew=2)
res.errorbar(xb, B[:, 2] / tb - 1, yerr=B[:, 3] / tb, fmt="s", color=BR, ms=13, capsize=4, mew=2)
res.errorbar(xb, B[:, 4] / tb - 1, yerr=B[:, 5] / tb, fmt="s", color=BR, mfc="none", ms=17, capsize=4, mew=2)
res.errorbar(-chalf[:, 0] / VC * 1.012, chalf[:, 4] / th - 1, yerr=chalf[:, 5] / th, fmt="^", color=CAT, mfc="none", ms=13, capsize=4, mew=2,
             label=r"cat, extrapolated, $\Delta\tau/2$")
res.set_xscale("log")
res.set_ylim(-0.25, 0.2)
res.set_xlabel(r"$|V_0|/|V_0|_c$")
res.set_ylabel(r"DMC / two-mode $-\,1$")
res.grid(True, which="major", ls="--", alpha=0.4)
res.legend(loc="lower right")
for a in (ax, res):
    a.set_xlim(*XLIM)
    a.xaxis.set_major_locator(FixedLocator(TICKS))
    a.xaxis.set_minor_locator(NullLocator())
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.tick_params(labelbottom=False)

Ns = np.logspace(np.log10(15), np.log10(3e4), 50)
axn.loglog(Ns, A_QUARTIC * Ns ** (1 / 3), ":", color="0.3", lw=3.5, label=r"$0.228\,N^{1/3}$")
axn.loglog(TH[:, 0], TH[:, 1], "-", color="C0", lw=2)
axn.loglog(TH[:, 0], TH[:, 1], "o", color="C0", ms=13, label="two-mode")
axn.set_title("exact two-mode, no QMC", fontsize=30, pad=14)
axn.set_xlabel(r"$N$")
axn.set_ylabel(r"$N-\lambda_0$ at $|V_0|_c(N)$")
axn.grid(True, which="major", ls="--", alpha=0.4)
axn.legend(loc="upper left")

os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "critical_window_N40.pdf"), bbox_inches="tight")
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], bbox_inches="tight", dpi=60)      # optional PNG preview
print("Saved final_imag/critical_window_N40.pdf, numpy_arrays_OBDM/critical_window_final_N40.csv")
