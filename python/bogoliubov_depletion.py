"""
Bogoliubov quantum depletion around the single-branch GP state (cf. Nagy, Szirmai, Domokos, Fig. 8, g_c = 0),
for H = sum_i (-1/2 d^2_i) + (V0/N) sum_{i<j} cos(2 pi x_i) cos(2 pi x_j), N = 40.

GP branch phi_A (same equations as julia/gp/gp_scf.jl, via python/dmc_gp_delta._ground): ground state of
h[A] = -1/2 d^2 + A cos(2 pi x), A = V0 (N-1)/N Theta, mu = its eigenvalue. Bogoliubov-de Gennes with the
separable kernel W(x, y) = (V0/N) cos(2 pi x) cos(2 pi y):
    L = h[A] - mu,   X = V0 (N-1)/N |cos phi_A><cos phi_A|      (exchange / pairing; phi_A real)
    [[ Q(L+X)Q,  QXQ ], [ -QXQ, -Q(L+X)Q ]] (u, v) = omega (u, v),   Q = 1 - |phi_A><phi_A|.
Since W has period lambda, only Bloch momentum q = 0 (lambda-periodic modes, plane waves n = -NMAX..NMAX on one
cell) feels the pairing: modes with q != 0 are free particles with v = 0 and carry no depletion.
Depletion: rho_dep = sum_k v_k v_k^T (positive-norm modes, |u|^2 - |v|^2 = 1), N' = tr rho_dep (an absolute
number of atoms, independent of N up to the (N-1)/N factor). Split:
    N'_in  = tr(P rho_dep), P = projector on span{phi_+, phi_-} = span{phi_A, phi_B}  (the two-mode space);
    N'_out = N' - N'_in   (atoms outside both density-wave modes = what QMC measures as N - N_0 - N_1).
Below threshold phi_A = 1 and phi_- = sqrt(2) cos(2 pi x).
Second figure, Nagy's axis (2008, Fig. 8: kappa = 0, Delta_C = -300, N U_0 = -10, so |Delta_c~| = 295):
sqrt(N) eta = sqrt(|V0[omega_R]| |Delta_c~| / 2) in units of omega_R, V0[omega_R] = V0[eps] / 2 pi^2. The map covers
only the U_1 term; in this model another detuning only rescales the x axis (N U_0 = -100: |Delta_c~| = 250, factor 0.92).
Two-mode curves: the exact ground state of H2, and Bogoliubov restricted to the two-mode space, which it approaches as
N grows (ratio 1.25 at N = 40, 1.01 at N = 800, V0 = -30). This is not N'_in = N' - N'_out: the full collective mode leaves
the two-mode space more and more deep in the phase.

Usage (from the repository root):  python python/bogoliubov_depletion.py
Output: final_imag/bogoliubov_depletion_N40.pdf, bogoliubov_depletion_sqrtNeta_N40.pdf, numpy_arrays_GP/bogoliubov_depletion_N40.csv
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc_gp_delta import _ground, NMAX, EPSK  # noqa: E402
from obdm_analysis import load_raw, symmetrize  # noqa: E402
from pump_axis import add_pump_axis  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 24,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amssymb}",
})

N = 40
R = (N - 1) / N
n = np.arange(-NMAX, NMAX + 1)
M = len(n)
COS = 0.5 * (np.eye(M, k=1) + np.eye(M, k=-1))          # cos(2 pi x) in the plane-wave basis


def gp_branch(V0):
    """(c, A, mu) of the GP branch at coupling V0 (uniform below threshold)."""
    f = lambda A: A - V0 * R * _ground(A)[1]
    lo, hi = V0 * R, -1e-8
    if V0 >= 0 or f(lo) * f(hi) > 0:
        A = 0.0
    else:
        A = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)
    h = np.diag(EPSK * n.astype(float) ** 2) + A * COS
    w, v = np.linalg.eigh(h)
    c = v[:, 0] * np.sign(v[NMAX, 0])
    return c, A, w[0], h


def bogoliubov(V0):
    c, A, mu, h = gp_branch(V0)
    Q = np.eye(M) - np.outer(c, c)
    g = COS @ c
    X = V0 * R * np.outer(g, g)
    Lp = Q @ (h - mu * np.eye(M) + X) @ Q
    Xp = Q @ X @ Q
    B = np.block([[Lp, Xp], [-Xp, -Lp]])
    w, V = np.linalg.eig(B)
    w, V = w.real, V.real
    rho = np.zeros((M, M))
    omega_min = np.inf
    for k in np.where(w > 1e-7)[0]:
        u, v = V[:M, k], V[M:, k]
        nrm = u @ u - v @ v
        if nrm <= 1e-12:
            continue
        u, v = u / np.sqrt(nrm), v / np.sqrt(nrm)
        rho += np.outer(v, v)
        omega_min = min(omega_min, w[k])
    # two-mode space span{phi_+, phi_-}
    cp, cm = c * (n % 2 == 0), c * (n % 2 != 0)
    cp /= np.linalg.norm(cp)
    cm = cm / np.linalg.norm(cm) if np.linalg.norm(cm) > 1e-12 else np.where(np.abs(n) == 1, 1 / np.sqrt(2), 0.0)
    P = np.outer(cp, cp) + np.outer(cm, cm)
    Nd = np.trace(rho)
    Nin = np.trace(P @ rho)
    lam = np.sort(np.linalg.eigvalsh(rho))[::-1]
    return Nd, Nin, Nd - Nin, omega_min, lam[:3]


VC = 2 * np.pi ** 2 / R
V0S = np.unique(np.concatenate([np.linspace(-0.5, -4, 8), np.linspace(-5, -19, 15), -VC * np.array([0.98, 0.99, 0.995, 1.005, 1.01, 1.02]),
                                np.linspace(-21, -40, 20), np.linspace(-45, -340, 60)]))[::-1]
rows = []
for V in V0S:
    Nd, Nin, Nout, om, lam = bogoliubov(V)
    rows.append((V, Nd, Nin, Nout, om, *lam))
Rw = np.array(rows)
np.savetxt(os.path.join("numpy_arrays_GP", f"bogoliubov_depletion_N{N}.csv"), Rw, delimiter=",", comments="",
           fmt="%.10g", header="V0,N_dep,N_dep_in_two_mode_space,N_dep_outside,omega_min,mode0,mode1,mode2")

print(f"N = {N}, |V0|_c = {VC:.3f}")
print("    V0     N'        N'_in      N'_out     omega_min   (largest depleted-mode occupations)")
for V in (-5, -15, -19, -20, -21, -22, -25, -30, -40, -60, -100, -150, -200):
    Nd, Nin, Nout, om, lam = bogoliubov(float(V))
    print(f"  {V:6.0f}  {Nd:9.4f}  {Nin:9.4f}  {Nout:.2e}   {om:9.3f}    {lam[0]:.4f} {lam[1]:.2e} {lam[2]:.2e}")

# QMC at N = 40 (normalised to the measured trace). Below threshold the depletion lives in the odd orbital
# phi_- = sqrt(2) cos(2 pi x), so lambda_1 is compared with N' (N - N0 - N1 would subtract it). Just above threshold
# lambda_1 includes the cat weight and is not depletion, so those points are not shown; N - N0 - N1 is compared with N'_out.
D = np.loadtxt(os.path.join("numpy_arrays_OBDM", f"depletion_N{N}.csv"), delimiter=",", skiprows=1)
dmc = D[D[:, 1] == 1]
above = dmc[:, 0] < -VC
dep = N * (1 - dmc[:, 2] - dmc[:, 3])
pos = dep > 0
BASE = os.path.join("numpy_arrays_DMC", "constant_density_1.0")
V0_BELOW = [-10.0, -15.0, -20.0]              # N = 40 runs with an OBDM below threshold (-10 is 200 walkers, tau = 3)
WINDOW = VC * np.array([1 - N ** (-2 / 3), 1 + N ** (-2 / 3)])    # critical window |t| < N^(-2/3)


def odd_occupation(M, L):
    """(<phi_-|rho|phi_->, rms occupation of the neighbouring modes |n/L - 1| <= 1/4, uncoupled so zero up to noise).
    This is lambda_1 below threshold; the top eigenvalue of the full matrix is biased upwards by noise (-10: 0.21)."""
    M = symmetrize(M, L)
    dx = L / M.shape[0]
    M = M * (N / (np.trace(M) * dx))
    x = (np.arange(M.shape[0]) + 0.5) * dx
    occ = lambda g: (g @ M @ g) * dx * dx * 2 / L
    k = 2 * np.pi * x / L
    other = [occ(f(m * k)) for m in range(30, 51) for f in (np.cos, np.sin) if not (m == 40 and f is np.cos)]
    return occ(np.cos(40 * k)), np.sqrt(np.mean(np.square(other)))


below = []
for V in V0_BELOW:
    v, L = load_raw(BASE, N, V, "VMC")
    d, _ = load_raw(BASE, N, V, "DMC")
    below.append((V, *odd_occupation(d, L), *odd_occupation(2 * d - v, L)))
below = np.array(below)


def draw_dmc(ax, fx):
    """DMC lambda_1 (below threshold) and |N - N0 - N1| (above threshold, open marker = negative value)."""
    x, nz = fx(dmc[:, 0]), N * dmc[:, -1]
    xb = fx(below[:, 0])
    ax.errorbar(xb, below[:, 1], yerr=below[:, 2], fmt="D", color="C4", ms=11, mew=2, capsize=4,
                label=r"DMC mixed: $\lambda_1=\langle\phi_-|\rho^{(1)}|\phi_-\rangle$ (cf.\ $N'$)")
    ax.errorbar(xb, below[:, 3], yerr=below[:, 4], fmt="D", color="C4", mfc="none", ms=11, mew=2, capsize=4,
                label=r"DMC extrapolated ($2\,$DMC$-$VMC): same $\lambda_1$")
    ax.errorbar(x[above & pos], dep[above & pos], yerr=nz[above & pos], fmt="s", color="k", ms=11, mew=2, capsize=4,
                label=r"DMC $|N-\lambda_0-\lambda_1|$ (cf.\ $N'_{\rm out}$; open: negative)")
    ax.errorbar(x[above & ~pos], -dep[above & ~pos], yerr=nz[above & ~pos], fmt="s", color="k", mfc="none", ms=11, mew=2,
                capsize=4)
    ax.axvspan(*sorted(fx(WINDOW)), color="C1", alpha=0.35, lw=0, zorder=1,
               label=r"critical window $|t|\lesssim N^{-2/3}$: Gaussian theory invalid")


print(f"Critical window |t| < N^(-2/3): {WINDOW[0]:.2f} < |V0| < {WINDOW[1]:.2f}")
print("Below threshold:  V0   lambda_1 mixed      2DMC-VMC     N'")
for V, a, ea, b, eb in below:
    print(f"  {V:6.0f}  {a:.3f}+-{ea:.3f}  {b:.3f}+-{eb:.3f}  {bogoliubov(V)[0]:.4f}")
print("Above threshold:  V0   N - l0 - l1   noise   N'_out")
for r, d in zip(dmc[above], dep[above]):
    print(f"  {r[0]:6.0f}  {d:+7.3f}  {N * r[-1]:.3f}  {bogoliubov(r[0])[2]:.2e}")
ordered = Rw[:, 0] < -VC                      # N'_out = 0 below threshold, not plotted on the log axis


def two_mode(V0):
    """(E2, lambda_-, one-branch depletion, P(N/2)/P_max) of the two-mode ground state (H as in julia/gp/two_mode.jl).
    Nothing lies outside the two modes, so depletion means: below threshold the odd-mode occupation lambda_-; above
    threshold the smaller eigenvalue of rho^(1) in the n_A > N/2 half of the state (meaningful only if P(n_A) has two
    separate peaks, i.e. P(N/2) << P_max)."""
    c = gp_branch(V0)[0]
    cp, cm = c * (n % 2 == 0), c * (n % 2 != 0)
    cp /= np.linalg.norm(cp)
    cm = cm / np.linalg.norm(cm) if np.linalg.norm(cm) > 1e-12 else np.where(np.abs(n) == 1, 1 / np.sqrt(2), 0.0)
    a, b = (cp + cm) / np.sqrt(2), (cp - cm) / np.sqrt(2)
    Tk = EPSK * n.astype(float) ** 2
    T, t, Th = a @ (Tk * a), a @ (Tk * b), a @ COS @ a
    k = np.arange(N + 1)
    hop = np.sqrt((k[:-1] + 1) * (N - k[:-1]))
    H = np.diag(N * T + V0 / (2 * N) * Th ** 2 * ((2 * k - N) ** 2 - N)) + np.diag(t * hop, 1) + np.diag(t * hop, -1)
    w, v = np.linalg.eigh(H)
    p = np.abs(v[:, 0])                       # nodeless ground state (t < 0)
    p = np.sqrt(0.5 * (p ** 2 + p[::-1] ** 2))    # even state: deep in the phase eigh can mix it with the degenerate odd one
    lm = N / 2 - p[:-1] @ (p[1:] * hop)
    q = np.where(k < N // 2, 0.0, p)          # one branch; the central weight is shared with the other one
    q[N // 2] /= np.sqrt(2)
    q /= np.linalg.norm(q)
    nA = q ** 2 @ k
    gq = q[:-1] @ (q[1:] * hop)
    low = np.linalg.eigvalsh(np.array([[nA, gq], [gq, N - nA]]))[0]
    return w[0], lm, low, p[N // 2] ** 2 / np.max(p ** 2)


Vtm = -np.unique(np.concatenate([np.linspace(0.5, 19, 38), np.linspace(19, 30, 111), np.linspace(30, 340, 156)]))
TM = np.array([(V, *two_mode(V)) for V in Vtm])
np.savetxt(os.path.join("numpy_arrays_GP", f"two_mode_depletion_N{N}.csv"), TM, delimiter=",", comments="", fmt="%.10g",
           header="V0,E2_total,lambda_minus,one_branch_depletion,P_centre_over_P_max")
tm_lo = TM[TM[:, 0] > -VC]                    # lambda_-
tm_hi = TM[TM[:, 4] < 0.1]                    # one branch, once P(n_A) has two separate peaks
print(f"Two-mode depletion: lambda_- up to threshold, one branch from V0 = {tm_hi[:, 0].max():.2f}")
for V in (-10.0, -15.0, -20.0, -25.0, -27.0, -30.0, -40.0, -60.0, -100.0):
    r2 = two_mode(V)
    print(f"  {V:6.0f}  E2 = {r2[0]:10.3f}  lambda_- = {r2[1]:7.4f}  one branch = {r2[2]:.2e}  P(N/2)/P_max = {r2[3]:.1e}  "
          f"N' = {bogoliubov(V)[0]:.4f}  N'_in = {bogoliubov(V)[1]:.4f}")


def two_mode_bogoliubov(V0):
    """Bogoliubov occupation restricted to the two-mode space: one mode phi_perp in span{phi_+, phi_-}, orthogonal to phi_A."""
    c, A, mu, h = gp_branch(V0)
    cp, cm = c * (n % 2 == 0), c * (n % 2 != 0)
    cp, cm = cp / np.linalg.norm(cp), cm / np.linalg.norm(cm)
    perp = (cp @ c) * cm - (cm @ c) * cp
    L, X = perp @ h @ perp - mu, V0 * R * (perp @ COS @ c) ** 2
    return 0.5 * ((L + X) / np.sqrt(L * (L + 2 * X)) - 1)


tm_b = np.array([two_mode_bogoliubov(V) for V in tm_hi[:, 0]])
print("Two-mode space, V0: exact one branch, restricted Bogoliubov, N'_in = N' - N'_out")
for V in (-30.0, -40.0, -60.0, -100.0):
    print(f"  {V:6.0f}  {two_mode(V)[2]:.3e}  {two_mode_bogoliubov(V):.3e}  {bogoliubov(V)[1]:.3e}")


def draw_two_mode(ax, fx):
    ax.semilogy(fx(tm_lo[:, 0]), tm_lo[:, 2], "-.", color="C5", lw=3.5, label=r"two-mode exact: $\lambda_-$ (below), $n_A>N/2$ half (above)")
    ax.semilogy(fx(tm_hi[:, 0]), tm_hi[:, 3], "-.", color="C5", lw=3.5)
    ax.semilogy(fx(tm_hi[:, 0]), tm_b, ":", color="C5", lw=2.5, label=r"Bogoliubov restricted to the two modes")


def draw_band(ax, a, b):
    """Resolution of N - lambda_0 - lambda_1 (a difference of numbers ~ 40), so only over the points that use it."""
    ax.fill_between([a, b], 1e-12, 0.2, color="0.85", alpha=0.6, lw=0, zorder=0, label=r"resolution of $N-\lambda_0-\lambda_1$")

fig, ax = plt.subplots(figsize=(15, 11))
ax.semilogy(Rw[:, 0], Rw[:, 1], "-", color="C0", lw=3, label=r"Bogoliubov $N'$ (out of $\varphi_A$)")
ax.semilogy(Rw[:, 0], Rw[:, 2], "--", color="C2", lw=2.5, label=r"$N'_{\rm in}=N'-N'_{\rm out}$: inside span$\{\phi_+,\phi_-\}$")
ax.semilogy(Rw[ordered, 0], Rw[ordered, 3], ":", color="C3", lw=3, label=r"$N'_{\rm out}$: outside span$\{\phi_+,\phi_-\}$")
draw_two_mode(ax, lambda v: v)
draw_dmc(ax, lambda v: v)
draw_band(ax, -205, -VC)
ax.axvline(-VC, color="0.4", ls="--", lw=2)
ax.text(-VC - 5, 2e-5, r"$|V_0|_c(N)$", ha="right", fontsize=28, color="0.3")
ax.set_xlabel(r"$V_0/\varepsilon$")
add_pump_axis(ax)
ax.set_ylabel("depleted atoms")
ax.set_title(fr"$N={N}$, $g=0$", fontsize=30)
ax.set_ylim(1e-5, 10)
ax.set_xlim(-205, 0)
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=22)
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", f"bogoliubov_depletion_N{N}.pdf"), bbox_inches="tight")

# Nagy 2008 axis: sqrt(N) eta in units of omega_R, parameters of their Fig. 8
fnagy = lambda v, dt: np.sqrt(np.abs(v) / (2 * np.pi ** 2) * dt / 2)
fig, ax = plt.subplots(figsize=(15, 11))
ax.semilogy(fnagy(Rw[:, 0], 295.0), Rw[:, 1], "-", color="C0", lw=3, label=r"Bogoliubov $N'$ (out of $\varphi_A$)")
ax.semilogy(fnagy(Rw[:, 0], 295.0), Rw[:, 2], "--", color="C2", lw=2.5, label=r"$N'_{\rm in}=N'-N'_{\rm out}$: inside span$\{\phi_+,\phi_-\}$")
ax.semilogy(fnagy(Rw[ordered, 0], 295.0), Rw[ordered, 3], ":", color="C3", lw=3, label=r"$N'_{\rm out}$: outside span$\{\phi_+,\phi_-\}$")
draw_two_mode(ax, lambda v: fnagy(v, 295.0))
draw_dmc(ax, lambda v: fnagy(v, 295.0))
draw_band(ax, fnagy(VC, 295.0), 50)
NG8 = np.loadtxt(os.path.join("numpy_arrays_GP", "nagy2008_fig8_NU0m10.csv"), delimiter=",", skiprows=3)
ax.semilogy(NG8[:, 0], NG8[:, 1], "x", color="C6", ms=10, mew=2.5, label=r"Nagy et al.\ 2008, Fig.\ 8 ($NU_0=-10$)")
ax.axvline(fnagy(VC, 295.0), color="0.4", ls="--", lw=2)
ax.set_xlabel(r"$\sqrt{N}\eta\,[\omega_R]$")
ax.set_ylabel("depleted atoms")
ax.set_title(fr"$N={N}$, $g=0$, $\kappa=0$, $|\tilde\Delta_c|=295$ ($NU_0=-10$)", fontsize=30)
ax.set_ylim(1e-5, 10)
ax.set_xlim(0, 50)
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=22)
fig.tight_layout()
fig.savefig(os.path.join("final_imag", f"bogoliubov_depletion_sqrtNeta_N{N}.pdf"), bbox_inches="tight")
print("Nagy axis, N' at sqrt(N) eta (|Delta_c~| = 295):")
for V in (-15, -21, -25, -40, -60, -100, -150, -200, -335):
    print(f"  V0 = {V:5d}  sqrt(N) eta = {fnagy(V, 295.0):6.2f}  N' = {bogoliubov(float(V))[0]:.4f}")
i = np.argmax(Rw[:, 1])
print(f"Peak on the grid: V0 = {Rw[i, 0]:.3f}, N' = {Rw[i, 1]:.3f}, sqrt(N) eta = {fnagy(Rw[i, 0], 295.0):.2f}")
print(f"Saved final_imag/bogoliubov_depletion_N{N}.pdf, final_imag/bogoliubov_depletion_sqrtNeta_N{N}.pdf and numpy_arrays_GP/bogoliubov_depletion_N{N}.csv")
