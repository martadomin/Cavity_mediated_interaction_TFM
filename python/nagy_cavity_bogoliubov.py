"""
Bogoliubov depletion with the cavity mode kept, Nagy, Szirmai, Domokos, EPJD 48, 127 (2008), Eqs. (1), (2), (12), kappa = 0,
g_c = 0, N -> infinity, units omega_R = 1, plane waves exp(i n k x) on one period (Bloch q = 0).

Parameters u = N U_0, y = sqrt(N) eta, a = alpha_0 / sqrt(N) (real for kappa = 0):
    steady state:  a = y Theta / (Delta_C - u B),  h = T + a^2 u cos^2 + 2 a y cos,  h psi_0 = mu psi_0,
                   Theta = <cos>, B = <cos^2>.
Fluctuations, c = delta alpha, phi = sqrt(N) delta psi; Eqs. (12) are the equations of motion of
    H2 = A c^dag c + phi^dag H0 phi + (c + c^dag) int G (phi + phi^dag),
    A = -Delta_C + u B,  H0 = h - mu,  G = (a u cos^2 + y cos) psi_0.
BdG in the basis (c, phi): [[M, W], [-W, -M]] (U, V) = omega (U, V), M = [[A, G^T], [G, H0]], W = [[0, G^T], [G, 0]].
N' = sum_k |V_k(atoms)|^2 over positive-norm modes, n_ph = sum_k |V_k(c)|^2.
"projected": phi orthogonal to psi_0 (fixed atom number, as in python/bogoliubov_depletion.py); G -> Q G, H0 -> Q H0 Q.
"unprojected": the linearised GP equations as they stand, Nagy's matrix (17), which has the zero mode delta psi ~ psi_0.

Runs: (1) u = 0, Delta_C = -295 (same |Delta_c~| as Fig. 8 and our model without the photon, same threshold);
      (2) u = -10, Delta_C = -300 and (3) u = -100, Delta_C = -300 (Nagy's Fig. 8).
Compared with: our eliminated model at N -> infinity (V0[omega_R] = -2 y^2 / |Delta_c~|) and the vertices of Fig. 8.
Result: unprojected reproduces Fig. 8 at both NU_0 (ratio 0.996 - 1.02 away from the peak); above threshold almost all of
that N' is the component of the modes along psi_0 itself (plateau ~ 0.05 also at U_0 = 0, ~ 0.04 without the photon).
Projected, the photon changes our N' by ~ 4e-4 atoms.

Usage (from the repository root):  python python/nagy_cavity_bogoliubov.py
Output: final_imag/nagy_cavity_bogoliubov.pdf, numpy_arrays_GP/nagy_cavity_bogoliubov.csv
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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

NMAX = 40
n = np.arange(-NMAX, NMAX + 1)
M = len(n)
T = np.diag(n.astype(float) ** 2)
COS = 0.5 * (np.eye(M, k=1) + np.eye(M, k=-1))
COS2 = 0.5 * np.eye(M) + 0.25 * (np.eye(M, k=2) + np.eye(M, k=-2))


def steady_state(y, u, DC):
    """(a, psi_0 coefficients, mu, h); a = 0, uniform psi_0 below threshold."""
    def ground(a):
        h = T + a ** 2 * u * COS2 + 2 * a * y * COS
        w, v = np.linalg.eigh(h)
        return w[0], v[:, 0] * np.sign(v[NMAX, 0]), h
    F = lambda a: a - y * (lambda c: c @ COS @ c)(ground(a)[1]) / (DC - u * (lambda c: c @ COS2 @ c)(ground(a)[1]))
    lo, hi = -2 * y / abs(DC) * 1.2, -1e-9
    a = brentq(F, lo, hi, xtol=1e-13, rtol=1e-13) if F(lo) * F(hi) < 0 else 0.0
    mu, c, h = ground(a)
    return a, c, mu, h


def depletion(y, u, DC, projected=True):
    """(N', n_ph, lowest omega, Theta, part of N' along psi_0)."""
    a, c, mu, h = steady_state(y, u, DC)
    A = -DC + u * (c @ COS2 @ c)
    G = (a * u * COS2 + y * COS) @ c
    H0 = h - mu * np.eye(M)
    if projected:
        Q = np.eye(M) - np.outer(c, c)
        G, H0 = Q @ G, Q @ H0 @ Q
    Mm = np.zeros((M + 1, M + 1))
    Mm[0, 0] = A
    Mm[0, 1:] = Mm[1:, 0] = G
    Mm[1:, 1:] = H0
    W = np.zeros((M + 1, M + 1))
    W[0, 1:] = W[1:, 0] = G
    ev, V = np.linalg.eig(np.block([[Mm, W], [-W, -Mm]]))
    ev, V = ev.real, V.real
    Nd, nph, om, par = 0.0, 0.0, np.inf, 0.0
    for k in np.where(ev > 1e-6)[0]:
        U_, V_ = V[:M + 1, k], V[M + 1:, k]
        nrm = U_ @ U_ - V_ @ V_
        if nrm > 1e-12:
            Nd += V_[1:] @ V_[1:] / nrm
            nph += V_[0] ** 2 / nrm
            par += (c @ V_[1:]) ** 2 / nrm
            om = min(om, ev[k])
    return Nd, nph, om, c @ COS @ c, par


def eliminated(y, DT, projected=True):
    """Our model without the photon, N -> infinity, same plane-wave basis (projected = python/bogoliubov_depletion.py)."""
    V0 = -2 * y ** 2 / DT
    f = lambda A: A - V0 * (lambda c: c @ COS @ c)(np.linalg.eigh(T + A * COS)[1][:, 0] * np.sign(np.linalg.eigh(T + A * COS)[1][NMAX, 0]))
    lo, hi = V0, -1e-9
    A = brentq(f, lo, hi, xtol=1e-13, rtol=1e-13) if f(lo) * f(hi) < 0 else 0.0
    h = T + A * COS
    w, v = np.linalg.eigh(h)
    c = v[:, 0] * np.sign(v[NMAX, 0])
    Q = np.eye(M) - np.outer(c, c)
    g = COS @ c
    X = V0 * np.outer(g, g)
    Lp, Xp = h - w[0] * np.eye(M) + X, X
    if projected:
        Lp, Xp = Q @ Lp @ Q, Q @ Xp @ Q
    ev, V = np.linalg.eig(np.block([[Lp, Xp], [-Xp, -Lp]]))
    ev, V = ev.real, V.real
    s = 0.0
    for k in np.where(ev > 1e-6)[0]:
        U_, V_ = V[:M, k], V[M:, k]
        nrm = U_ @ U_ - V_ @ V_
        if nrm > 1e-12:
            s += V_ @ V_ / nrm
    return s


RED = np.loadtxt(os.path.join("numpy_arrays_GP", "nagy2008_fig8_NU0m10.csv"), delimiter=",", skiprows=3)
GRN = np.loadtxt(os.path.join("numpy_arrays_GP", "nagy2008_fig8_NU0m100.csv"), delimiter=",", skiprows=3)


def fig8(x, d):
    i = d[:, 1].argmax()
    part = d[: i + 1] if x <= d[i, 0] else d[i:]
    return 10 ** np.interp(x, part[:, 0], np.log10(part[:, 1]))


print("Thresholds, Eq. (10):", [round(np.sqrt((DC - u / 2) ** 2 / (u - 2 * DC)), 3) for u, DC in ((0.0, -295.0), (-10.0, -300.0), (-100.0, -300.0))])
XS = [7, 9, 10, 11, 11.8, 13, 14, 15, 17, 20, 25, 30, 35, 40, 45, 50]
print("\n          no photon          | photon, u = 0               | photon, u = -10: Nagy's Fig. 8 red | photon, u = -100: Fig. 8 green")
print(" y     proj     unproj    | proj     unproj   along psi0 | proj     unproj   along psi0  Fig8  | proj     unproj   Fig8")
for x in XS:
    x = float(x)
    r1, r1u = depletion(x, 0.0, -295.0), depletion(x, 0.0, -295.0, False)
    r2, r2u = depletion(x, -10.0, -300.0), depletion(x, -10.0, -300.0, False)
    r3, r3u = depletion(x, -100.0, -300.0), depletion(x, -100.0, -300.0, False)
    print(f"{x:5.1f} {eliminated(x, 295.0):8.5f} {eliminated(x, 295.0, False):8.5f} | {r1[0]:8.5f} {r1u[0]:8.5f} {r1u[4]:8.5f}   |"
          f" {r2[0]:8.5f} {r2u[0]:8.5f} {r2u[4]:8.5f} {fig8(x, RED):7.4f} | {r3[0]:8.5f} {r3u[0]:8.5f} {fig8(x, GRN):7.4f}")

print("\nElimination test: U_0 = 0, projected, fixed V0, y = sqrt(|V0[omega_R]| D / 2), D = |Delta_c~| (values: photon kept; last row: eliminated)")
print("  (N'(D) - N'_el) * D should tend to a constant if the difference is a 1/D effect; lowest omega against D says when D >> omega fails")
for V_eps in (-15.0, -30.0, -60.0, -120.0, -335.0):
    v = abs(V_eps) / (2 * np.pi ** 2)
    el = eliminated(np.sqrt(v * 295.0 / 2), 295.0)
    print(f"  V0 = {V_eps:6.0f} eps, eliminated N' = {el:.6e}")
    for D in (295.0, 1e3, 3e3, 1e4, 3e4, 1e5):
        r = depletion(np.sqrt(v * D / 2), 0.0, -D)
        print(f"     D = {D:8.0f}:  N' = {r[0]:.6e}  ratio to eliminated {r[0] / el:7.4f}  (N'-N'_el) D = {(r[0] - el) * D:9.4f}  n_ph = {r[1]:.2e}  omega_min = {r[2]:7.2f}")

print("\nRatio unprojected / Fig. 8 at the vertices of Fig. 8, away from the peak (|y - y_c| > 0.5):")
for name, u, d in (("u=-10", -10.0, RED), ("u=-100", -100.0, GRN)):
    yc = np.sqrt((-300.0 - u / 2) ** 2 / (u + 600.0))
    q = np.array([depletion(a, u, -300.0, False)[0] / b for a, b in d if abs(a - yc) > 0.5])
    print(f"  {name}: min {q.min():.3f}, max {q.max():.3f}, mean {q.mean():.3f}, from {len(q)} vertices")

Y = np.unique(np.concatenate([np.linspace(2, 10, 17), np.linspace(10, 14, 81), np.linspace(14, 50, 73)]))
rows = []
for x in Y:
    r1, r1u = depletion(x, 0.0, -295.0), depletion(x, 0.0, -295.0, False)
    r2, r2u = depletion(x, -10.0, -300.0), depletion(x, -10.0, -300.0, False)
    r3, r3u = depletion(x, -100.0, -300.0), depletion(x, -100.0, -300.0, False)
    rows.append((x, eliminated(x, 295.0), eliminated(x, 295.0, False), r1[0], r1u[0], r2[0], r2u[0], r3[0], r3u[0]))
C = np.array(rows)
np.savetxt(os.path.join("numpy_arrays_GP", "nagy_cavity_bogoliubov.csv"), C, delimiter=",", comments="", fmt="%.8g",
           header="sqrtN_eta,no_photon_proj,no_photon_unproj,u0_proj,u0_unproj,u10_proj,u10_unproj,u100_proj,u100_unproj")

fig, ax = plt.subplots(figsize=(15, 11))
ax.semilogy(C[:, 0], C[:, 1], "-", color="k", lw=3, label=r"no photon, $U_0=0$, projected (our $N'$, $N\to\infty$)")
ax.semilogy(C[:, 0], C[:, 3], "--", color="C0", lw=2.5, label=r"photon, $U_0=0$, projected")
ax.semilogy(C[:, 0], C[:, 5], ":", color="C3", lw=3, label=r"photon, $NU_0=-10$, projected")
ax.semilogy(C[:, 0], C[:, 7], ":", color="C2", lw=3, label=r"photon, $NU_0=-100$, projected")
ax.semilogy(C[:, 0], C[:, 4], "-", color="C0", lw=1.5, label=r"photon, $U_0=0$, not projected")
ax.semilogy(C[:, 0], C[:, 6], "-", color="C3", lw=2.5, label=r"photon, $NU_0=-10$, not projected")
ax.semilogy(C[:, 0], C[:, 8], "-", color="C2", lw=2.5, label=r"photon, $NU_0=-100$, not projected")
ax.semilogy(RED[:, 0], RED[:, 1], "x", color="C3", ms=12, mew=2.5, label=r"Nagy et al.\ 2008, Fig.\ 8, $NU_0=-10$")
ax.semilogy(GRN[:, 0], GRN[:, 1], "+", color="C2", ms=14, mew=2.5, label=r"Nagy et al.\ 2008, Fig.\ 8, $NU_0=-100$")
ax.set_xlabel(r"$\sqrt{N}\eta\,[\omega_R]$")
ax.set_ylabel(r"$N'$")
ax.set_title(r"Bogoliubov depletion with the cavity mode kept, $\kappa=0$, $g=0$, $N\to\infty$", fontsize=28)
ax.set_xlim(2, 50)
ax.set_ylim(1e-4, 10)
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=22)
fig.tight_layout()
fig.savefig(os.path.join("final_imag", "nagy_cavity_bogoliubov.pdf"), bbox_inches="tight")
print("Saved final_imag/nagy_cavity_bogoliubov.pdf and numpy_arrays_GP/nagy_cavity_bogoliubov.csv")
