"""
Approach to the threshold from below: our exact one-mode N' (N -> infinity), Nagy's asymptotic Eq. (25) and the vertices of
their Fig. 8 (NU_0 = -10), against the distance from threshold d = 1 - sqrt(N) eta / (sqrt(N) eta)_c, log-log axes. All of
them diverge at d = 0, so a factor of 2 shows up as two parallel lines.
    ours:     N' = (A / omega - 1) / 2,  A = omega_R (1 - s / 2),  omega = lambda_1 = omega_R sqrt(1 - s),  s = (eta / eta_c)^2
    Eq. (25): N' = omega_R / (8 lambda_1) = (1/8) [omega_R delta_C / (8 N (eta_c - eta)^2)]^(1/4),  delta_C = 295 omega_R

Usage (from the repository root):  python python/plots/plot_nagy_eq25.py
Output: final_imag/nagy_eq25_threshold.pdf
"""

import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 25,
    "text.usetex": True,
})

DC = 295.0
YC = np.sqrt(DC / 2)
d = np.logspace(-6, np.log10(0.5), 400)
s = (1 - d) ** 2
lam = np.sqrt(1 - s)
ours = 0.5 * ((1 - s / 2) / lam - 1)
eq25_lam = 1 / (8 * lam)
eq25_eta = (DC / (8 * (YC * d) ** 2)) ** 0.25 / 8

F8 = np.loadtxt(os.path.join("numpy_arrays_GP", "nagy2008_fig8_NU0m10.csv"), delimiter=",", skiprows=3)
F8 = F8[F8[:, 0] < 12.1]                      # the vertices from 12.13 on lie in the numerically rounded peak of Fig. 8

for dd in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    i = np.argmin(np.abs(d - dd))
    print(f"d = {d[i]:.1e}: ours = {ours[i]:9.3f}, Eq. 25 = {eq25_lam[i]:9.3f} / {eq25_eta[i]:9.3f}, ours / Eq. 25 = {ours[i] / eq25_lam[i]:.3f}")

y = YC * (1 - d)
yf, lf = F8[:, 0], np.sqrt(1 - (F8[:, 0] / YC) ** 2)
fig, (ax, axp) = plt.subplots(1, 2, figsize=(28, 11))

# left: N' itself; 2 x Eq. (25) lands on our curve near threshold
ax.semilogy(y, ours, "-", color="C0", lw=4, label=r"ours: $\frac12(A/\lambda_1-1)$")
ax.semilogy(y, eq25_lam, "--", color="C3", lw=3.5, label=r"Nagy Eq.\ (25): $\omega_R/8\lambda_1$")
ax.semilogy(y, 2 * eq25_lam, ":", color="C2", lw=4, label=r"$2\times$ Eq.\ (25): $\omega_R/4\lambda_1$")
ax.semilogy(yf, F8[:, 1], "kx", ms=14, mew=3, label=r"Nagy Fig.\ 8, $NU_0=-10$")
ax.axvline(YC, color="0.4", ls="--", lw=2)
ax.set_xlabel(r"$\sqrt{N}\eta\,[\omega_R]$")
ax.set_ylabel(r"$N'$")
ax.set_xlim(10.5, 12.3)
ax.set_ylim(0.1, 100)
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(loc="upper left")

# right: N' lambda_1 / omega_R removes the divergence, so the prefactor can be read off directly
axp.semilogx(d, ours * lam, "-", color="C0", lw=4, label=r"ours $\to 1/4$")
axp.semilogx(d, eq25_lam * lam, "--", color="C3", lw=3.5, label=r"Nagy Eq.\ (25) $=1/8$")
axp.semilogx(1 - yf / YC, F8[:, 1] * lf, "kx", ms=14, mew=3, label=r"Nagy Fig.\ 8, $NU_0=-10$")
axp.axhline(0.25, color="C0", ls=":", lw=2)
axp.set_xlabel(r"$1-\sqrt{N}\eta/\sqrt{N}\eta_c$")
axp.set_ylabel(r"$N'\lambda_1/\omega_R$")
axp.set_xlim(1e-6, 0.5)
axp.set_ylim(0, 0.3)
axp.grid(True, which="both", ls="--", alpha=0.4)
axp.legend(loc="upper right")
print("Fig. 8 vertices: 1 - y/y_c, N' lambda_1 / omega_R (ours -> 1/4, Eq. 25 = 1/8):")
for yy, nn, ll in zip(yf, F8[:, 1], lf):
    print(f"  {1 - yy / YC:.2e}  {nn * ll:.4f}")
fig.tight_layout()
os.makedirs("final_imag", exist_ok=True)
fig.savefig(os.path.join("final_imag", "nagy_eq25_threshold.pdf"), bbox_inches="tight")
print("Saved final_imag/nagy_eq25_threshold.pdf")
