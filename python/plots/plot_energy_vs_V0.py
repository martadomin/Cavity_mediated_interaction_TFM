"""
Energy vs V0 at N = 40: GP, VMC, DMC, and the same figure vs the pump amplitude.

  top:    E/N (GP line, VMC and DMC points)
  bottom: total-energy differences E - E_GP (error bar = N sigma); symmetric-log y axis (linear within +-1),
          because VMC lies far above GP in the ordered phase (Jastrow trial function built from the two-body
          cavity problem, no GP one-body orbital) and would otherwise hide DMC.
Threshold line: N -> infinity value V0 = -2 pi^2 eps (matches Nagy's normalisation). The finite-N GP threshold
is 2 pi^2 N/(N-1) = 20.245 at N = 40, so E_GP = 0 down to V0 = -20.245, just past the line.

Conventions: stored energies are TOTAL (energy files: [E, err]). GP uses N(N-1)/2 pairs (a product state, so
variational): E_GP/N = kinetic + (V0/2)(N-1)/N Theta^2 (python/dmc_gp_delta.gp_energy).
DMC: longest imaginary time, then most walkers (tau >= 8), blocking errors of the energy series (filled tau = 32,
open tau = 8). VMC: reference run of the same DMC run. V0 = -200 is left out (unconverged, violates the bound),
V0 = -10 has only a tau = 3 run at N = 40.
Pump axis: sqrt(N) eta / (sqrt(N) eta)_c = sqrt(|V0| / 2 pi^2 eps); the unknown cavity prefactor cancels.
Third figure, Nagy's axis (2008, Fig. 8 parameters: kappa = 0, Delta_C = -300, N U_0 = -10, so |Delta_c~| = 295):
sqrt(N) eta = sqrt(|V0[omega_R]| |Delta_c~| / 2) in units of omega_R, V0[omega_R] = V0[eps] / 2 pi^2. The map covers
only the U_1 term; Nagy's U_2 light shift is not in our Hamiltonian.

Printed checks: E_DMC <= E_VMC, E_DMC <= E_GP within 2 sigma, and below threshold the Bogoliubov zero-point
energy E_Bog = [omega - e - V0 r/2]/2, omega = sqrt(e (e + V0 r)), e = 2 pi^2 eps, r = (N-1)/N.

Usage (from the repository root):  python python/plots/plot_energy_vs_V0.py
Output: final_imag/energy_vs_V0_N40.pdf, energy_vs_eta_N40.pdf, energy_vs_sqrtNeta_N40.pdf, numpy_arrays_GP/energy_vs_V0_N40.csv
"""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc_gp_delta import analyse, gp_energy, BASE  # noqa: E402
from pump_axis import add_pump_axis  # noqa: E402

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

N = 40
E_R = 2 * np.pi ** 2
V0S = [-15.0, -20.0, -21.0, -22.0, -23.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0]


def best_run(V):
    """(walkers, steps, dtau, tau) of the longest / largest DMC run with tau >= 8, or None."""
    v = V / N
    d = os.path.join(BASE, "energies", f"V0={v:.3f}")
    pat = re.compile(rf"^energies_(\d+)_([\d.]+)_{N}_(\d+)_{v:.3f}_{float(N):.1f}_IS\.npy$")
    best = None
    for f in (os.listdir(d) if os.path.isdir(d) else []):
        m = pat.match(f)
        if m:
            w, dt, st = int(m[1]), float(m[2]), int(m[3])
            key = (round(st * dt, 6), w)
            if key[0] >= 8 - 1e-9 and (best is None or key > best[0]):
                best = (key, w, st, dt)
    return None if best is None else (best[1], best[2], best[3], best[0][0])


rows = []
for V in V0S:
    r = best_run(V)
    if r is None:
        continue
    w, st, dt, tau = r
    (_, Ed, ed, Eg, _, _, _), = analyse(V, [N], w, st, dt)[0]
    v = V / N
    Ev, ev = np.load(os.path.join(BASE, "energy", f"V0={v:.3f}", f"energy_vmc_{N}_{v:.3f}_{float(N):.1f}.npy"))
    rows.append((V, Eg * N, Ed * N, ed * N, Ev, ev, w, tau))              # all total energies
R = np.array(rows)
np.savetxt(os.path.join("numpy_arrays_GP", f"energy_vs_V0_N{N}.csv"), R, delimiter=",", comments="", fmt="%.10g",
           header="V0,E_GP_total,E_DMC_total,E_DMC_err,E_VMC_total,E_VMC_err,DMC_walkers,DMC_tau")

print(f"N = {N}, total energies     sqrt(|V0|/2pi^2)   checks")
print("    V0    x      E_GP        E_VMC              E_DMC              DMC-GP           DMC<=VMC  DMC<=GP(2s)  [run]")
for V, Eg, Ed, ed, Ev, ev, w, tau in R:
    ok1, ok2 = Ed <= Ev + 2 * np.hypot(ed, ev), Ed <= Eg + 2 * ed
    print(f"  {V:5.0f}  {np.sqrt(-V / E_R):.3f}  {Eg:10.3f}  {Ev:10.3f}±{ev:.3f}  {Ed:10.3f}±{ed:.3f}  {Ed - Eg:+8.3f}±{ed:.3f}"
          f"     {'ok' if ok1 else 'NO':>3}       {'ok' if ok2 else 'NO':>3}     [{int(w)}w tau={tau:g}]")
r = (N - 1) / N
print("Below threshold, E_DMC vs Bogoliubov zero-point energy:")
for V, Eg, Ed, ed, *_ in R:
    if Eg == 0:
        om = np.sqrt(E_R * (E_R + V * r))
        print(f"  V0 = {V:.0f}: E_DMC = {Ed:.3f}±{ed:.3f}, E_Bog = {0.5 * (om - E_R - V * r / 2):.3f}")

Vd = np.linspace(-155, -0.5, 400)
Egd = np.array([gp_energy(v, N) for v in Vd])
conv, old = R[:, 7] >= 32 - 1e-9, R[:, 7] < 32 - 1e-9
VMC = dict(fmt="o", color="C1", ms=9, mfc="none", mew=2, capsize=4)
DMC = dict(fmt="s", color="C0", ms=10, capsize=4, mew=2)


def figure(fx, xlabel, thr, xlim, out, top_axis):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 16), height_ratios=[1.3, 1], sharex=True)
    ax.plot(fx(Vd), Egd, "k-", lw=2.5, label="GP")
    ax.errorbar(fx(R[:, 0]), R[:, 4] / N, yerr=R[:, 5] / N, label="VMC", **VMC)
    ax.errorbar(fx(R[conv, 0]), R[conv, 2] / N, yerr=R[conv, 3] / N, label=r"DMC ($\tau=32$)", **DMC)
    ax.errorbar(fx(R[old, 0]), R[old, 2] / N, yerr=R[old, 3] / N, label=r"DMC ($\tau=8$)", **{**DMC, "mfc": "none"})
    ax.set_ylabel(r"$E/N\,[\varepsilon]$")
    ax.legend(loc="best")
    ax2.axhline(0, color="k", lw=2.5)
    ax2.errorbar(fx(R[:, 0]), R[:, 4] - R[:, 1], yerr=R[:, 5], **VMC)
    ax2.errorbar(fx(R[conv, 0]), R[conv, 2] - R[conv, 1], yerr=R[conv, 3], **DMC)
    ax2.errorbar(fx(R[old, 0]), R[old, 2] - R[old, 1], yerr=R[old, 3], **{**DMC, "mfc": "none"})
    ax2.set_yscale("symlog", linthresh=1.0, linscale=1.5)
    ax2.set_ylabel(r"$E-E_{\rm GP}\,[\varepsilon]$")
    ax2.set_xlabel(xlabel)
    for a in (ax, ax2):
        a.axvline(thr, color="0.4", ls="--", lw=2)
        a.grid(True, ls="--", alpha=0.6)
        a.set_xlim(*xlim)
    if top_axis:
        add_pump_axis(ax, "eta")
    ax.set_title(fr"$N={N}$", fontsize=32, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join("final_imag", out), bbox_inches="tight")
    plt.close(fig)


os.makedirs("final_imag", exist_ok=True)
figure(lambda v: np.asarray(v), r"$V_0/\varepsilon$", -E_R, (-155, 0), f"energy_vs_V0_N{N}.pdf", True)
figure(lambda v: np.sqrt(np.abs(v) / E_R), r"$\sqrt{N}\eta/(\sqrt{N}\eta)_c=\sqrt{|V_0|/2\pi^2\varepsilon}$", 1.0,
       (0, 2.85), f"energy_vs_eta_N{N}.pdf", False)
# Nagy 2008 axis: sqrt(N) eta in units of omega_R, parameters of their Fig. 8 (kappa = 0, Delta_C = -300, N U_0 = -10)
DT = 295.0                                               # |Delta_C - N U_0 / 2| in units of omega_R
fnagy = lambda v: np.sqrt(np.abs(v) / E_R * DT / 2)
figure(fnagy, r"$\sqrt{N}\eta\,[\omega_R]$", np.sqrt(DT / 2), (0, 35), f"energy_vs_sqrtNeta_N{N}.pdf", False)
print(f"Nagy axis: threshold sqrt(N) eta_c = {np.sqrt(DT / 2):.2f} omega_R; V0 = -150 -> {fnagy(-150.0):.2f} omega_R")
print(f"Saved final_imag/energy_vs_V0_N{N}.pdf, final_imag/energy_vs_eta_N{N}.pdf, numpy_arrays_GP/energy_vs_V0_N{N}.csv")
