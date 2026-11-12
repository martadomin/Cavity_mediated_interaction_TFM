"""
N - lambda_0 from the existing N = 40 DMC runs near threshold (numpy_arrays_DMC, not the single-branch runs), for the VMC,
mixed DMC and 2 DMC - VMC matrices. No new runs.

The top eigenvalue of the full 1000 x 1000 matrix is biased upwards by noise, so rho^(1) (symmetrised, translations by lambda)
is projected on its exact symmetry sectors: lambda-periodic plane waves (the interaction only changes momenta by 2 pi), split
into even and odd n under x -> x + 1/2 (the cat is even). Small blocks, so the noise bias is negligible.

No trace rescaling: the estimator is normalised to N on average, and the trace is the one quantity it measures badly, as it comes
from the diagonal bins only (tr - N wanders by up to 1 atom, and VMC at -21 has lambda_0 > tr). Reported:
  N-l0   N - lambda_0, lambda_0 = top eigenvalue of the even block, unscaled     <- the result
  scaled N (1 - lambda_0 / tr), the same with the usual rescaling, for comparison
  l1     top eigenvalue of the odd block, unscaled: independent of lambda_0, and N - lambda_0 = lambda_1 + (N'_out <= 2e-3)
  err    max(|N - lambda_0 - l1|, bar): closure of that sum rule, or noise
  low-q  summed occupation of the plane waves |m| = 1..5 (m / L, not lambda-periodic): zero in the ground state, so anything above
         noise (bar sqrt(10)) would be a correlation of the trial wave function not yet projected out
  bar    rms occupation of uncoupled plane waves around 2 pi: noise of one mode occupation

Estimator calibration in the two-mode model (no QMC): a trial psi_T(n_A) is tuned to the measured VMC N - lambda_0, then
    mixed = <T|n_-|0> / <T|0>,  extrapolated = 2 mixed - <T|n_-|T>,  exact = <0|n_-|0>,  n_- = a_-^dag a_- = N/2 - (a_A^dag a_B + h.c.)/2
with |0> the H2 ground state. Two trials, to see how much it depends on the form:
    "H2"   ground state of H2 with a weaker interaction V_T;
    "Jas"  sqrt(binomial(N, n_A)) exp[beta (2 n_A - N)^2 / N]: phi_+ product times a Jastrow of Sigma cos, as the QMC trial.

Usage (from the repository root):  python python/critical_window_dmc.py
Output: numpy_arrays_OBDM/critical_window_dmc_N40.csv (all runs), numpy_arrays_OBDM/critical_window_calibration_N40.csv
"""

import glob
import os
import re
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from critical_window_two_mode import bogoliubov, two_mode, two_mode_state, Vc  # noqa: E402
from obdm_analysis import symmetrize  # noqa: E402

N, L = 40, 40.0
BASE = os.path.join("numpy_arrays_DMC", "constant_density_1.0")
NMAX = 8
V0S = (-10.0, -15.0, -20.0, -21.0, -22.0, -23.0, -25.0, -30.0)


def runs(V):
    """(tau, walkers, dtau, file) of the N = 40 runs at V, sorted by tau."""
    out = []
    for f in glob.glob(os.path.join(BASE, "obdm", f"V0={V / N:.3f}", f"obdm_*_{N}_*_IS.npy")):
        m = re.match(r"obdm_(\d+)_([\d.]+)_\d+_(\d+)_", os.path.basename(f))
        if m:
            out.append((float(m[2]) * int(m[3]), int(m[1]), float(m[2]), f))
    return sorted(out)


def measure(M):
    M = symmetrize(M, L)
    nb = M.shape[0]
    dx = L / nb
    x = (np.arange(nb) + 0.5) * dx
    tr = np.trace(M) * dx

    def block(ns):
        B = np.column_stack([np.sqrt(1 / L) * np.exp(2j * np.pi * k * x) for k in ns])
        return np.linalg.eigvalsh((B.conj().T @ M @ B).real * dx * dx)

    ev = block([k for k in range(-NMAX, NMAX + 1) if k % 2 == 0])
    od = block([k for k in range(-NMAX, NMAX + 1) if k % 2 != 0])
    occ = lambda g: (g @ M @ g) * dx * dx * 2 / L
    lowq = sum(occ(f(2 * np.pi * m * x / L)) for m in range(1, 6) for f in (np.cos, np.sin))
    bar = np.sqrt(np.mean([occ(f(2 * np.pi * m * x / L)) ** 2 for m in range(30, 51) for f in (np.cos, np.sin)
                           if not (m == 40 and f is np.cos)]))
    d = N - ev[-1]
    return dict(tr=tr, d=d, scaled=N * (1 - ev[-1] / tr), l1=od[-1], lowq=lowq, bar=bar, err=max(abs(d - od[-1]), bar), second=ev[-2])


def calibrate(V, target):
    """Exact, and (trial, mixed, extrapolated) of each trial form tuned to N - lambda_0 = target."""
    psi, hop, *_ = two_mode_state(V, N)
    nm = lambda a, b: (N / 2 * (a @ b) - 0.5 * (a[:-1] @ (hop * b[1:]) + a[1:] @ (hop * b[:-1])))
    k = np.arange(N + 1)
    logb = 0.5 * (gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))

    def jas(beta):
        t = np.exp(logb + beta * (2 * k - N) ** 2 / N)
        return t / np.linalg.norm(t)

    forms = {"H2": lambda u: two_mode_state(V, N, V_int=u)[0], "Jas": jas}
    bounds = {"H2": (V * 1e-4, V), "Jas": (0.0, 2.0)}
    out = {"exact": nm(psi, psi)}
    for name, form in forms.items():
        F = lambda u: nm(form(u), form(u)) - target
        lo, hi = bounds[name]
        if F(lo) * F(hi) > 0:
            out[name] = (np.nan,) * 3
            continue
        T = form(brentq(F, lo, hi, xtol=1e-8))
        tt, mix = nm(T, T), nm(T, psi) / (T @ psi)
        out[name] = (tt, mix, 2 * mix - tt)
    return out


def main():
    rows = []
    print("  V0  walkers tau  dtau  est.  |  N-l0   scaled |    l1     err    bar   |  tr-N  | low-q | E")
    for V in V0S:
        v = V / N
        vm = np.load(os.path.join(BASE, "obdm", f"V0={v:.3f}", f"obdm_vmc_{N}_{v:.3f}_{L:.1f}.npy"))
        evm = np.load(os.path.join(BASE, "energy", f"V0={v:.3f}", f"energy_vmc_{N}_{v:.3f}_{L:.1f}.npy"))
        for tau, w, dt, f in runs(V):
            d = np.load(f)
            e = np.load(f.replace(os.sep + "obdm" + os.sep, os.sep + "energy" + os.sep).replace("obdm_", "energy_"))
            for name, M, code in (("VMC", vm, 0), ("mixed", d, 1), ("2D-V", 2 * d - vm, 2)):
                r = measure(M)
                en = {0: f"{evm[0]:.3f}+-{evm[1]:.3f}", 1: f"{e[0]:.3f}+-{e[1]:.3f}", 2: ""}[code]
                print(f"{V:5.0f} {w:5d} {tau:4.0f} {dt:.4f} {name:6s}| {r['d']:6.3f} {r['scaled']:7.3f} | {r['l1']:6.3f} {r['err']:6.3f} {r['bar']:6.3f} | "
                      f"{r['tr'] - N:+6.3f} | {r['lowq']:+.3f} | {en}")
                rows.append((V, w, tau, dt, code, r['d'], r['scaled'], r['l1'], r['err'], r['bar'], r['tr'], r['lowq'], r['second']))
    A = np.array(rows)
    np.savetxt(os.path.join("numpy_arrays_OBDM", "critical_window_dmc_N40.csv"), A, delimiter=",", comments="", fmt="%.8g",
               header="V0,walkers,tau,dtau,estimator_0VMC_1mixed_2extrap,N_minus_l0,N_minus_l0_trace_rescaled,lambda1_odd,err,noise_bar,trace,lowq_occupation,second_even")

    print("\nBest run of each V0 (longest tau, then most walkers) against theory, and two-mode calibration of the estimators")
    print("   V0   V0/Vc   Bog N'  | 2-mode exact | DMC VMC  mixed  extrap | model H2: mixed extrap | model Jas: mixed extrap")
    cal = []
    for V in V0S:
        S = A[A[:, 0] == V]
        best = S[(S[:, 2] == S[:, 2].max())]
        best = best[best[:, 1] == best[:, 1].max()]
        vd, md, xd = [best[best[:, 4] == c][0] for c in (0, 1, 2)]
        c = calibrate(V, vd[5])
        nb = bogoliubov(V, N)[0] if -V < Vc(N) else np.nan
        print(f"  {V:5.0f}  {-V / Vc(N):.3f}  {nb:6.3f}  | {c['exact']:8.3f}     | {vd[5]:6.3f} {md[5]:6.3f} {xd[5]:6.3f} | "
              f"{c['H2'][1]:8.3f} {c['H2'][2]:6.3f}    | {c['Jas'][1]:8.3f} {c['Jas'][2]:6.3f}")
        cal.append((V, nb, c['exact'], vd[5], vd[8], md[5], md[8], xd[5], xd[8], md[1], md[2], *c['H2'][1:], *c['Jas'][1:]))
    np.savetxt(os.path.join("numpy_arrays_OBDM", "critical_window_calibration_N40.csv"), np.array(cal), delimiter=",", comments="", fmt="%.8g",
               header="V0,Bogoliubov,two_mode_exact,VMC,VMC_err,mixed,mixed_err,extrap,extrap_err,walkers,tau,model_H2_mixed,model_H2_extrap,model_Jas_mixed,model_Jas_extrap")

    print("\nSensitivity of the raw normalisation deep in the phase: N - l0 - l1, unscaled vs trace-rescaled (mixed, longest run)")
    for V in (-30.0, -40.0, -60.0, -100.0, -200.0):
        R = [r for r in runs(V)]
        if not R:
            continue
        M = symmetrize(np.load(R[-1][3]), L)
        dx = L / M.shape[0]
        w = np.linalg.eigvalsh(M * dx)[::-1]
        tr = np.trace(M) * dx
        print(f"  {V:6.0f}: tau {R[-1][0]:.0f}, tr-N {tr - N:+.3f}; N-l0-l1 unscaled {N - w[0] - w[1]:+.3f}, rescaled {N - (w[0] + w[1]) * N / tr:+.3f}, noise |l_min| {-w[-1]:.3f}")


if __name__ == "__main__":
    main()
