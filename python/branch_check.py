"""
Sanity checks of the single-branch DMC runs (julia/dmc/run_dmc.jl --branch, N = 40 by default).

Away from threshold one branch should give lambda_0 = N - N' and tiny lambda_1, lambda_2, ...; a large lambda_1 means
that the walkers mixed branches (tunnelling). Reported for VMC, DMC mixed and 2 DMC - VMC:
  * lambda_0..3 of rho^(1) (symmetrised, translations by lambda only, trace rescaled to N) against the noise floor
    |lambda_min|: the top eigenvalues of a noisy matrix are biased upwards by about that much;
  * N - lambda_0, compared with the Bogoliubov N' (precision limited by the trace normalisation, ~0.1-0.2 atoms);
  * n_perp = <phi_perp|rho^(1)|phi_perp>, phi_perp = Q cos(2 pi x) phi_A normalised: the BdG collective mode, compared
    with N'. Linear in rho^(1), so not biased by noise; bar = rms occupation of neighbouring uncoupled plane waves.
Independent test of tunnelling, from the final walkers: fraction with Theta < 0 and <Theta> against Theta_GP.

Usage (from the repository root):  python python/branch_check.py --V0kac -25 [--N 40] [--walkers 400 --tau 2]
(without --walkers/--tau: longest, then largest, run of each coupling)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branch_trial import branch, n as NS, COS  # noqa: E402
from obdm_analysis import symmetrize  # noqa: E402

BASE = os.path.join("numpy_arrays_DMC_branch", "constant_density_1.0")


def find_run(V, N, walkers, tau):
    """Run name (walkers_dtau_N_steps_V0_L_IS) of the longest / largest run with these filters, or None."""
    v = V / N
    best = None
    for f in glob.glob(os.path.join(BASE, "obdm", f"V0={v:.3f}", f"obdm_*_{N}_*_IS.npy")):
        m = re.match(r"obdm_(\d+)_([\d.]+)_\d+_(\d+)_", os.path.basename(f))
        if m is None:
            continue
        w, t = int(m[1]), float(m[2]) * int(m[3])
        if (walkers and w != walkers) or (tau and not np.isclose(t, tau)):
            continue
        if best is None or (t, w) > best[0]:
            best = ((t, w), os.path.basename(f)[len("obdm_"):-len(".npy")])
    return best[1] if best else None


def report(label, M, L, N, phi_A, phi_perp, Nd):
    M = symmetrize(M, L)
    nb = M.shape[0]
    dx = L / nb
    tr = np.trace(M) * dx
    # 2 x 2 block on span{phi_A, phi_perp}, no trace rescaling (the diagonal is the noisiest part of the matrix): its
    # smaller eigenvalue is the depletion in this mode whatever small rotation the condensate has relative to GP
    aa, ap, pp = (phi_A @ M @ phi_A) * dx * dx, (phi_A @ M @ phi_perp) * dx * dx, (phi_perp @ M @ phi_perp) * dx * dx
    block = np.linalg.eigvalsh(np.array([[aa, ap], [ap, pp]]))
    print(f"  {label:9s} block: <A|rho|A> {aa:7.3f}  <A|rho|perp> {ap:+.3f}  <perp|rho|perp> {pp:+.4f} -> "
          f"depletion {block[0]:+.4f}, condensate {block[1]:.3f} (unscaled), rotation {np.arctan2(2 * ap, aa - pp) / 2:+.4f}")
    M = M * (N / tr)
    w = np.linalg.eigvalsh(M * dx)[::-1]
    x = -L / 2 + (np.arange(nb) + 0.5) * dx
    occ = lambda g: (g @ M @ g) * dx * dx
    bg = [occ(np.sqrt(2 / L) * f(2 * np.pi * m * x / L)) for m in range(30, 51) for f in (np.cos, np.sin)
          if not (m == round(L) and f is np.cos)]
    print(f"  {label:9s} trace {tr:7.3f} | l0..3 {w[0]:7.3f} {w[1]:6.3f} {w[2]:6.3f} {w[3]:6.3f} | noise {-w[-1]:.3f} | "
          f"N-l0 {N - w[0]:+7.3f} | n_perp {occ(phi_perp):+.4f} +- {np.sqrt(np.mean(np.square(bg))):.4f} | N' {Nd:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=[-23.0, -25.0, -30.0, -40.0])
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--walkers", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.0)
    args = p.parse_args()
    N, L = args.N, float(args.N)
    for V in args.V0kac:
        run = find_run(V, N, args.walkers, args.tau)
        if run is None:
            print(f"V0 = {V:.0f}: no run")
            continue
        v = V / N
        vd = f"V0={v:.3f}"
        c, th, g2, r, Nd = branch(V, N)
        Qg = COS @ c - th * c
        Qg /= np.linalg.norm(Qg)
        nb_x = lambda x: np.real(np.exp(2j * np.pi * np.outer(x, NS)) @ Qg) / np.sqrt(L)
        d = np.load(os.path.join(BASE, "obdm", vd, f"obdm_{run}.npy"))
        vm = np.load(os.path.join(BASE, "obdm", vd, f"obdm_vmc_{N}_{v:.3f}_{L:.1f}.npy"))
        nb = d.shape[0]
        phi_perp = nb_x(-L / 2 + (np.arange(nb) + 0.5) * L / nb)

        e = np.load(os.path.join(BASE, "energy", vd, f"energy_{run}.npy"))
        ev = np.load(os.path.join(BASE, "energy", vd, f"energy_vmc_{N}_{v:.3f}_{L:.1f}.npy"))
        print(f"V0 = {V:.0f}, run {run}: E_VMC = {ev[0]:.3f} +- {ev[1]:.3f}, E_DMC = {e[0]:.3f} +- {e[1]:.3f}")
        X = np.load(os.path.join(BASE, "walkers", vd, f"walkers_{run}.npy"))
        T = np.cos(2 * np.pi * X).mean(axis=1)
        print(f"  walkers: {len(T)}, fraction Theta < 0 = {(T < 0).mean():.4f}, <Theta> = {T.mean():.4f}, "
              f"std = {T.std():.4f}, Theta_GP = {th:.4f}, GP std = {np.sqrt(g2 / N):.4f}")
        phi_A = np.real(np.exp(2j * np.pi * np.outer(-L / 2 + (np.arange(nb) + 0.5) * L / nb, NS)) @ c) / np.sqrt(L)
        report("VMC", vm, L, N, phi_A, phi_perp, Nd)
        report("DMC mixed", d, L, N, phi_A, phi_perp, Nd)
        report("2DMC-VMC", 2 * d - vm, L, N, phi_A, phi_perp, Nd)


if __name__ == "__main__":
    main()
