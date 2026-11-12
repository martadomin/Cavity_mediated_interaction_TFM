"""
Momentum-space one-body density matrix of the VMC / DMC runs (Task B of gp_obdm_brief.md).

For every (estimator, N, V0_kac), estimator = VMC, DMC (mixed), lin (2 DMC - VMC):
  1. load rho(x,x'), symmetrize, average over translations by lambda (exact symmetry of H),
     rescale the trace to N (the stored matrices have a 0.3-2 % deficit from bin smearing;
     the factor is reported). Report the most negative eigenvalue of rho (noise level).
  2. rho(k_j, k_j') = dx^2/L sum_{x,x'} exp(-i k_j x) rho(x,x') exp(+i k_j' x'),  k_j = 2 pi j / L.
     The trace of the momentum-space diagonal is checked to still be N.
  3. n(k) = rho(k,k); peaks at k = n kL, i.e. j = n N.
  4. Condensate block rho_nn' = rho(n kL, n' kL), |n|,|n'| <= nmax; diagonalise, keep the two
     leading eigenvectors d^(0), d^(1); depletion outside the block = N - tr(rho_nn').
  5. Parity projection: phi_+ (even n) and phi_- (odd n) components of the leading
     eigenvectors, each normalised. Overlaps with the GP orbitals (gp_scf.jl output) are
     |sum_n c_n^pm d_n^pm|^2.

Usage (from the repository root):
    python python/obdm_momentum.py                          # N = 40, default V0_kac grid
    python python/obdm_momentum.py --N 30 40 --V0kac -40 -100
Output: numpy_arrays_OBDM/momentum_N<N>_V0kac<V0kac>.npz and numpy_arrays_OBDM/summary_N<N>.csv
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obdm_analysis import load_raw, symmetrize   # same loaders / symmetrisation as the OBDM analysis

V0_GRID = [-15., -20., -25., -30., -35., -40., -50., -60., -80., -100., -150., -200.]
ESTIMATORS = ["VMC", "DMC", "lin"]


def load_gp(N, V0kac):
    """GP orbital c_n (n = -30..30) from julia/gp/gp_scf.jl, or (None, None)."""
    f = os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv")
    if not os.path.isfile(f):
        return None, None
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    row = d[np.isclose(d[:, 0], V0kac)]
    return (row[0, 1:], np.arange(-30, 31)) if len(row) else (None, None)


def estimator_matrix(base, N, V0kac, est):
    v, L = load_raw(base, N, V0kac, "VMC")
    if est == "VMC":
        return v, L
    d, _ = load_raw(base, N, V0kac, "DMC")
    if d is None or v is None:
        return None, L
    return (d if est == "DMC" else 2 * d - v), L


def momentum_transform(rho, L, js):
    """rho(k_j, k_j') for the integer momenta js (k_j = 2 pi j / L), bin-centre positions."""
    nb = rho.shape[0]
    dx = L / nb
    x = -L / 2 + (np.arange(nb) + 0.5) * dx
    F = np.exp(-1j * 2 * np.pi / L * np.outer(js, x))          # F[j, x] = exp(-i k_j x)
    return (dx ** 2 / L) * F @ rho @ F.conj().T


def analyse(rho, N, L, nmax, gp):
    nb = rho.shape[0]
    dx = L / nb
    out = {}
    rho = symmetrize(rho, L)
    tr = np.trace(rho) * dx
    rho = rho * (N / tr)                                        # trace -> N
    out["trace_factor"] = N / tr
    w = np.linalg.eigvalsh(rho * dx)
    out["min_eig_over_N"] = w[0] / N                            # < 0 only through noise

    # full momentum distribution on j = -nb/2 .. nb/2-1 (diagonal only)
    nj = np.arange(-nb // 2, nb // 2)
    x = -L / 2 + (np.arange(nb) + 0.5) * dx
    F = np.exp(-1j * 2 * np.pi / L * np.outer(nj, x))
    nk = ((dx ** 2 / L) * np.sum((F @ rho) * F.conj(), axis=1)).real
    out["j"], out["nk"] = nj, nk
    out["trace_k"] = nk.sum()                                   # = N up to the cutoff |j| < nb/2

    # condensate block, n kL <-> j = n N (L = N lambda)
    ns = np.arange(-nmax, nmax + 1)
    R = momentum_transform(rho, L, ns * int(round(L)))
    out["ns"], out["block"] = ns, R
    out["block_diag"] = np.diag(R).real                         # n(n kL)
    out["depletion"] = N - np.trace(R).real
    w, V = np.linalg.eigh(R)
    w, V = w[::-1], V[:, ::-1]
    out["block_eigs_over_N"] = w[:2] / N
    d = []
    for i in range(2):
        v = V[:, i]
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))  # global phase: largest component real > 0
        d.append(v)
    out["imag_frac"] = max(np.linalg.norm(v.imag) / np.linalg.norm(v) for v in d)
    d = [v.real for v in d]
    even = (ns % 2 == 0)

    def proj(v, mask):
        p = np.where(mask, v, 0.0)
        nrm = np.linalg.norm(p)
        return (p / nrm if nrm > 0 else p), nrm ** 2
    pp = [proj(v, even) for v in d]
    pm = [proj(v, ~even) for v in d]
    out["d"] = np.array(d)
    out["even_weight"] = np.array([a[1] for a in pp])
    out["odd_weight"] = np.array([a[1] for a in pm])
    # sector orbitals: the leading eigenvector with the largest weight in each parity sector
    out["phi_plus"] = pp[int(np.argmax(out["even_weight"]))][0]
    out["phi_minus"] = pm[int(np.argmax(out["odd_weight"]))][0]

    if gp[0] is not None:
        c = np.zeros(len(ns))
        sel = np.abs(gp[1]) <= nmax
        c[np.isin(ns, gp[1][sel])] = gp[0][sel]
        out["gp_c"] = c
        for tag, mask in (("plus", even), ("minus", ~even)):
            cs = np.where(mask, c, 0.0)
            nrm = np.linalg.norm(cs)
            cs = cs / nrm if nrm > 0 else cs
            out[f"overlap_{tag}"] = float(np.dot(cs, out[f"phi_{tag}"]) ** 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, nargs="+", default=[40])
    p.add_argument("--V0kac", type=float, nargs="+", default=V0_GRID)
    p.add_argument("--nmax", type=int, default=8)
    p.add_argument("--density", type=float, default=1.0)
    args = p.parse_args()

    out_dir = "numpy_arrays_OBDM"
    os.makedirs(out_dir, exist_ok=True)
    base = f"numpy_arrays_DMC/constant_density_{args.density:.1f}"
    for N in args.N:
        rows = []
        print(f"\nN = {N}")
        print(f"{'V0':>7} {'est':>4} {'trfac':>6} {'minEig/N':>9} {'tr_k/N':>7} {'lam0/N':>7} {'lam1/N':>7} "
              f"{'deplet/N':>8} {'imag':>6} {'ov+':>6} {'ov-':>6}")
        for V in args.V0kac:
            res = {}
            for est in ESTIMATORS:
                rho, L = estimator_matrix(base, N, V, est)
                if rho is None:
                    continue
                r = analyse(rho, N, L, args.nmax, load_gp(N, V))
                res[est] = r
                print(f"{V:7.1f} {est:>4} {r['trace_factor']:6.3f} {r['min_eig_over_N']:9.4f} "
                      f"{r['trace_k'] / N:7.4f} {r['block_eigs_over_N'][0]:7.4f} {r['block_eigs_over_N'][1]:7.4f} "
                      f"{r['depletion'] / N:8.4f} {r['imag_frac']:6.3f} "
                      f"{r.get('overlap_plus', np.nan):6.3f} {r.get('overlap_minus', np.nan):6.3f}")
                rows.append([V, ESTIMATORS.index(est), r["trace_factor"], r["min_eig_over_N"], r["trace_k"] / N,
                             *r["block_eigs_over_N"], r["depletion"] / N, r["imag_frac"],
                             r.get("overlap_plus", np.nan), r.get("overlap_minus", np.nan)])
            if res:
                flat = {f"{e}_{k}": v for e, r in res.items() for k, v in r.items()}
                np.savez(os.path.join(out_dir, f"momentum_N{N}_V0kac{V:.0f}.npz"), **flat)
        if rows:
            np.savetxt(os.path.join(out_dir, f"summary_N{N}.csv"), np.array(rows), delimiter=",", comments="",
                       header="V0_kac,estimator(0=VMC,1=DMC,2=lin),trace_factor,min_eig_over_N,trace_k_over_N,"
                              "lambda0_over_N,lambda1_over_N,depletion_over_N,imag_frac,overlap_plus,overlap_minus")


if __name__ == "__main__":
    main()
