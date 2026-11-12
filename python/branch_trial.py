"""
Single-branch trial function for DMC (julia/dmc/run_dmc.jl --branch), written in the pair-product form the DMC
core already uses, so that no sampling code changes:

    Psi_T = prod_{i<j} f(x_i, x_j),
    f(x, y) = [phi_A(x) phi_A(y)]^(1/(N-1)) exp[ (beta/N) (cos 2 pi x - Theta) (cos 2 pi y - Theta) ].

The first factor multiplies out to prod_i phi_A(x_i), the GP branch (ground state of h[A], python/dmc_gp_delta._ground).
A bare product state is not enough: a_perp |Psi_T> = 0 makes the mixed rho^(1) exactly zero outside phi_A, so the mixed
estimator could not see depletion. The second factor is exp[beta/(2N) (S - N Theta)^2] up to O(1/N) one-body terms,
S = sum_i cos 2 pi x_i: a Gaussian squeezing of the collective mode. Its width relative to the product state is
    r_T = Var(S) / Var_GP(S),   1/r_T = 1 - 2 beta g2,   g2 = <cos^2>_phi - Theta^2,
and beta is set so that r_T = r_Bog = sum_k (g.(u_k + v_k))^2 / g2 from the BdG modes (same equations as
python/bogoliubov_depletion.py).

Usage (from the repository root):  python python/branch_trial.py --V0kac -23 -25 -30 -40 [--N 40]
Output: numpy_arrays_VMC/wavefunction_branch/V0=.../wavefunction_<V0>_<N>.npy (Ng x Ng on [-1/2, 1/2), as read by
periodic_wave_function), and the parameters printed.
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc_gp_delta import _ground, NMAX, EPSK  # noqa: E402

n = np.arange(-NMAX, NMAX + 1)
M = len(n)
COS = 0.5 * (np.eye(M, k=1) + np.eye(M, k=-1))
NG = 64                                       # spline grid per cell (f is much smoother than the 20 x 20 two-body grids)


def branch(V0, N):
    """(c, Theta, g2, r_Bog, N'_Bog) of the GP branch at (V0, N); c = plane-wave coefficients on n = -NMAX..NMAX."""
    R = (N - 1) / N
    f = lambda A: A - V0 * R * _ground(A)[1]
    lo, hi = V0 * R, -1e-8
    if f(lo) * f(hi) > 0:
        raise ValueError(f"V0 = {V0}: uniform GP state, no branch to select")
    A = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14)
    h = np.diag(EPSK * n.astype(float) ** 2) + A * COS
    w, v = np.linalg.eigh(h)
    c = v[:, 0] * np.sign(v[NMAX, 0])
    Q = np.eye(M) - np.outer(c, c)
    g = COS @ c
    X = V0 * R * np.outer(g, g)
    Lp = Q @ (h - w[0] * np.eye(M) + X) @ Q
    Xp = Q @ X @ Q
    ev, V = np.linalg.eig(np.block([[Lp, Xp], [-Xp, -Lp]]))
    ev, V = ev.real, V.real
    Qg = Q @ g
    s, Nd = 0.0, 0.0
    for k in np.where(ev > 1e-7)[0]:
        u, vv = V[:M, k], V[M:, k]
        nrm = u @ u - vv @ vv
        if nrm > 1e-12:
            u, vv = u / np.sqrt(nrm), vv / np.sqrt(nrm)
            s += (Qg @ (u + vv)) ** 2
            Nd += vv @ vv
    g2 = Qg @ Qg
    return c, c @ g, g2, s / g2, Nd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=[-23.0, -25.0, -30.0, -40.0])
    p.add_argument("--N", type=int, default=40)
    args = p.parse_args()
    N = args.N
    x = -0.5 + np.arange(NG) / NG
    print("   V0      Theta    g2      r_Bog    beta     N'_Bog    s = <phi_A|phi_B>   s^N")
    for Vk in args.V0kac:
        c, th, g2, r, Nd = branch(Vk, N)
        beta = (1 - 1 / r) / (2 * g2)
        phi = np.real(np.exp(2j * np.pi * np.outer(x, n)) @ c)
        if phi.min() <= 0:
            raise ValueError(f"V0 = {Vk}: phi_A is not positive on the grid (min {phi.min():.2e})")
        cs = np.cos(2 * np.pi * x)
        logf1 = np.log(phi) / (N - 1)
        logf = logf1[:, None] + logf1[None, :] + beta / N * np.outer(cs - th, cs - th)
        v = Vk / N
        d = os.path.join("numpy_arrays_VMC", "wavefunction_branch", f"V0={v:.3f}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"wavefunction_{v:.3f}_{N}.npy"), np.exp(logf))
        s = c @ (c * (-1.0) ** n)                  # phi_B(x) = phi_A(x + 1/2): c_n -> (-1)^n c_n
        print(f"  {Vk:6.1f}  {th:.4f}  {g2:.4f}  {r:7.4f}  {beta:7.4f}  {Nd:.4f}    {s:.4f}   {s ** N:.2e}")


if __name__ == "__main__":
    main()
