"""
Z2-symmetric trial function that can describe the cat, in the pair-product form the DMC core already uses:

    Psi_T = prod_{i<j} f(x_i, x_j),   f(x, y) = [u(x) u(y)]^(1/(N-1)) exp[(beta/N) g(x) g(y)]
          = prod_i u(x_i) exp[beta/(2N) (S^2 - sum_i g_i^2)],   S = sum_i g(x_i)
    ln u = gamma2 cos 4 pi x + gamma4 cos 8 pi x          (period 1/2)
    g    = cos 2 pi x + d3 cos 6 pi x + d5 cos 10 pi x    (g(x + 1/2) = -g(x), so S -> -S)

Psi_T is even under x -> x + 1/2; for large beta |Psi_T|^2 has two peaks in S (the two branches), for small beta one.

Everything is exact, with no Monte Carlo, because exp[(beta/N) S^2] is a Gaussian integral over a field h that decouples the atoms:
    |Psi_T|^2 = int dh exp(-N h^2 / (4 beta)) prod_i w_h(x_i),   w_h = u^2 exp(-beta g^2 / N + h g)
so, with P(h) ~ exp(-N h^2 / (4 beta)) Z_h^N and <.>_h the average with w_h / Z_h,
    <V> = V0 (N-1)/2 int dh P <cos 2 pi x>_h^2
    <T> = (N/2) int dh P [ <a^2>_h + 2 (beta/N) (N-1) <g>_h <a g'>_h + (beta/N)^2 ((N-1) var_h(g) + (N-1)^2 <g>_h^2) <g'^2>_h ],  a = (ln u)'
    rho(x, x') ~ u(x) u(x') exp[-(beta/N) G^2] int dh exp(-N h^2 / (4 beta)) Z_h^(N-1) exp(h G),  G = (g(x) + g(x'))/2
The five parameters minimise E_T = <T> + <V> at each (V0, N).

Usage (from the repository root):  python python/cat_trial.py --V0kac -20 -22 -25 [--N 40] [--save]
Output: table of E_T, N - lambda_0 and parameters; numpy_arrays_VMC/cat_trial_N<N>.csv;
with --save, numpy_arrays_VMC/wavefunction_cat/V0=.../wavefunction_<V0>_<N>.npy (Ng x Ng on [-1/2, 1/2)).
"""

import argparse
import os

import numpy as np
from scipy.optimize import minimize

NX, NH, NG = 256, 1601, 64
X = -0.5 + np.arange(NX) / NX
DX = 1.0 / NX
TP = 2 * np.pi


def fields(p, x=X):
    """g, g', ln u, (ln u)' on x, for p = (beta, gamma2, gamma4, d3, d5)."""
    _, c2, c4, d3, d5 = p
    g = np.cos(TP * x) + d3 * np.cos(3 * TP * x) + d5 * np.cos(5 * TP * x)
    dg = -TP * (np.sin(TP * x) + 3 * d3 * np.sin(3 * TP * x) + 5 * d5 * np.sin(5 * TP * x))
    lu = c2 * np.cos(2 * TP * x) + c4 * np.cos(4 * TP * x)
    a = -TP * (2 * c2 * np.sin(2 * TP * x) + 4 * c4 * np.sin(4 * TP * x))
    return g, dg, lu, a


def h_grid(beta, g, N):
    """Field grid covering the Gaussian width and the largest possible saddle 2 beta max|g|."""
    H = 2 * beta * np.abs(g).max() + 8 * np.sqrt(2 * beta / N)
    return np.linspace(-H, H, NH)


def log_weights(p, N, power):
    """(h, weight ~ exp(-N h^2/(4 beta)) Z_h^power normalised, w_h / Z_h on the rows)."""
    beta = p[0]
    g, _, lu, _ = fields(p)
    h = h_grid(beta, g, N)
    lw0 = 2 * lu - beta * g ** 2 / N
    W = np.exp(lw0 + np.outer(h, g))
    Z = W.sum(axis=1) * DX
    lp = -N * h ** 2 / (4 * beta) + power * np.log(Z)
    P = np.exp(lp - lp.max())
    return h, P / P.sum(), W / Z[:, None] * DX


def energy(p, V0, N):
    """Exact <H> of the trial; T and V separately."""
    beta = p[0]
    if beta <= 1e-5:
        return np.inf, 0.0, 0.0
    g, dg, _, a = fields(p)
    c = np.cos(TP * X)
    _, P, W = log_weights(p, N, N)
    mg, mc = W @ g, W @ c
    vg = W @ g ** 2 - mg ** 2
    q = beta / N
    T = 0.5 * N * P @ (W @ a ** 2 + 2 * q * (N - 1) * mg * (W @ (a * dg)) + q ** 2 * ((N - 1) * vg + (N - 1) ** 2 * mg ** 2) * (W @ dg ** 2))
    V = V0 * (N - 1) / 2 * P @ mc ** 2
    return T + V, T, V


def depletion(p, N):
    """(N - lambda_0, lambda_1, P(S=0)/P_max) of the trial, from the exact rho(x, x')."""
    beta = p[0]
    g, _, lu, _ = fields(p)
    h, P, _ = log_weights(p, N, N - 1)
    G = 0.5 * (g[:, None] + g[None, :])
    R = np.zeros((NX, NX))
    for hh, pp in zip(h[::4], P[::4]):
        if pp > 1e-12 * P.max():
            R += pp * np.exp(hh * G)
    R *= np.exp(lu[:, None] + lu[None, :] - beta / N * G ** 2)
    R *= N / (np.trace(R) * DX)
    w = np.linalg.eigvalsh(R * DX)[::-1]
    h0, P0, _ = log_weights(p, N, N)
    return N - w[0], w[1], P0[NH // 2] / P0.max()


def optimise(V0, N, starts=None):
    best = None
    for b0 in (starts or (0.05, 0.2, 0.5, 1.0, 2.0)):
        s = minimize(lambda p: energy(p, V0, N)[0], [b0, 0.0, 0.0, 0.0, 0.0], method="Nelder-Mead",
                     options=dict(maxiter=6000, maxfev=6000, xatol=1e-5, fatol=1e-7, adaptive=True))
        if best is None or s.fun < best.fun:
            best = s
    return best.x, best.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V0kac", type=float, nargs="+", required=True)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    N = args.N
    rows = []
    print("    V0      E_T        T        V     N-l0     l1   P(0)/Pmax  beta  gamma2  gamma4    d3      d5")
    for V in args.V0kac:
        p, E = optimise(V, N)
        _, T, Vv = energy(p, V, N)
        d, l1, p0 = depletion(p, N)
        print(f"  {V:6.1f} {E:10.4f} {T:8.3f} {Vv:9.3f} {d:7.3f} {l1:7.3f} {p0:8.1e} {p[0]:6.3f} {p[1]:7.4f} {p[2]:7.4f} {p[3]:7.4f} {p[4]:7.4f}")
        rows.append((V, E, T, Vv, d, l1, p0, *p))
        if args.save:
            x = -0.5 + np.arange(NG) / NG
            g, _, lu, _ = fields(p, x)
            lf1 = lu / (N - 1)
            f = np.exp(lf1[:, None] + lf1[None, :] + p[0] / N * np.outer(g, g))
            v = V / N
            d_ = os.path.join("numpy_arrays_VMC", "wavefunction_cat", f"V0={v:.3f}")
            os.makedirs(d_, exist_ok=True)
            np.save(os.path.join(d_, f"wavefunction_{v:.3f}_{N}.npy"), f)
    np.savetxt(os.path.join("numpy_arrays_VMC", f"cat_trial_N{N}.csv"), np.array(rows), delimiter=",", comments="", fmt="%.10g",
               header="V0,E_T,T,V,N_minus_l0,lambda1,P0_over_Pmax,beta,gamma2,gamma4,d3,d5")


if __name__ == "__main__":
    main()
