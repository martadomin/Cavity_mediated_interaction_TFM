"""
N - lambda_0 of the exact two-mode ground state across the critical window, against the one-mode Bogoliubov N'.

H2 as in julia/gp/two_mode.jl, on the GP orbitals at each (V0, N):
    H2(n_A, n_A) = N T + V0/(2N) Theta'^2 [(2 n_A - N)^2 - N],   H2(n_A+1, n_A) = t sqrt((n_A+1)(N-n_A))
The ground state is even under n_A <-> N - n_A, so rho^(1) is diagonal in phi_+, phi_-:
    lambda_0 = N/2 + g,  lambda_- = N - lambda_0 = N/2 - g,  g = <a_A^dag a_B>.
Below threshold N - lambda_0 is the depletion (all in phi_- = sqrt(2) cos 2 pi x); above it, it also counts the two-branch
structure of the cat (-> N (1 - s)/2 deep in the phase), so it is not a depletion.
Bogoliubov (below threshold): N' = (A / omega - 1) / 2, A = eps + V0 R / 2, omega = sqrt(eps (eps + V0 R)), R = (N-1)/N.
Large N at threshold: quartic oscillator, N - lambda_0 -> (1/2) <q^2> N^(1/3), with H = p^2/2 + q^4/4 (see two_mode_quartic).

Usage (from the repository root):  python python/critical_window_two_mode.py
Output: numpy_arrays_GP/critical_window_two_mode_N40.csv, numpy_arrays_GP/critical_window_threshold_vs_N.csv
"""

import os
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq, curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmc_gp_delta import _ground, NMAX, EPSK  # noqa: E402

n = np.arange(-NMAX, NMAX + 1)
COS = 0.5 * (np.eye(len(n), k=1) + np.eye(len(n), k=-1))
EPS = 2 * np.pi ** 2
Vc = lambda N: EPS * N / (N - 1)


def gp_orbital(V0, N):
    """GP orbital coefficients at (V0, N); uniform at and below threshold."""
    R = (N - 1) / N
    f = lambda A: A - V0 * R * _ground(A)[1]
    lo, hi = V0 * R, -1e-8
    A = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14) if (V0 < 0 and f(lo) * f(hi) < 0) else 0.0
    w, v = np.linalg.eigh(np.diag(EPSK * n.astype(float) ** 2) + A * COS)
    return v[:, 0] * np.sign(v[NMAX, 0])


def two_mode_state(V0, N, V_int=None):
    """Even ground state psi(n_A) of H2 on the GP orbitals at V0 (interaction strength V_int, default V0), hop, diagonal, t.
    N even. Even sector: basis (|k> + |N-k>)/sqrt(2), k < N/2, plus |N/2>; tridiagonal, so stable when the branches decouple."""
    c = gp_orbital(V0, N)
    cp, cm = c * (n % 2 == 0), c * (n % 2 != 0)
    cp /= np.linalg.norm(cp)
    cm = cm / np.linalg.norm(cm) if np.linalg.norm(cm) > 1e-12 else np.where(np.abs(n) == 1, 1 / np.sqrt(2), 0.0)
    a, b = (cp + cm) / np.sqrt(2), (cp - cm) / np.sqrt(2)
    Tk = EPSK * n.astype(float) ** 2
    T, t, Th = a @ (Tk * a), a @ (Tk * b), a @ COS @ a
    k = np.arange(N + 1)
    hop = np.sqrt((k[:-1] + 1) * (N - k[:-1]))
    d = N * T + (V0 if V_int is None else V_int) / (2 * N) * Th ** 2 * ((2 * k - N) ** 2 - N)
    m = N // 2
    de, oe = d[:m + 1].copy(), t * hop[:m].copy()
    oe[-1] *= np.sqrt(2)
    w, v = eigh_tridiagonal(de, oe, select="i", select_range=(0, 2))
    p = np.abs(v[:, 0])
    psi = np.concatenate([p[:-1] / np.sqrt(2), [p[-1]], p[:-1][::-1] / np.sqrt(2)])
    return psi, hop, d, t, w


def two_mode(V0, N):
    """(N - lambda_0, first gap in the even sector, first gap, <(n_A - N/2)^2>) of the two-mode ground state, N even."""
    psi, hop, d, t, w = two_mode_state(V0, N)
    k = np.arange(N + 1)
    g = psi[:-1] @ (psi[1:] * hop)
    w_all = eigh_tridiagonal(d, t * hop, eigvals_only=True, select="i", select_range=(0, 1))
    return N / 2 - g, w[1] - w[0], w_all[1] - w_all[0], (psi ** 2) @ (k - N / 2) ** 2


def bogoliubov(V0, N):
    """Analytic one-mode N' and frequency, below threshold (V0 > -Vc(N))."""
    R = (N - 1) / N
    A, om2 = EPS + V0 * R / 2, EPS * (EPS + V0 * R)
    return 0.5 * (A / np.sqrt(om2) - 1), np.sqrt(om2)


def two_mode_quartic(M=400):
    """<q^2>, gaps of H = p^2/2 + q^4/4, in a harmonic-oscillator basis."""
    a = np.diag(np.sqrt(np.arange(1, M)), 1)
    q = (a + a.T) / np.sqrt(2)
    p2 = -((a - a.T) @ (a - a.T)) / 2
    H = 0.5 * p2 + 0.25 * np.linalg.matrix_power(q, 4)
    H = H[:M - 6, :M - 6]
    w, v = np.linalg.eigh(H)
    q2 = (q @ q)[:M - 6, :M - 6]
    return v[:, 0] @ q2 @ v[:, 0], w[1] - w[0], w[2] - w[0]


def main():
    N = 40
    VC = Vc(N)
    win = N ** (-2 / 3)
    x = np.unique(np.concatenate([np.linspace(0.02, 0.8, 79), np.linspace(0.8, 1.2, 401), np.linspace(1.2, 3.0, 181)]))
    rows = []
    for xx in x:
        V = -xx * VC
        lm, ge, g1, j2 = two_mode(V, N)
        nb = bogoliubov(V, N)[0] if xx < 1 else np.nan
        rows.append((xx, V, lm, nb, ge, g1, j2))
    R = np.array(rows)
    np.savetxt(os.path.join("numpy_arrays_GP", "critical_window_two_mode_N40.csv"), R, delimiter=",", comments="", fmt="%.10g",
               header="V0_over_Vc,V0,N_minus_lambda0,Bogoliubov_Nprime,gap_even_sector,gap_lowest,Jz2")

    print(f"N = {N}: |V0|_c = {VC:.4f}, window |t| < {win:.4f}: {VC * (1 - win):.2f} < |V0| < {VC * (1 + win):.2f}")
    print("    V0    V0/Vc      t     N-l0(2 mode)  N'(Bog)    even gap   lowest gap")
    for V in (-10, -15, -17, -18, -18.5, -19, -19.5, -20, -VC, -20.5, -21, -21.5, -22, -23, -25, -30, -40):
        lm, ge, g1, _ = two_mode(float(V), N)
        nb = bogoliubov(V, N)[0] if -V < VC * (1 - 1e-12) else np.nan
        print(f"  {V:7.3f}  {-V / VC:.4f}  {-V / VC - 1:+.4f}  {lm:10.4f}  {nb:9.4f}  {ge:9.3f}  {g1:9.3e}")

    q2, g1q, g2q = two_mode_quartic()
    print(f"\nQuartic oscillator p^2/2 + q^4/4: <q^2> = {q2:.5f} -> N - l0 -> {q2 / 2:.5f} N^(1/3) (offset of the Bogoliubov vacuum not included); "
          f"gaps {g1q:.4f} (odd), {g2q:.4f} (even), times eps N^(-1/3)")

    # at threshold
    Ns = np.array([20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480])
    T = np.array([(M, *two_mode(-Vc(M), M)) for M in Ns])
    np.savetxt(os.path.join("numpy_arrays_GP", "critical_window_threshold_vs_N.csv"), T, delimiter=",", comments="", fmt="%.10g",
               header="N,N_minus_lambda0,gap_even_sector,gap_lowest,Jz2")
    print("\nAt V0 = -|V0|_c(N):    N    N-l0      local slope    N-l0 / N^(1/3)   gap_even    gap_even N^(1/3) / eps")
    for i, r in enumerate(T):
        sl = np.log(T[i, 1] / T[i - 1, 1]) / np.log(T[i, 0] / T[i - 1, 0]) if i else np.nan
        print(f"  {int(r[0]):7d}  {r[1]:9.4f}  {sl:7.4f}  {r[1] / r[0] ** (1 / 3):9.4f}   {r[2]:9.4f}   {r[2] * r[0] ** (1 / 3) / EPS:.4f}")
    sel = T[T[:, 0] <= 320]
    p, cov = np.polyfit(np.log(sel[:, 0]), np.log(sel[:, 1]), 1, cov=True)
    print(f"Fit N = 20..320: exponent = {p[0]:.4f} (+- {np.sqrt(cov[0, 0]):.4f} from residuals), prefactor {np.exp(p[1]):.4f}")
    lg = T[T[:, 0] >= 1280]
    p2 = np.polyfit(np.log(lg[:, 0]), np.log(lg[:, 1]), 1)
    print(f"Fit N >= 1280: exponent = {p2[0]:.4f}")
    # with the leading correction a N^(1/3) + b
    A = np.vstack([sel[:, 0] ** (1 / 3), np.ones(len(sel))]).T
    print("Fit N = 20..320 of a N^(1/3) + b: a, b =", np.linalg.lstsq(A, sel[:, 1], rcond=None)[0])
    # the threshold shift t ~ 1/N is N^(-1/3) in the scaling variable t N^(2/3), so N^(1/3) f(t N^(2/3)) has a constant offset
    f3 = lambda M, a, e, b: a * M ** e + b
    for lab, s in (("20..320", T[:, 0] <= 320), ("20..20480", T[:, 0] > 0), ("160..20480", T[:, 0] >= 160)):
        q, c3 = curve_fit(f3, T[s, 0], T[s, 1], p0=(0.2, 0.33, -0.3), maxfev=20000)
        print(f"Fit N = {lab} of a N^p + b: a = {q[0]:.4f}, p = {q[1]:.4f} +- {np.sqrt(c3[1, 1]):.4f}, b = {q[2]:.4f}")
    print("Offset N - l0 - (analytic a) N^(1/3):", np.round(T[:, 1] - q2 / 2 * T[:, 0] ** (1 / 3), 3))


if __name__ == "__main__":
    main()
