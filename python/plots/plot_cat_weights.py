"""
Two-mode ground state in the Fock basis |n_A, N - n_A> of the Lowdin modes chi_A, chi_B: weights |c_{n_A}|^2
vs n_A/N on a linear scale (data from julia/gp/two_mode.jl, numpy_arrays_GP/two_mode_weights_N40.csv), N = 40,
compared with the GP cat.

GP cat in the same basis: phi_A = a chi_A + b chi_B, phi_B = b chi_A + a chi_B with
    a = [sqrt((1+s)/2) + sqrt((1-s)/2)]/sqrt(2),  b = [sqrt((1+s)/2) - sqrt((1-s)/2)]/sqrt(2)
(s = <phi_A|phi_B> from two_mode.csv), so phi_A^N|0> is a binomial in n_A peaked at n_A/N = a^2 and
    |c_{n_A}|^2 (cat) ∝ C(N, n_A) [a^n_A b^(N-n_A) + b^n_A a^(N-n_A)]^2,
bimodal with peaks at a^2 = (1 + sqrt(1 - s^2))/2 and b^2 = 1 - a^2 that move to the edges as s -> 0.
n_A counts atoms in the orthogonal mode chi_A; the label "phi_A^N + phi_B^N" names the state (GP cat), not the basis.

Usage (from the repository root, after julia julia/gp/two_mode.jl):
    python python/plots/plot_cat_weights.py                    # V0 = -25 -60 -150
Output: final_imag/cat_weights_N40.pdf
"""

import argparse
import os
from math import comb

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 28,
    "text.usetex": True,
})

N = 40


def read_csv(path):
    with open(path) as f:
        cols = f.readline().strip().split(",")
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return {c: data[:, i] for i, c in enumerate(cols)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=[-25.0, -60.0, -150.0])
    args = p.parse_args()

    wts = read_csv("numpy_arrays_GP/two_mode_weights_N40.csv")
    tab = read_csv("numpy_arrays_GP/two_mode.csv")
    k = np.arange(N + 1)
    Vs = [V for V in args.V0kac if f"w_V0{V:.0f}" in wts]
    for V in [V for V in args.V0kac if V not in Vs]:
        print(f"V0 = {V:.0f}: not in two_mode_weights_N40.csv (V0 = -25, -40, -60, -150), skipped")

    fig, axes = plt.subplots(1, len(Vs), figsize=(13 * len(Vs), 9.5), squeeze=False)
    print(f"N = {N}:  V0     s      a^2    peak of two-mode weights (n_A/N)   overlap |<cat|2-mode>|")
    for ax, V in zip(axes[0], Vs):
        w = wts[f"w_V0{V:.0f}"]
        s = float(tab["s"][(tab["V0"] == V) & (tab["N"] == N)][0])
        u, v = np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)
        a, b = (u + v) / np.sqrt(2), (u - v) / np.sqrt(2)
        amp = np.array([np.sqrt(comb(N, n)) * (a ** n * b ** (N - n) + b ** n * a ** (N - n)) for n in k])
        amp /= np.linalg.norm(amp)
        ax.bar(k / N, w, width=0.8 / N, color="C0", alpha=0.75, label="two-mode ground state")
        ax.plot(k / N, amp ** 2, "o-", color="k", ms=7, lw=2, label=r"GP cat $\propto\varphi_A^N+\varphi_B^N$")
        ax.set_title(fr"$V_0={V:.0f}\,\varepsilon$,\ $s={s:.2f}$")
        ax.set_xlabel(r"$n_A/N$ (atoms in $\chi_A$)")
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.6)
        half = k < N / 2
        ov = abs(np.dot(np.sqrt(w), amp))                # both ground states are positive in this basis
        print(f"        {V:5.0f}  {s:.3f}  {a * a:.3f}   {k[half][np.argmax(w[half])] / N:.3f}"
              f"                              {ov:.4f}")
    axes[0, 0].set_ylabel(r"$|c_{n_A}|^2$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False)
    os.makedirs("final_imag", exist_ok=True)
    out = os.path.join("final_imag", f"cat_weights_N{N}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
