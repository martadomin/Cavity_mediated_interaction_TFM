"""
Cat branch vs Lowdin (Fock) modes from the same two orbitals, GP, N = 40.

phi_+ and phi_- are the normalised even / odd parts of the GP orbital (julia/gp/gp_scf.jl), i.e. the natural
orbitals of the cat, with occupations lambda_pm/N ~ (1 +- s)/2. From them:
    branches (cat): phi_{A,B} = sqrt((1+s)/2) phi_+ +- sqrt((1-s)/2) phi_-  (weights = occupations; phi_A = GP
                    orbital, phi_B(x) = phi_A(x + lambda/2))
    Lowdin (Fock):  chi_{A,B} = (phi_+ +- phi_-)/sqrt(2)                   (equal weights)
so the only difference is the weights. Near threshold (s large) chi_A is sharper than phi_A and has less
weight on the B sites; deep in the phase (s -> 0) they coincide.
Plotted: the (real) amplitudes phi(x), chi(x), normalised to one on a period (uniform state: 1); the
printed peaks and weights refer to the densities |...|^2.

Usage (from the repository root, after julia julia/gp/gp_scf.jl):
    python python/plots/plot_branch_vs_lowdin.py                 # V0 = -25 -60 -150
    python python/plots/plot_branch_vs_lowdin.py --V0kac -30 -100
Output: final_imag/branch_vs_lowdin_N40.pdf (or branch_vs_lowdin_N40_V0<...>.pdf for a custom list)
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 30,
    "text.usetex": True,
})

N = 40
DEFAULT = [-25.0, -60.0, -150.0]


def field(c, n, x):
    return np.real(np.exp(2j * np.pi * np.outer(x, n)) @ c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=DEFAULT)
    args = p.parse_args()

    path = os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv")
    with open(path) as f:
        n = np.array([int(v) for v in f.readline().strip().split(",")[1:]])
    cn = np.loadtxt(path, delimiter=",", skiprows=1)
    x = np.linspace(0, 2, 801)
    onB = np.abs(((x + 0.5) % 1.0) - 0.5) > 0.25                  # B sublattice: |x - lambda/2| < lambda/4 mod lambda

    Vs = [V for V in args.V0kac if np.isclose(cn[:, 0], V).any() and
          np.linalg.norm(cn[np.isclose(cn[:, 0], V)][0, 1:] * (n % 2 != 0)) > 1e-12]
    for V in [V for V in args.V0kac if V not in Vs]:
        print(f"V0 = {V:.0f}: no GP orbital with an odd part (not in {path} or below threshold), skipped")
    fig, axes = plt.subplots(1, len(Vs), figsize=(13 * len(Vs), 9.5), squeeze=False)

    print(f"N = {N}:   V0     s     peak |phi_A|^2  peak |chi_A|^2   weight on B sites: phi_A   chi_A")
    for ax, V in zip(axes[0], Vs):
        c = cn[np.isclose(cn[:, 0], V)][0, 1:]
        s = np.sum(c * c * (-1.0) ** n)
        cp, cm = c * (n % 2 == 0), c * (n % 2 != 0)
        cp, cm = cp / np.linalg.norm(cp), cm / np.linalg.norm(cm)
        fphi, fphiB = field(c, n, x), field(c * (-1.0) ** n, n, x)
        fA, fB = field((cp + cm) / np.sqrt(2), n, x), field((cp - cm) / np.sqrt(2), n, x)
        rphi, rA = fphi ** 2, fA ** 2
        ax.plot(x, fphi, "-", color="k", lw=3.5, label=r"branch $\varphi_A$")
        ax.plot(x, fphiB, "--", color="0.45", lw=3.5, label=r"branch $\varphi_B$")
        ax.plot(x, fA, "-", color="C0", lw=3, label=r"Fock mode $\chi_A$")
        ax.plot(x, fB, "--", color="C3", lw=3, label=r"Fock mode $\chi_B$")
        ax.axhline(1, color="0.6", lw=1)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(fr"$V_0={V:.0f}\,\varepsilon$,\ $s={s:.2f}$")
        ax.set_xlabel(r"$x/\lambda$")
        ax.grid(True, ls="--", alpha=0.6)
        wphi, wchi = rphi[onB].sum() / rphi.sum(), rA[onB].sum() / rA.sum()
        print(f"        {V:6.0f}  {s:.3f}    {rphi.max():.3f}          {rA.max():.3f}"
              f"                       {wphi:.3f}    {wchi:.3f}")
    axes[0, 0].set_ylabel(r"amplitude $\times\,\sqrt{\lambda}$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False)

    os.makedirs("final_imag", exist_ok=True)
    tag = "" if args.V0kac == DEFAULT else "_V0" + "_".join(f"{V:.0f}" for V in Vs)
    out = os.path.join("final_imag", f"branch_vs_lowdin_N{N}{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
