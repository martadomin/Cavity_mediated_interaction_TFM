"""
Natural orbitals phi_+ (even) and phi_- (odd) in real space at N = 40: QMC vs GP, several V0 in one figure.

QMC: the even / odd natural orbitals of the condensate block of rho^(1) (python/obdm_momentum.py,
numpy_arrays_OBDM/momentum_N40_V0kac<V>.npz, |n| <= 8): DMC mixed and the extrapolated 2 DMC - VMC.
GP: the normalised even / odd parts of the GP orbital (julia/gp/gp_scf.jl, numpy_arrays_GP/gp_cn_N40.csv),
i.e. the natural orbitals of the GP cat. Signs of the QMC orbitals fixed by a positive overlap with GP.
Amplitudes normalised to one on a period (uniform state: 1); the overlaps |<phi^QMC|phi^GP>|^2 are printed.

Usage (from the repository root, after julia julia/gp/gp_scf.jl and python python/obdm_momentum.py):
    python python/plots/plot_orbitals_qmc_gp.py                    # V0 = -25 -40 -60 -150
    python python/plots/plot_orbitals_qmc_gp.py --V0kac -30 -80
Output: final_imag/orbitals_qmc_gp_N40.pdf (or orbitals_qmc_gp_N40_V0<...>.pdf for a custom list)
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
    "legend.fontsize": 24,
    "text.usetex": True,
})

N = 40
DEFAULT = [-25.0, -40.0, -60.0, -150.0]


def field(c, n, x):
    return np.real(np.exp(2j * np.pi * np.outer(x, n)) @ c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=DEFAULT)
    args = p.parse_args()

    path = os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv")
    with open(path) as f:
        ngp = np.array([int(v) for v in f.readline().strip().split(",")[1:]])
    cn = np.loadtxt(path, delimiter=",", skiprows=1)
    x = np.linspace(0, 1, 401)
    xm = np.linspace(0, 1, 26)

    Vs = [V for V in args.V0kac
          if np.isclose(cn[:, 0], V).any() and os.path.isfile(f"numpy_arrays_OBDM/momentum_N{N}_V0kac{V:.0f}.npz")
          and np.linalg.norm(cn[np.isclose(cn[:, 0], V)][0, 1:] * (ngp % 2 != 0)) > 1e-12]
    for V in [V for V in args.V0kac if V not in Vs]:
        print(f"V0 = {V:.0f}: no OBDM data or GP below threshold, skipped")
    cmap = plt.get_cmap("plasma")
    cols = {V: cmap(0.85 * i / max(len(Vs) - 1, 1)) for i, V in enumerate(Vs)}

    fig, axes = plt.subplots(1, 2, figsize=(26, 10.5))
    print(f"N = {N}:   V0    |<phi+|GP+>|^2: DMC    lin   |  |<phi-|GP->|^2: DMC    lin")
    for V in Vs:
        c = cn[np.isclose(cn[:, 0], V)][0, 1:]
        gp = {"plus": c * (ngp % 2 == 0), "minus": c * (ngp % 2 != 0)}
        gp = {k: v / np.linalg.norm(v) for k, v in gp.items()}
        d = np.load(f"numpy_arrays_OBDM/momentum_N{N}_V0kac{V:.0f}.npz")
        ov = {}
        for ax, key in zip(axes, ("plus", "minus")):
            ax.plot(x, field(gp[key], ngp, x), "-", color=cols[V], lw=3, label=fr"$V_0={V:.0f}\,\varepsilon$ GP")
            for est, mk, filled in (("DMC", "o", True), ("lin", "^", False)):
                ns = d[f"{est}_ns"]
                q = d[f"{est}_phi_{key}"].copy()
                g = np.array([gp[key][ngp == k][0] for k in ns])
                if np.dot(q, g) < 0:
                    q = -q
                ov[(est, key)] = np.dot(q, g) ** 2
                ax.plot(xm, field(q, ns, xm), mk, color=cols[V], ms=11, mew=2, mfc=cols[V] if filled else "none")
        print(f"        {V:6.0f}          {ov[('DMC', 'plus')]:.5f} {ov[('lin', 'plus')]:.5f}   |"
              f"            {ov[('DMC', 'minus')]:.5f} {ov[('lin', 'minus')]:.5f}")
    for ax, t in zip(axes, (fr"$N={N}$: $\phi_+$ (even)", fr"$N={N}$: $\phi_-$ (odd)")):
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_title(t)
        ax.grid(True, ls="--", alpha=0.6)
    axes[0].set_ylabel(r"amplitude $\times\,\sqrt{\lambda}$")
    h, lab = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", ls="", color="0.3", ms=11),
          plt.Line2D([], [], marker="^", ls="", color="0.3", mfc="none", mew=2, ms=11)]
    lab += ["DMC mixed (markers)", r"$2\,$DMC$-$VMC (markers)"]
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False)

    os.makedirs("final_imag", exist_ok=True)
    tag = "" if args.V0kac == DEFAULT else "_V0" + "_".join(f"{V:.0f}" for V in Vs)
    out = os.path.join("final_imag", f"orbitals_qmc_gp_N{N}{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
