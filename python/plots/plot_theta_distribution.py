"""
Distribution of the Z2 order parameter Theta = (1/N) sum_i cos(2 pi x_i): QMC vs GP cat and two-mode, N = 40.
The QMC counterpart of the two-mode Fock weights |c_{n_A}|^2 (python/plots/plot_cat_weights.py): in the two-mode
space sum_i cos(2 pi x_i) -> Theta' (n_A - n_B), so a cat gives two peaks at +- Theta.

GP cat (phi_A^N + phi_B^N, cross terms ~ s^N): each branch is a product state, so Theta is a sum of N
independent terms (central limit theorem): two Gaussians of weight 1/2 at +- Theta_GP = +- sum_n c_n c_{n+1}
with variance (<cos^2>_phi - Theta_GP^2)/N, <cos^2>_phi = (1 + sum_n c_n c_{n+2})/2.
Two-mode ground state (julia/gp/two_mode.jl, weights |c_{n_A}|^2 from two_mode_weights_N40.csv): each Fock
state chi_A^n_A chi_B^(N-n_A) is a product state, so its Theta is Gaussian with mean Theta' (2 n_A - N)/N and
variance (<cos^2>_chi - Theta'^2)/N (same for chi_A and chi_B); P(Theta) = sum_n_A |c_n_A|^2 x Gaussian.
This neglects the interference between neighbouring n_A (matrix elements of nonlinear functions of
sum cos between Fock states), which does not move the peaks.
QMC:
  * VMC: Theta at each measurement of the standalone VMC runs (julia/src/sampler.jl, |Psi_T|^2,
    numpy_arrays_VMC/.../theta, 10^5 samples; not available at V0 = -25 for N = 40);
  * DMC: Theta of the final walker configurations (numpy_arrays_DMC/.../walkers, one snapshot of
    ~1200-2400 walkers of the longest / largest run), sampled from the MIXED distribution Psi_T Psi_0.

Usage (from the repository root, after julia julia/gp/gp_scf.jl):
    python python/plots/plot_theta_distribution.py            # V0 = -25 -60 -150
Output: final_imag/theta_distribution_N40.pdf
"""

import argparse
import os
import re

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
BV = os.path.join("numpy_arrays_VMC", "constant_density_1.0", "theta")
BW = os.path.join("numpy_arrays_DMC", "constant_density_1.0", "walkers")


def dmc_walkers(V):
    """(positions, label) of the longest / largest DMC run, or (None, '')."""
    v = V / N
    d = os.path.join(BW, f"V0={v:.3f}")
    pat = re.compile(rf"^walkers_(\d+)_([\d.]+)_{N}_(\d+)_{v:.3f}_{float(N):.1f}_IS\.npy$")
    best = None
    for f in (os.listdir(d) if os.path.isdir(d) else []):
        m = pat.match(f)
        if m:
            w, dt, st = int(m[1]), float(m[2]), int(m[3])
            key = (round(st * dt, 6), w)
            if best is None or key > best[0]:
                best = (key, f, w, st * dt)
    if best is None:
        return None, ""
    return np.load(os.path.join(d, best[1])), f"{best[2]} walkers, $\\tau={best[3]:g}$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--V0kac", type=float, nargs="+", default=[-25.0, -60.0, -150.0])
    args = p.parse_args()

    path = os.path.join("numpy_arrays_GP", f"gp_cn_N{N}.csv")
    cn = np.loadtxt(path, delimiter=",", skiprows=1)
    t = np.linspace(-1, 1, 801)
    bins = np.linspace(-1, 1, 51)
    wp = os.path.join("numpy_arrays_GP", f"two_mode_weights_N{N}.csv")
    with open(wp) as fh:
        wc = fh.readline().strip().split(",")
    wd = np.loadtxt(wp, delimiter=",", skiprows=1)
    wts = {name: wd[:, i] for i, name in enumerate(wc)}

    Vs = [V for V in args.V0kac if np.isclose(cn[:, 0], V).any()]
    fig, axes = plt.subplots(1, len(Vs), figsize=(13 * len(Vs), 9.5), squeeze=False)
    print(f"N = {N}:  V0    Theta_GP  sigma_GP | <|Theta|>: VMC     DMC   | <Theta^2>: GP     VMC     DMC   "
          f"2-mode  [DMC run]")
    for ax, V in zip(axes[0], Vs):
        c = cn[np.isclose(cn[:, 0], V)][0, 1:]
        th = np.sum(c[:-1] * c[1:])
        cos2 = 0.5 * (1 + np.sum(c[:-2] * c[2:]))
        sig = np.sqrt(max(cos2 - th ** 2, 1e-12) / N)
        g = 0.5 * (np.exp(-(t - th) ** 2 / (2 * sig ** 2)) + np.exp(-(t + th) ** 2 / (2 * sig ** 2))) / np.sqrt(2 * np.pi * sig ** 2)
        ax.plot(t, g, "k-", lw=3, label="GP cat")

        # two-mode ground state: mixture over n_A of the Fock-state Gaussians (see docstring)
        n = np.arange(-(len(c) // 2), len(c) // 2 + 1)
        cp, cm = c * (n % 2 == 0), c * (n % 2 != 0)
        cp, cm = cp / np.linalg.norm(cp), cm / np.linalg.norm(cm)
        a = (cp + cm) / np.sqrt(2)
        thp = np.sum(a[:-1] * a[1:])
        var = 0.5 * (1 + np.sum(a[:-2] * a[2:])) - thp ** 2
        col = f"w_V0{V:.0f}"
        th2m = np.nan
        if col in wts:
            k = np.arange(N + 1)
            mu = thp * (2 * k - N) / N
            p2 = np.sum(wts[col][:, None] * np.exp(-(t[None, :] - mu[:, None]) ** 2 / (2 * var / N)), axis=0)
            p2 /= np.sqrt(2 * np.pi * var / N)
            ax.plot(t, p2, "--", color="C3", lw=3, label="two-mode")
            th2m = np.sum(wts[col] * (mu ** 2 + var / N))

        v = V / N
        fv = os.path.join(BV, f"V0={v:.3f}", f"theta_{N}_{v:.3f}_{float(N):.1f}.npy")
        tv = np.load(fv) if os.path.isfile(fv) else None
        if tv is not None:
            ax.hist(tv, bins=bins, density=True, histtype="step", lw=3, color="C1", label="VMC")
        X, run = dmc_walkers(V)
        td = np.cos(2 * np.pi * X).mean(axis=1) if X is not None else None
        if td is not None:
            ax.hist(td, bins=bins, density=True, color="C2", alpha=0.5, label="DMC (mixed, final walkers)")
        ax.set_title(fr"$V_0={V:.0f}\,\varepsilon$")
        ax.set_xlabel(r"$\Theta=\frac{1}{N}\sum_i\cos 2\pi x_i$")
        ax.set_xlim(-1, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.6)
        f = lambda a, fn: np.nan if a is None else fn(a)
        print(f"        {V:5.0f}  {th:.4f}    {sig:.4f}  |   {f(tv, lambda a: np.abs(a).mean()):.4f}   "
              f"{f(td, lambda a: np.abs(a).mean()):.4f} |   {th ** 2 + sig ** 2:.4f}  {f(tv, lambda a: (a ** 2).mean()):.4f}"
              f"  {f(td, lambda a: (a ** 2).mean()):.4f}  {th2m:.4f}  [{run}, {0 if td is None else len(td)} configs]")
    axes[0, 0].set_ylabel(r"$P(\Theta)$")
    h, lab = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh); lab.append(ll)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False)
    os.makedirs("final_imag", exist_ok=True)
    out = os.path.join("final_imag", f"theta_distribution_N{N}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
