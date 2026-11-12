# Mean-field (GP), two-orbital and DMC-comparison code

This folder and a set of Python scripts benchmark the VMC/DMC results for the 1D cavity
Bose gas with **g = 0** against single-orbital mean field (Gross–Pitaevskii, GP) and a
two-orbital (two-mode) model. This document explains what each script computes, how,
which checks it runs, and where its output goes.

All scripts are run **from the repository root**.

---

## 1. Model and conventions

$$
H=\sum_i\Big(-\tfrac12\,\partial_{x_i}^2\Big)+\frac{V_0}{N}\sum_{i<j}\cos(2\pi x_i)\cos(2\pi x_j)
$$

- Units: ħ = m = 1, length λ = 1, energy ε = ħ²/(mλ²). Cavity wave number k = 2π.
- Recoil energy ε_k = ħ²k²/2m = 2π² ≈ 19.739 ε.
- V0 is the **Kac-scaled** amplitude; the pair amplitude in H is V0/N. The sum runs over
  i < j (no extra ½). In the QMC code this is `V0_bare = V0_kac / N`
  (`julia/dmc/run_dmc.jl`), used in `julia/src/energy.jl`.
- Density n = 1, so the box has L = N and contains N cavity periods (periodic boundaries).
- The ground state of H is periodic with period λ, so all mean-field orbitals live on
  **one unit cell** and are expanded in plane waves e^{i2πnx}, n = −n_max … n_max.

H has two symmetries used throughout:
- **Z₂ (half-period shift)** x → x + ½: cos(2πx) → −cos(2πx) for every particle, so H is
  invariant. In plane waves this is c_n → (−1)ⁿ c_n.
- **Reflection** x → −x: the ground-state orbitals have c_{−n} = c_n and are real.

---

## 2. GP by self-consistent field — `gp_scf.jl`

### Energy of a product state
For Ψ = φ(x₁)⋯φ(x_N) with φ(x) = Σ_n c_n e^{i2πnx} (c_n real, Σc_n² = 1):

- kinetic energy per particle: ε_k Σ_n n² c_n²
- ⟨cos 2πx⟩ in φ: Θ₁ = Σ_n c_n c_{n+1} (cos couples n to n ± 1)
- the N(N−1)/2 pairs each contribute (V0/N) Θ₁²

so

$$
\frac{E}{N}=\varepsilon_k\sum_n n^2c_n^2+\frac{V_0}{2}\,\frac{N-1}{N}\,\Theta_1^2 .
$$

The factor (N−1)/N is the exact finite-N pair counting. N → ∞ sets it to 1.

### Self-consistency
Minimising E/N over normalised c gives a one-body eigenvalue problem
h[A] c = μ c, with

- diagonal ε_k n²,
- off-diagonal A/2 (between n and n ± 1), i.e. the potential A cos(2πx),
- **A = V0 (N−1)/N · Θ₁** (the mean field felt by one particle).

h[A] is a `SymTridiagonal` matrix, so for a given A its lowest eigenvector is cheap.
Self-consistency reduces to **one scalar equation**:

$$
f(A)=A-V_0\,\tfrac{N-1}{N}\,\Theta_1(A)=0 ,
$$

where Θ₁(A) is evaluated from the lowest eigenvector of h[A]. A = 0 (uniform state) is
always a root; the nontrivial root is found with **Brent's method** (`Roots.jl`):

- bracket A ∈ [V0 (N−1)/N, −10⁻⁸]: at the lower end Θ₁ ≤ 1 gives f < 0; near A = 0,
  second-order perturbation theory gives Θ₁ ≈ −A/ε_k, so f ≈ A (1 − |V0|(N−1)/(N ε_k)),
  which is positive only above threshold;
- **no sign change → uniform state** (c_n = δ_{n0}, Θ₁ = 0, E = 0).

This also gives the **threshold** analytically: |V0| = ε_k N/(N−1)
(20.245 ε for N = 40, 19.739 ε for N → ∞).

Brent is used instead of the usual damped fixed-point iteration (Θ ← Θ(A(Θ))) because
it needs no mixing parameter and doesn't slow down near threshold, where the nontrivial
root appears continuously. Both converge to the same solution.

**The energy is computed from the formula above, not from the eigenvalue μ**
(μ = ⟨T⟩ + V0 (N−1)/N Θ₁² double-counts the interaction).

### Derived quantities
- **Branch overlap** s = Σ_n (−1)ⁿ c_n² = ⟨φ | φ(· + ½)⟩: overlap of the GP orbital with
  its Z₂ partner.
- **Cat-state occupations.** The symmetric two-branch state |φ^N⟩ + |φ(·+½)^N⟩ has two
  natural orbitals with
  λ±/N = (1 ± s)(1 ± s^{N−1}) / (2(1 + s^N)), → (1 ± s)/2 for N → ∞.

### Checks (printed as PASS/FAIL)
1. V0 = 0: c_n = δ_{n0}, E = 0, Θ₁ = 0.
2. Threshold at |V0| = ε_k N/(N−1) for N = 40 and N → ∞ (Θ₁ = 0 just below, > 0 just above).
3. n_max = 20 vs 40: |ΔE/N| < 10⁻¹⁰ on the whole V0 grid.
4. E/N < 0 on the whole V0 grid. As written, this prints FAIL because the grid includes
   V0 = −15 and −20, which are below threshold at N = 40, where the solution is uniform and
   E = 0 exactly. Every V0 above threshold has E/N < 0. The failure comes from how the
   check is written; the physics is fine.
5. Variational ordering E_DMC ≤ E_GP within one error bar (DMC at N = 40 from the newest
   energy file, DMC N → ∞ from `thermodynamic_limit.py`). **This check currently fails**
   at some couplings. Together with check 4, this makes the script end with exit code 1. The cause is DMC bias (old
   short runs, and population-control bias at large N and |V0|), not GP. See
   `python/dmc_gp_delta.py` and `python/dmc_equilibration.py`.

The results agree with the earlier independent Python mean-field code to all printed
digits (e.g. E/N = −58.15629 at V0 = −200, N → ∞).

### Run and output
```
julia julia/gp/gp_scf.jl
```
| file (`numpy_arrays_GP/`) | content |
|---|---|
| `gp_scf_N40.csv`, `gp_scf_Ninf.csv` | V0, A, Θ₁, E/N, s, λ±/N on the V0 grid −15 … −200 |
| `gp_cn_N40.csv`, `gp_cn_Ninf.csv` | GP orbital c_n, n = −30 … 30, one row per V0 |
| `gp_dense.csv` | E/N, Θ₁, s, λ±/N on a 0.5 ε grid (0 … −210), N = 40 and N → ∞ |

Default n_max = 30 (converged to 10⁻¹⁴).

---

## 3. Two-orbital ("split") Fock states — `split_energy.jl`

Question: how much energy does it cost to fragment the condensate between the two Z₂
configurations?

### Orthonormal two-orbital basis
From the GP coefficients c_n:
- c⁺ = even-n part of c, normalised; c⁻ = odd-n part, normalised;
- φ_A = (φ⁺ + φ⁻)/√2, φ_B = (φ⁺ − φ⁻)/√2, so c^B_n = (−1)ⁿ c^A_n.

φ_A and φ_B are orthonormal and map into each other under x → x + ½. When the branch
overlap s is small, φ_A ≈ the GP orbital (they coincide for s = 0). Near threshold they
don't: φ_A always has exactly half its weight on odd n, which costs at least ε_k/2 ≈ 9.9
in kinetic energy.

### Why the Fock-state energy is exact in this basis
cos(2πx) connects n only with n ± 1, i.e. even ↔ odd, so

- ⟨+|cos|+⟩ = ⟨−|cos|−⟩ = 0 and ⟨A|cos|B⟩ = 0,
- ⟨A|cos|A⟩ = −⟨B|cos|B⟩ = Θ' = Σ_n c^A_n c^A_{n+1}.

Since cos is diagonal in {A, B}, the exchange term of the pair interaction vanishes, and
the direct term of the Fock state |n_A, n_B⟩ (n_B = N − n_A) is

(V0/N) · ½ Θ'² [n_A(n_A−1) + n_B(n_B−1) − 2 n_A n_B] = (V0/2N) Θ'² [(n_A − n_B)² − N].

With T = 2π² Σ n² (c^A_n)² (equal for A and B):

$$
\frac{E(n_A)}{N}=T+\frac{V_0}{2N^2}\,\Theta'^2\big[(n_A-n_B)^2-N\big].
$$

The fragmentation gap (50/50 minus all in one orbital) is |V0| Θ'²/2 per particle, so the
total gap grows ∝ N. For odd N, "50/50" means n_A = (N−1)/2.

### Inputs, checks, output
- GP orbitals: N = 40 from `numpy_arrays_GP/gp_cn_N40.csv`. Other N are computed with
  `solve_gp` from `gp_scf.jl`. **Only its definitions are loaded** (`const`, `struct`,
  `function` expressions, via `Meta.parseall`). Its top-level code doesn't run, and the
  file is not modified. The same mechanism is used in `two_mode.jl`.
- Checks: ⟨A|B⟩ = 0, ⟨A|cos|B⟩ = 0, T_A = T_B, E(n_A) = E(N − n_A), minimum at n_A ∈ {0, N},
  E(all in A) ≥ E_GP (φ_A^N is a product state, so GP is lower).
- DMC energy shown alongside: for each (N, V0), the run with the longest imaginary time,
  then the most walkers.
- Runs: V0 = −25 … −150 at N = 40, and V0 = −150 at N = 15, 20, 25, 30.

```
julia julia/gp/split_energy.jl
python python/plots/plot_split.py
```
| output | content |
|---|---|
| `numpy_arrays_GP/split_energy.csv` | V0, N, T, Θ', s, E(all A)/N, E(50/50)/N, gap/N, total gap, E_GP/N, E_DMC/N ± err, DMC run |
| `numpy_arrays_GP/split_energy_curves_N40.csv` | E(n_A)/N vs n_A/N at N = 40 (data behind the figure) |
| `final_imag/split_energy_N40.pdf` | E(n_A)/N for V0 = −40, −80, −150 with E_GP and E_DMC(−150) lines |

---

## 4. Two-mode ground state — `two_mode.jl`

Instead of a single Fock state, diagonalise H in the full space spanned by φ_A and φ_B
(all superpositions of |n_A, N − n_A⟩, n_A = 0 … N).

### Hamiltonian
In second quantisation restricted to {A, B}:
- kinetic: T (n_A + n_B) + t (a†b + b†a), with t = ⟨A|T|B⟩ = 2π² Σ_n (−1)ⁿ n² (c^A_n)²
  = (T₊ − T₋)/2 < 0;
- interaction: the projected cos ⊗ cos is exact as in §3.

So H is an (N+1)×(N+1) real symmetric tridiagonal matrix:
- diagonal H(n_A) = N T + (V0/2N) Θ'² [(2n_A − N)² − N],
- off-diagonal ⟨n_A+1|H|n_A⟩ = t √((n_A+1)(N−n_A)).

This is the projection of the full H onto a subspace, so **E2 ≥ E_exact**. The GP product
state (√p_even φ⁺ + √p_odd φ⁻)^N lies in this subspace, so also **E2 ≤ E_GP(N)**.

### Symmetric sector
H commutes with n_A ↔ N − n_A, and with t < 0 its ground state is the symmetric one.
Deep in the phase the symmetric and antisymmetric states are split by ∝ t^N, which is
below machine precision; a plain `eigen` then returns an arbitrary mixture localised on one
branch (λ+/N = 1). The script therefore diagonalises in the **symmetric sector**, basis
(|k⟩ + |N−k⟩)/√2 for k < N/2 plus |N/2⟩ for even N, and checks that its lowest eigenvalue
equals the lowest eigenvalue of the full matrix.

### Observables
From the ground state ψ(n_A):
- E2/N = lowest eigenvalue / N;
- 2×2 one-body density matrix: ρ_AA = Σ|ψ|² n_A, ρ_BB = N − ρ_AA,
  ρ_AB = Σ ψ(n_A+1) ψ(n_A) √((n_A+1)(N−n_A)); eigenvalues λ±;
- weights |ψ(n_A)|².

### Below threshold: choice of φ⁻
Below threshold (e.g. V0 = −20 at N = 40, and V0 = 0) the GP orbital is uniform and its
odd part is empty, so φ⁻ can't be built from it. There the script uses

φ⁻(x) = √2 cos(2πx), i.e. c⁻_{±1} = 1/√2,

because it is
- the lowest-energy orbital that is odd under the λ/2 shift,
- the one the cavity term couples to the constant (sin 2πx is also odd but not coupled, and
  the ground state is even under x → −x),
- with φ⁺ = 1, the standard two-mode (Dicke) description of self-organisation
  (Nagy et al.).

This gives t = −π². At V0 = 0 the ground state is then exactly all particles in φ⁺.

### Checks
- V0 = 0: λ+ = N, E2 = 0.
- GP orbitals identical to `gp_cn_N40.csv`.
- V0 = −150: weight concentrated at n_A ∈ {0, N}; λ±/N = (1 ± s)/2; E2 = E_GP to 10⁻⁶.
- E2 ≤ E_GP(N) at every (V0, N).

### Run and output
```
julia julia/gp/two_mode.jl
python python/plots/plot_two_mode.py
```
Runs V0 = −20, −25, −30, −35, −40, −50, −60, −80, −100, −150 at N = 40. Other N
(15 … 35) are added to the table where a DMC run with τ ≥ 32 exists.

| output | content |
|---|---|
| `numpy_arrays_GP/two_mode.csv` | V0, N, E_GP/N, E2/N, E_DMC/N ± err, E_DMC − E2, λ±/N (two-mode), λ₀,λ₁/N from QMC (VMC, DMC mixed, 2·DMC−VMC; N = 40), s, T, t, Θ', DMC run |
| `numpy_arrays_GP/two_mode_dense_N40.csv` | E_GP/N, E2/N, λ±/N on a 0.5 ε grid (0 … −60) |
| `numpy_arrays_GP/two_mode_weights_N40.csv` | \|ψ(n_A)\|² for V0 = −25, −40, −60, −150 |
| `final_imag/two_mode_energy_N40.pdf` | E/N vs V0 in the crossover (GP, two-mode, DMC) and (E − E_GP)/N |
| `final_imag/two_mode_weights_N40.pdf` | \|ψ(n_A)\|² vs n_A/N (log scale) |

---

## 5. GP superfluid fraction — `python/gp_superfluid.py`

Leggett formula on one unit cell: ρ_s/ρ = 1 / (⟨n⟩⟨1/n⟩), evaluated on 8000 points.
Two GP densities:
- **single branch** n(x) = |φ(x)|² (period λ). For a single-orbital state in 1D this is
  the exact phase-twist superfluid fraction of that state.
- **Z₂-symmetric** n(x) = ½[|φ(x)|² + |φ(x+½)|²] (period λ/2). This is the like-for-like
  comparison with the QMC values (`python/leggett_analysis.py`), which symmetrise the
  measured density under x → x + ½. For a cat state it is an upper bound, not ρ_s itself.

The GP orbital is computed with the same equations as `gp_scf.jl` (solver imported from
`python/dmc_gp_delta.py`) and checked against `gp_cn_N40.csv`.

Output: `numpy_arrays_GP/gp_superfluid.csv` (0 … −200, N = 40 and N → ∞) and
`final_imag/gp_superfluid.pdf` (with VMC, DMC mixed and 2·DMC−VMC at N = 40 and N → ∞).

---

## 6. DMC vs GP at finite N — `python/dmc_gp_delta.py`

Δ(N) = E_DMC/N − E_GP(N)/N. E_GP(N) is a rigorous upper bound, so the exact Δ ≤ 0. With Kac
scaling, mean field is exact for N → ∞ and Δ → 0 as O(1/N).

- **E_GP(N)** in Python: same tridiagonal h[A] (`scipy.linalg.eigh_tridiagonal`) and Brent
  root (`scipy.optimize.brentq`), n_max = 40. Agrees with `gp_scf.jl` to 5×10⁻¹⁴.
- **DMC error**: Flyvbjerg–Petersen blocking of the per-step energy series
  (`numpy_arrays_DMC/.../energies/`) after the 20 % burn-in used by `run_dmc.jl`. The
  error is the largest blocked error over levels with ≥ 25 blocks (conservative plateau).
  The built-in `blocking_error` in `dmc_core.jl` stops at 32 blocks, which for short
  runs (τ = 8) lies before the plateau.
- **Fit** Δ = a + b/N (weighted). An intercept a significantly > 0 signals a DMC bias.
- Selects runs explicitly by walkers, steps and Δτ (default 1200 walkers, 32000 steps,
  Δτ = 10⁻³), not by file date.

```
python python/dmc_gp_delta.py --V0kac -150
python python/dmc_gp_delta.py --V0kac -25 -30 -40 -60 --N 15 20 25 30 40
```
Output: `numpy_arrays_DMC/gp_delta/constant_density_1.0/delta_V0kac<V>.{csv,md}`,
`final_imag/gp_delta_vs_invN_V0<V…>.pdf`, `final_imag/blocking_V0<V>_N<N>.pdf`.

## 7. Equilibration check — `python/dmc_equilibration.py`

For each run: energy trace E(τ) (0.1-τ block means), running averages from τ = 0 and from
the burn-in, means of the first and second half of the production data, and E/N for 1×,
2× and 4× the default burn-in, all with blocking errors. A real equilibration problem
shows up as E/N drifting down when more is discarded.

```
python python/dmc_equilibration.py                 # V0 = -150, N = 35 40
python python/dmc_equilibration.py --N 15 20 25 30
```
Output: `final_imag/equilibration_V0<V>.pdf`.

---

## 8. Momentum-space OBDM and GP comparison plots

| script | what it does | output |
|---|---|---|
| `python/obdm_momentum.py` | For VMC, DMC mixed and 2·DMC−VMC: symmetrise ρ(x,x'), average over translations by λ, rescale the trace to N (stored traces are off by up to ~3 %), report the most negative eigenvalue; Fourier transform ρ(k_j,k_j') = (dx²/L) Σ e^{−ik_j x} ρ(x,x') e^{ik_j' x'}; n(k); condensate block ρ_{nn'} = ρ(nk, n'k) (\|n\| ≤ 8), its two leading eigenvectors, depletion; parity projection (even/odd n) and overlaps with the GP orbitals. | `numpy_arrays_OBDM/momentum_N40_V0kac<V>.npz`, `summary_N40.csv` |
| `python/plots/plot_rho_k.py` | Heatmap of \|ρ_{nn'}\|/N | `final_imag/rho_kk_N40_V0<V>_<est>.pdf` |
| `python/plots/plot_gp.py` | E/N vs V0 (GP N = 40 and ∞, VMC, DMC); n(nk) QMC vs N c_n²; GP cat \|ρ_{nn'}\|; λ/N vs V0 (QMC vs GP cat); orbital overlaps per parity sector | `final_imag/gp_energy.pdf`, `gp_nk_*`, `gp_rho_cat_*`, `gp_lambda_N40.pdf`, `gp_overlap_N40.pdf` |
| `python/plots/plot_gp_g1.py` | GP cat prediction for \|g¹(x,x')\|² on one period | `final_imag/g1sq_map_GP_N40_V0<V>.pdf` |

OBDM caveat: the 2·DMC−VMC estimator is reliable for |V0| ≥ 80 only. Near threshold
(−25 … −60) the VMC, DMC mixed and extrapolated estimators disagree because of
trial-wavefunction bias; there the two-mode λ±/N is the better estimate of the
occupations. At −150 and −200 (old runs) the extrapolated matrix has λ₀ + λ₁ > N and
negative eigenvalues down to −3.5 % of N.

---

## 9. Order of execution

```
julia julia/gp/gp_scf.jl              # GP orbitals and tables   -> numpy_arrays_GP/
julia julia/gp/split_energy.jl        # needs gp_cn_N40.csv
julia julia/gp/two_mode.jl            # needs gp_cn_N40.csv (+ OBDM summary for QMC λ)
python python/obdm_momentum.py        # needs QMC ρ(x,x') and gp_cn_*.csv
python python/plots/plot_gp.py; python python/plots/plot_rho_k.py; python python/plots/plot_gp_g1.py
python python/plots/plot_split.py; python python/plots/plot_two_mode.py
python python/gp_superfluid.py        # needs leggett_analysis.py output for the QMC points
python python/dmc_gp_delta.py --V0kac ...;  python python/dmc_equilibration.py
```
Everything except the DMC runs takes seconds to minutes.
