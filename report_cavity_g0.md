# 1D cavity Bose gas, g = 0: DMC vs mean field (GP) and the two-mode model

**Goal.** Show that DMC energies agree with single-orbital GP across the V0 range, that
fragmented states cost energy, and that the two macroscopic OBDM eigenvalues come from a
cat-type superposition of the two Z₂ branches rather than from Fock-type fragmentation.

**Main results**
- E_DMC(N) ≤ E_GP(N) within error bars at every converged point (V0 = −25 … −150,
  N = 15 … 40). The
  difference Δ = (E_DMC − E_GP)/N vanishes with N, cleanly as O(1/N) deep in the
  self-organised phase. Near threshold, N ≤ 40 is not yet asymptotic.
- Fragmenting the condensate between the two Z₂ configurations costs |V0|Θ'²/2 per
  particle (6.5 ε at V0 = −25, 55 ε at −150). DMC sits at the unfragmented end, on E_GP.
- A two-mode model (GP orbital + its λ/2-shifted partner) recovers 79–95 % of the energy
  beyond GP at V0 = −25 (N = 40 to 15), decreasing to 15–20 % at −60. Near threshold the
  soft cavity mode carries most of the correlation energy; deeper in the phase other modes
  dominate.
- For |V0| ≥ 80 the two macroscopic OBDM eigenvalues match the Z₂ cat prediction
  λ±/N = (1 ± s)/2 to ~0.01, with natural-orbital overlaps 0.98–1.00 and depletion ≤ 3 %
  (preliminary: older runs, 200–600 walkers, τ = 8; being rerun).
- The one DMC point above a variational bound (V0 = −150, N = 40) was population-control
  bias. With 2400 walkers the point is consistent with the bound (1.1σ); a residual bias
  at 2400 walkers can't be excluded yet (section 2).

Methods, scripts and file formats are documented in `julia/gp/README.md`.

---

## 1. Model, conventions and GP checks

H = Σ_i (−½ ∂²_i) + (V0/N) Σ_{i<j} cos(2πx_i) cos(2πx_j), with ħ = m = λ = 1. Energies are
in ε = ħ²/(mλ²), and ε_k = 2π² ≈ 19.739 ε. Density 1, so L = N, with periodic boundaries.
V0 is the Kac-scaled amplitude.

**Conventions verified in the QMC code**
1. Long-range term: `V0_bare = V0_kac/N` (`julia/dmc/run_dmc.jl`), summed over i < j with
   no extra ½ (`julia/src/energy.jl`). ✓
2. Units: kinetic term −½∇², cavity wave number 2π, so energies are in ε and lengths in λ. ✓
3. ρ(x,x') is stored in `numpy_arrays_{VMC,DMC}/constant_density_1.0/obdm/V0=<V0_kac/N>/`.
   Grid: 25 bins per λ (1000 × 1000 at N = 40), symmetric. Normalised so that
   Σ ρ(x,x) dx = N up to Monte Carlo noise of ≲ 3 %; the momentum analysis rescales the
   trace to N.
4. The earlier tabulated E_DMC/N values (e.g. −56.67 at V0 = −200) are 1/N extrapolations
   (N = 15–40), not N = 40 values.

**GP by self-consistent field (`julia/gp/gp_scf.jl`).** Single-orbital product state.
Self-consistency is reduced to one scalar equation f(A) = A − V0 (N−1)/N Θ₁(A) = 0,
solved with Brent's method. The energy is taken from the product-state formula, not from
the eigenvalue μ.

| check | result |
|---|---|
| V0 = 0: c_n = δ_{n0}, E = 0, Θ₁ = 0 | pass |
| threshold at \|V0\| = ε_k N/(N−1) = 20.245 (N = 40), 19.739 (N → ∞) | pass |
| n_max = 20 vs 40: \|ΔE/N\| < 10⁻¹⁰ | pass (7×10⁻¹⁵) |
| E/N < 0 above threshold | pass (the script's check also covers V0 = −15 and −20, below threshold at N = 40, where E = 0 exactly, so it prints FAIL; that comes from how the check is written) |
| independent Python mean-field code (N → ∞) | agrees to all printed digits (E/N = −58.15629 at V0 = −200) |
| E_DMC ≤ E_GP | holds within error bars at every converged point (sections 2–3); the old τ = 8 runs violated it at strong coupling |

The brief's "rough expectation" E_GP/N ≈ −55 at V0 = −200 was off: the N → ∞ value is −58.16.

---

## 2. DMC convergence (V0 = −150, N = 40)

All DMC runs use the second-order propagator with Z₂ shift moves and importance sampling.

**Walkers and Δτ at τ = 8** (exploratory):

| walkers | Δτ | E/N |
|---|---|---|
| 200 | 10⁻³ | −37.542 ± 0.076 |
| 600 | 10⁻³ | −37.659 ± 0.025 |
| 1200 | 10⁻³ | −37.731 ± 0.020 |
| 2400 | 10⁻³ | −37.692 ± 0.016 |
| 1200 | 5×10⁻⁴ | −37.638 ± 0.020 |

The run-to-run scatter (±0.05) was larger than the quoted errors. The energy has slow
fluctuations on a scale of about 1 ħ/ε, so at τ = 8 the error bars were underestimated by
about 2–3×. The production settings were therefore set to **1200 walkers, τ = 32,
Δτ = 10⁻³**.

**Error bars.** Flyvbjerg–Petersen blocking of the per-step energy after a 20 % burn-in.
The error quoted is the largest blocked error over levels with ≥ 25 blocks. At τ = 32 the
plateau is reached from block lengths of about 0.1 ħ/ε (`blocking_V0<V>_N40.pdf`).

**Equilibration** (`python/dmc_equilibration.py`). At V0 = −150 (N = 15–40) and −30
(N = 40) the running average settles within 1–2 ħ/ε. The first and second halves of the
production data agree within 0.3–1.0σ, and discarding 2× or 4× more initial time does not
lower E/N. No equilibration problem.

**Population-control bias.** With 1200 walkers, N = 35 and 40 at V0 = −150 sat above the
GP bound. The excess was stationary (it did not change with the discard). The walker test
settled it:

| V0 = −150, N = 40, τ = 32, Δτ = 10⁻³ | E/N | E/N − E_GP(N)/N |
|---|---|---|
| 1200 walkers | −37.6988 ± 0.0124 | +0.043 ± 0.012 (3.4σ above the bound) |
| **2400 walkers** | **−37.7316 ± 0.0092** | **+0.010 ± 0.009 (1.1σ)** |

So 1200 walkers shifted the energy up by about 0.03 ε per particle at this point (strong
coupling, large N). With 2400 walkers the point is consistent with the bound (1.1σ). If the
bias scales as 1/N_w, the 1200→2400 shift (0.033 ± 0.015) implies a residual of ~0.03 ε per
particle at 2400 walkers; a 1/N_w extrapolation gives E/N ≈ −37.765, i.e. Δ ≈ −0.02. A
4800-walker run is needed to confirm. The planned Δτ = 5×10⁻⁴ test was not run.

Runs at other V0 used 1200 walkers. The bias grows with |V0| and N, so the small Δ at −60
(≈ −0.003) and its 1/N fit may be affected. The bias pushes the energy *up*, so any
residual near threshold would make the DMC-below-two-mode gaps of section 4 slightly
larger, not create them.

---

## 3. DMC vs GP(N): Δ = (E_DMC − E_GP)/N

E_GP(N) is the energy of a product state, so it is a rigorous upper bound to the exact
energy and Δ ≤ 0. With Kac scaling, mean field becomes exact for N → ∞, so Δ → 0, expected
as O(1/N). All runs: 1200 walkers, τ = 32, Δτ = 10⁻³ (except where noted).

**V0 = −150**

| N | E_DMC/N | E_GP(N)/N | Δ | Δ/\|E_GP\| |
|---|---|---|---|---|
| 15 | −35.4420 ± 0.0087 | −35.4410 | −0.0010 ± 0.0087 | −3×10⁻⁵ |
| 20 | −36.3493 ± 0.0099 | −36.3590 | +0.0097 ± 0.0099 | +3×10⁻⁴ |
| 25 | −36.9062 ± 0.0100 | −36.9113 | +0.0051 ± 0.0100 | +1×10⁻⁴ |
| 30 | −37.2780 ± 0.0118 | −37.2800 | +0.0020 ± 0.0118 | +5×10⁻⁵ |
| 35 | −37.5245 ± 0.0126 | −37.5436 | +0.0191 ± 0.0126 | +5×10⁻⁴ |
| 40 | −37.6988 ± 0.0124 | −37.7415 | +0.0428 ± 0.0124 | +1×10⁻³ |
| 40 (2400 walkers) | −37.7316 ± 0.0092 | −37.7415 | +0.0100 ± 0.0092 | +3×10⁻⁴ |

Δ is zero within errors: the correlation energy beyond GP at −150 is below the resolution
of about 0.01 ε per particle. N = 35 with 1200 walkers is probably also slightly
population-biased (1.5σ).

**V0 = −25**

| N | E_DMC/N | E_GP(N)/N | Δ | Δ/\|E_GP\| |
|---|---|---|---|---|
| 15 | −0.3532 ± 0.0011 | −0.1597 | −0.1936 ± 0.0011 | −1.2 |
| 20 | −0.3223 ± 0.0009 | −0.1955 | −0.1268 ± 0.0009 | −0.65 |
| 25 | −0.3064 ± 0.0011 | −0.2185 | −0.0879 ± 0.0011 | −0.40 |
| 30 | −0.2984 ± 0.0011 | −0.2344 | −0.0640 ± 0.0011 | −0.27 |
| 40 | −0.2935 ± 0.0016 | −0.2549 | −0.0386 ± 0.0016 | −0.15 |

**V0 = −30**

| N | E_DMC/N | E_GP(N)/N | Δ | Δ/\|E_GP\| |
|---|---|---|---|---|
| 15 | −0.7992 ± 0.0017 | −0.7110 | −0.0882 ± 0.0017 | −0.12 |
| 20 | −0.8305 ± 0.0017 | −0.7866 | −0.0439 ± 0.0017 | −0.056 |
| 25 | −0.8624 ± 0.0021 | −0.8333 | −0.0292 ± 0.0021 | −0.035 |
| 30 | −0.8851 ± 0.0021 | −0.8649 | −0.0202 ± 0.0021 | −0.023 |
| 40 | −0.9206 ± 0.0020 | −0.9049 | −0.0157 ± 0.0020 | −0.017 |

**V0 = −40**

| N | E_DMC/N | E_GP(N)/N | Δ | Δ/\|E_GP\| |
|---|---|---|---|---|
| 15 | −2.4955 ± 0.0021 | −2.4714 | −0.0241 ± 0.0021 | −1.0×10⁻² |
| 20 | −2.6318 ± 0.0016 | −2.6194 | −0.0124 ± 0.0016 | −4.7×10⁻³ |
| 25 | −2.7187 ± 0.0020 | −2.7092 | −0.0094 ± 0.0020 | −3.5×10⁻³ |
| 30 | −2.7786 ± 0.0013 | −2.7696 | −0.0090 ± 0.0013 | −3.3×10⁻³ |
| 40 | −2.8481 ± 0.0016 | −2.8455 | −0.0026 ± 0.0016 | −9×10⁻⁴ |

**V0 = −60**

| N | E_DMC/N | E_GP(N)/N | Δ | Δ/\|E_GP\| |
|---|---|---|---|---|
| 15 | −7.2696 ± 0.0011 | −7.2613 | −0.0083 ± 0.0011 | −1.1×10⁻³ |
| 20 | −7.5536 ± 0.0010 | −7.5462 | −0.0074 ± 0.0010 | −1.0×10⁻³ |
| 25 | −7.7214 ± 0.0008 | −7.7182 | −0.0032 ± 0.0008 | −4×10⁻⁴ |
| 30 | −7.8369 ± 0.0009 | −7.8333 | −0.0036 ± 0.0009 | −5×10⁻⁴ |
| 40 | −7.9809 ± 0.0007 | −7.9776 | −0.0033 ± 0.0007 | −4×10⁻⁴ |

**Δ vs 1/N** (`gp_delta_vs_invN_V0-25_-30_-40_-60.pdf`, `gp_delta_vs_invN_V0-150.pdf`).
Weighted fits Δ = a + b/N:

| V0 | a | b | χ²/dof | O(1/N) → 0? |
|---|---|---|---|---|
| −150 (N ≤ 30) | consistent with 0 | — | — | yes (Δ = 0 within errors) |
| −60 | +0.0007 ± 0.0012 | −0.134 ± 0.029 | 2.1 | **yes** |
| −40 | +0.0076 ± 0.0024 | −0.45 ± 0.06 | 1.8 | linear fit rejected: a lies 3σ above zero, which is unphysical since Δ ≤ 0. The a + b/N + c/N² fit gives a = 0.002 ± 0.008, consistent with 0 → yes |
| −30 | intercept depends on the fit form | | 20 (linear) | not yet asymptotic |
| −25 | intercept depends on the fit form | | 7.4 (linear) | not yet asymptotic |

At −25 and −30, Δ falls steadily with N, but N·Δ is not constant (at −25 it goes from
−2.9 at N = 15 to −1.5 at N = 40). Adding c/N² gives intercepts of +0.036 ± 0.006 (−25)
and −0.031 ± 0.009 (−30), so the extrapolation is not reliable there. This is expected
close to the critical point, where the finite-N GP threshold ε_k N/(N−1) is itself still
moving. In the figure, fit lines are drawn only where a + b/N describes the data
(χ²/dof < 3); the other couplings are just joined by dotted lines.

---

## 4. Fragmented states and the two-mode model

### 4.1 Split Fock states (`julia/gp/split_energy.jl`)
From the GP orbital: φ± are the even/odd-n parts (normalised), and
χ_A,B = (φ⁺ ± φ⁻)/√2. These are orthonormal and map into each other under x → x + λ/2.
χ_A,B = (φ⁺ ± φ⁻)/√2 are the orthonormalised (Löwdin) sublattice orbitals; they differ
from the GP branches φ_A, φ_B, which overlap by s.
cos 2πx couples only even ↔ odd n, so ⟨χ_A|cos|χ_B⟩ = 0. The Fock state |n_A, N − n_A⟩
(n_A particles in χ_A, n_B = N − n_A in χ_B) then has the exact energy

E(n_A)/N = T + V0/(2N²) Θ'² [(n_A − n_B)² − N],  T = 2π² Σ n²(c^A_n)²,  Θ' = Σ c^A_n c^A_{n+1}.

Here c^A_n are the plane-wave coefficients of χ_A. All checks pass (orthogonality,
⟨χ_A|cos|χ_B⟩ = 0, symmetry n_A ↔ N − n_A, minimum at
n_A ∈ {0, N}, E(all A) ≥ E_GP).

| V0 (N = 40) | s | E_GP/N | E(all in A)/N | E(50/50)/N | gap/N = \|V0\|Θ'²/2 |
|---|---|---|---|---|---|
| −25 | 0.78 | −0.255 | 3.60 | 10.05 | 6.5 |
| −30 | 0.63 | −0.905 | 2.17 | 10.16 | 8.0 |
| −40 | 0.43 | −2.846 | −0.77 | 10.50 | 11.3 |
| −60 | 0.24 | −7.978 | −6.91 | 11.53 | 18.4 |
| −80 | 0.15 | −13.940 | −13.34 | 12.85 | 26.2 |
| −100 | 0.10 | −20.381 | −20.03 | 14.28 | 34.3 |
| −150 | 0.044 | −37.742 | −37.64 | 17.90 | 55.5 |

At V0 = −150 the gap is 54.9–55.5 ε per particle for all N = 15–40, so the total cost
grows ∝ N (≈ 2200 ε at N = 40). Measured from E_GP rather than from E(all in A), the cost of
the 50/50 state is larger still near threshold (10.3 ε vs 6.5 ε per particle at −25),
because E(all in A) > E_GP there.

- **E(all in A) is not the GP energy near threshold.** χ_A always has exactly half its
  weight on odd n, which costs at least ε_k/2 ≈ 9.9 ε in kinetic energy. E(all A) − E_GP is
  3.9 ε at −25 and 0.10 ε at −150.
- **DMC lies below the whole Fock curve,** on E_GP, at every V0: nowhere near the
  fragmented states (`split_energy_N40.pdf`).
- **The gap is an upper estimate of the fragmentation cost,** because the orbitals are not
  re-optimised for each n_A.

### 4.2 Two-mode ground state (`julia/gp/two_mode.jl`)
H projected on span{χ_A, χ_B} is an (N+1)×(N+1) tridiagonal matrix:
- diagonal: N T + (V0/2N) Θ'² [(2n_A − N)² − N],
- off-diagonal: t √((n_A+1)(N−n_A)), with t = ⟨χ_A|T|χ_B⟩ = 2π² Σ(−1)ⁿ n² (c^A_n)².

It is diagonalised in the symmetric sector (n_A ↔ N − n_A). Deep in the phase the
symmetric/antisymmetric splitting is ∝ t^N, below machine precision, and a plain
diagonalisation returns a single-branch mixture. span{χ_A, χ_B} = span{φ⁺, φ⁻} contains
the GP orbital, so the GP product state lies in this space and
**E_exact ≤ E2 ≤ E_GP(N)**.

**φ⁻ below threshold.** At V0 = −20 (and V0 = 0) the GP orbital at N = 40 is uniform and
has no odd part, so φ⁻ can't be built from it. It was chosen as **φ⁻ = √2 cos(2πx)**
(c⁻_{±1} = 1/√2), because it is:
- the lowest-energy orbital that is odd under the λ/2 shift;
- the one the cavity term couples to the constant (sin 2πx is also odd but uncoupled, and
  the ground state is even under x → −x);
- with φ⁺ = 1, the standard two-mode (Dicke) description of self-organisation
  (Nagy et al.).

This gives t = −π², and at V0 = 0 all particles are in φ⁺ (check passes).

**Checks.** V0 = 0: λ+ = N, E2 = 0. At −150: 98 % of the weight at n_A ∈ {0, N},
λ±/N = (1 ± s)/2 exactly, E2 = E_GP to 10⁻⁶. E2 ≤ E_GP everywhere. All pass.

**Energies at N = 40** (`two_mode_energy_N40.pdf`; open markers there are older τ = 8 runs):

| V0 | E_GP/N | E2/N | E_DMC/N | E_DMC − E2 | recovered by two-mode |
|---|---|---|---|---|---|
| −20 | 0.0000 | −0.0754 | −0.0764 ± 0.0005 * | −0.0010 ± 0.0005 | ≈ 99 % (provisional *) |
| −25 | −0.2549 | −0.2853 | −0.2935 ± 0.0016 | −0.0082 ± 0.0016 | 79 ± 3 % |
| −30 | −0.9049 | −0.9154 | −0.9206 ± 0.0020 | −0.0052 ± 0.0020 | 67 ± 9 % |
| −40 | −2.8455 | −2.8484 | −2.8481 ± 0.0016 | +0.0003 ± 0.0016 | (110 ± 65 %) |
| −60 | −7.9776 | −7.9780 | −7.9809 ± 0.0007 | −0.0029 ± 0.0007 | 13 ± 3 % |
| −150 | −37.7415 | −37.7415 | −37.7316 ± 0.0092 † | +0.010 ± 0.009 | — (beyond-GP energy unresolved) |

\* provisional: old 600-walker, τ = 8 run (V0 = −20 was not rerun), with error bars
underestimated by 2–3×. † 2400 walkers.

"Recovered" is (E_GP − E2)/(E_GP − E_DMC), the fraction of the beyond-GP energy captured by
two modes. Over all N:

| V0 | N = 15 | 20 | 25 | 30 | 40 |
|---|---|---|---|---|---|
| −25 | 95 % | 92 % | 88 % | 84 % | 79 % |
| −30 | 78 % | 74 % | 70 % | 76 % | 67 % |
| −40 | 46 % | 56 % | 55 % | 45 % | (110 ± 65 %) |
| −60 | 19 % | 14 % | 24 % | 17 % | 13 % |

**Interpretation.**
- **Just below threshold (−20),** GP gives the uniform state and misses the energy
  completely. The two-mode model recovers essentially all of it (provisional, see table):
  this is the quantum fluctuation of the soft cavity mode.
- **At −25 to −30,** two modes still capture most (67–95 %) of the beyond-GP energy.
  The remainder is resolved by the converged DMC (2.6–10σ) and comes from modes outside
  span{χ_A, χ_B}.
- **Deeper in the phase (−60),** the beyond-GP energy is small (≈ 0.003 ε per particle),
  and only 15–20 % of it is in the two-mode sector. Other modes dominate there.
- **At −150,** E2 = E_GP and the beyond-GP energy is below the DMC resolution.

**Occupations in the crossover.** For V0 = −25 to −60 the two-mode λ±/N are the best
estimate of the condensate occupations. The QMC estimators disagree there because of
trial-wavefunction bias (the VMC trial wavefunction stays almost unfragmented), so they are
listed for reference only:

| V0 (N = 40) | two-mode λ+/N, λ−/N | QMC DMC mixed λ₀, λ₁ | QMC 2·DMC − VMC λ₀, λ₁ |
|---|---|---|---|
| −25 | 0.903, 0.097 | 0.974, 0.028 | 0.957, 0.051 |
| −30 | 0.816, 0.184 | 0.914, 0.083 | 0.846, 0.155 |
| −40 | 0.718, 0.282 | 0.805, 0.194 | 0.661, 0.338 |
| −60 | 0.622, 0.378 | 0.658, 0.344 | 0.609, 0.390 |
| −80 | 0.576, 0.424 | 0.584, 0.421 | 0.586, 0.422 |
| −100 | 0.551, 0.449 | 0.539, 0.452 | 0.545, 0.437 |

QMC occupations at −25 … −60 come from the converged runs (1200 walkers, τ = 32). Those
at −80 and −100 come from the older runs (600 walkers, τ = 8). The converged runs give
almost the same occupations as the old τ = 8 runs (changes ≤ 0.013) and a smaller noise
floor: the most negative OBDM eigenvalue is −0.06 % of N for DMC mixed, against −0.17 %
before. The disagreement between the estimators near threshold is therefore systematic
(mixed-estimator bias from the trial wavefunction), not statistical.

The weights |ψ(n_A)|² (`two_mode_weights_N40.pdf`) concentrate at n_A ∈ {0, N} from
−60 on. At −25 they peak at n_A/N ≈ 0.18. That reflects the poor χ_A/χ_B basis near
threshold (s = 0.78), not fragmentation: a GP product state written in that basis spreads
the same way.

---

## 5. Superfluid fraction (`python/gp_superfluid.py`)

Leggett formula ρ_s/ρ = 1/(⟨n⟩⟨1/n⟩) over one unit cell, from the GP density:
- **single branch, n = |φ|²:** for a single-orbital state in 1D this is the exact
  phase-twist superfluid fraction;
- **Z₂-symmetric, n = ½[|φ(x)|² + |φ(x+½)|²]:** an upper bound for the cat state. It is
  the like-for-like comparison with the QMC values, which symmetrise the measured density
  under x → x + λ/2.

| V0 (N = 40) | GP single branch | GP Z₂-symmetric | QMC 2·DMC − VMC |
|---|---|---|---|
| −25 | 0.60 | 0.986 | 0.996 |
| −40 | 0.16 | 0.88 | 0.79 |
| −60 | 0.035 | 0.74 | 0.71 |
| −80 | 0.010 | 0.62 | 0.62 |
| −100 | 0.003 | 0.51 | 0.49 |
| −150 | 2.5×10⁻⁴ | 0.33 | 0.22 |

- The GP superfluid fraction of a single branch collapses within a few ε of threshold.
- The QMC Leggett bound agrees with the Z₂-symmetric GP bound from −60 to −100.
- **This agreement is a consistency check, not independent evidence that DMC samples both
  branches.** The QMC density is symmetrised under x → x + λ/2 in the analysis
  (`python/leggett_analysis.py`), so its Leggett bound is the two-branch one by
  construction. Testing whether the walkers actually populate both branches would require
  the unsymmetrised density.
- Near threshold the QMC estimators spread, because of trial-wavefunction bias.
- At −150 and −200 the QMC values come from old, unconverged runs with a density taken from
  the noisy OBDM diagonal. The Leggett bound is dominated by the density minima, so no
  physical conclusion is drawn there.

The QMC Leggett values have not been recomputed with the new τ = 32 runs.
(`gp_superfluid.pdf`, N = 40 and N → ∞.)

---

## 6. Momentum-space OBDM for |V0| ≥ 80 (N = 40)

From `python/obdm_momentum.py`: ρ(x,x') symmetrised, averaged over translations by λ and
rescaled to trace N, then Fourier-transformed. The condensate block ρ_{nn'} = ρ(n k, n' k)
(|n| ≤ 8) is diagonalised and projected onto parity sectors.
- **|V0| ≥ 80:** the QMC natural orbitals overlap the GP orbitals at 0.98–1.00 in each
  parity sector. λ±/N agrees with the GP cat prediction to about 0.01. Depletion outside
  the block is ≤ 3 %. The QMC n(n k_L) and |ρ_{nn'}| maps can be compared with the GP cat
  prediction N c_n² and c_n c_n' (1 + (−1)^{n−n'})/2 in `gp_nk_*`, `rho_kk_*` and
  `gp_rho_cat_*`. These figures have not been checked quantitatively in this report.
- **Near threshold (−25 … −60):** the OBDM is limited by trial-wavefunction bias. VMC, DMC
  mixed and 2·DMC − VMC disagree by up to 0.3 in λ₀/N (see section 4.2).
- **At −150 and −200:** the extrapolated matrix is unphysical (λ₀ + λ₁ > N, negative
  eigenvalues down to −3.5 % of N in the old runs). At −150 this persists with the
  converged 2400-walker, τ = 32 run: λ₀/N, λ₁/N = 0.553, 0.506 (sum 1.06, most negative
  eigenvalue −2.8 % of N). So the problem is systematic, not statistical. DMC mixed gives
  0.516, 0.493 (sum 1.008, most negative eigenvalue −0.7 % of N), close to the cat
  prediction 0.522, 0.478 but itself slightly outside the physical range. Neither is
  quoted as a result.

The OBDM analysis uses the converged runs where they exist (−25 … −60: 1200 walkers, τ = 32;
−150: 2400 walkers, τ = 32), and the older runs (200–600 walkers, τ = 8) elsewhere,
including −80 and −100.

---

## 7. Optional improvement: better trial wavefunction for DMC

**Status:** not implemented, and not triggered so far. With 2400 walkers the −150 point is
consistent with the bound (a residual bias is still possible, see section 2), and the
near-threshold DMC − E2 residuals are explained by modes beyond the two-mode space.
Documented for future work.

**Motivation.** For bosons in 1D the ground state is nodeless, so DMC converges to the exact
energy whatever the trial wavefunction Ψ_T is; the energies are not limited by a fixed-node
error. A better Ψ_T would instead
- reduce the local-energy fluctuations, and with them the population-control and time-step
  biases (the current Ψ_T gives VMC energies about 6 ε per particle above DMC at −150);
- reduce the mixed-estimator bias of the OBDM near threshold, which is the main payoff.

**Plan.**
1. *GP cat × Jastrow first.* Ψ_T = [Π_i φ(x_i) + Π_i φ(x_i + ½)] × (existing two-body
   Jastrow), with φ the GP orbital at the same (V0, N). It is positive by construction (no
   fixed-node error), Z₂-symmetric, and equals the two-mode state deep in the phase.
   Implemented as a new Ψ_T option, without changing the existing code paths. Tests: VMC
   energy and variance vs the current Ψ_T, Z₂ invariance, DMC energy and OBDM at V0 = −40
   and −150, N = 40.
2. *Two-mode state later, if step 1 isn't enough near threshold.*
   Ψ_T = Σ ψ(n_A) Sym[χ_A^{n_A} χ_B^{N−n_A}] (× Jastrow), evaluated as
   Π_i χ_B(x_i) · Σ_k ψ(k) e_k(r_1 … r_N) with r_i = χ_A(x_i)/χ_B(x_i) and e_k the
   elementary symmetric polynomials (O(N²) per move, plus gradients/Laplacians). Its
   positivity must be checked first: χ_A and χ_B can change sign, which would introduce
   nodes.

**Conditions for doing it.**
- the walker test and the Δτ = 5×10⁻⁴ test do **not** remove the excess above E_GP(N) at
  V0 = −150, N = 40 (so far not the case: with 2400 walkers the point is within 1.1σ of
  the bound; the 4800-walker run of section 10 would confirm); or
- step 3 shows **unresolved residuals near threshold** (not the case: DMC − E2 < 0 is
  allowed by the variational principle and explained by modes beyond χ_A, χ_B); or
- the near-threshold OBDM / occupations need to be quoted from QMC rather than from the
  two-mode model.

**Cost.** 1–2 days of coding and testing for step 1, then about 4–8 h of DMC per (V0, N).

---

## 8. Limitations

- Converged DMC (τ = 32) exists for V0 = −25, −30, −40, −60, −150. The points at −20, −35,
  −50, −80 and −100 come from older τ = 8 runs with underestimated error bars.
- V0 = −200 was dropped: the old runs there were far above the GP bound (0.4–0.6 ε per
  particle) and were not rerun.
- N ≤ 40 is not in the asymptotic regime at −25 and −30, so no N → ∞ extrapolation of Δ is
  quoted there.
- The superfluid analysis uses the older production runs. The OBDM uses converged runs
  at −25 … −60 and −150, but the OBDM / cat verification at −80 and −100 still rests on
  τ = 8 runs with 600 walkers.
- A residual population-control bias at 2400 walkers is possible (section 2: a 1/N_w
  extrapolation suggests ~0.03 ε per particle at V0 = −150, N = 40). The other couplings
  used 1200 walkers.
- The near-threshold occupations (−25 … −60) are a two-mode prediction, not validated by
  QMC.

---

## 9. Figures in `final_imag/`

**This analysis**

| figure | content | script | data |
|---|---|---|---|
| `gp_delta_vs_invN_V0-25_-30_-40_-60.pdf` | Δ vs 1/N near threshold. A fit line is drawn only where a + b/N fits (χ²/dof < 3) and its intercept is physical (a ≤ 2σ above 0): only −60. The other couplings, including −40 (a = +3σ), are joined by dotted lines | `python/dmc_gp_delta.py --V0kac -25 -30 -40 -60 --N 15 20 25 30 40` | `numpy_arrays_DMC/constant_density_1.0/energies/`, tables in `numpy_arrays_DMC/gp_delta/constant_density_1.0/delta_V0kac<V>.{csv,md}` |
| `gp_delta_vs_invN_V0-150.pdf` | Δ vs 1/N at −150 (1200 walkers) | `python/dmc_gp_delta.py --V0kac -150` | same |
| `blocking_V0{-25,-30,-40,-60,-150}_N40.pdf` | blocking plateau, N = 40 | `python/dmc_gp_delta.py` | same |
| `equilibration_V0-150.pdf`, `equilibration_V0-150_N15-30.pdf`, `equilibration_V0-30.pdf` | energy traces, running averages, discards | `python/dmc_equilibration.py [--V0kac V] [--N …]` | `numpy_arrays_DMC/.../energies/` |
| `split_energy_N40.pdf` | E(n_A)/N of split Fock states at −40, −80, −150 with E_GP(N) for each. E_DMC is drawn only at −150, where it sits on E_GP; at −40 and −80 only GP is shown | `julia/gp/split_energy.jl` + `python/plots/plot_split.py` | `numpy_arrays_GP/split_energy*.csv` |
| `two_mode_energy_N40.pdf` | E/N vs V0 (GP, two-mode, DMC) and (E − E_GP)/N | `julia/gp/two_mode.jl` + `python/plots/plot_two_mode.py` | `numpy_arrays_GP/two_mode.csv`, `two_mode_dense_N40.csv` |
| `two_mode_weights_N40.pdf` | \|ψ(n_A)\|² at −25, −40, −60, −150 | same | `numpy_arrays_GP/two_mode_weights_N40.csv` |
| `gp_superfluid.pdf` | ρ_s/ρ: GP single branch, GP Z₂-symmetric, QMC | `python/gp_superfluid.py` | `numpy_arrays_GP/gp_superfluid.csv`, `numpy_arrays_DMC/leggett_analysis/constant_density_1.0/leggett*.csv` |
| `gp_energy.pdf` | E/N vs V0: GP (N = 40, ∞), VMC, DMC | `python/plots/plot_gp.py` | `numpy_arrays_GP/gp_dense.csv`, `thermodynamic_limit` csv |
| `gp_nk_N40_V0{-40,-100}.pdf` | n(n k_L): QMC vs N c_n² | `python/plots/plot_gp.py` | `numpy_arrays_OBDM/momentum_*.npz`, `numpy_arrays_GP/gp_cn_N40.csv` |
| `gp_rho_cat_N40_V0{-40,-100}.pdf` | GP cat \|ρ_{nn'}\|/N | `python/plots/plot_gp.py` | `numpy_arrays_GP/gp_cn_N40.csv` |
| `gp_lambda_N40.pdf` | λ/N vs V0: GP cat λ±/N (Eq. for the symmetric two-branch state, full range) vs QMC estimators, shown only for −15 … −100 (−200 dropped; at −150 the extrapolated estimator is unphysical) | `python/plots/plot_gp.py` | `numpy_arrays_OBDM/summary_N40.csv`, `numpy_arrays_GP/gp_dense.csv` |
| `gp_overlap_N40.pdf` | overlaps of QMC natural orbitals with the GP φ± per parity sector, only above threshold (\|V0\| > 20.245 at N = 40, where φ⁻ exists) and for V0 ≥ −100 | `python/plots/plot_gp.py` | `numpy_arrays_OBDM/summary_N40.csv` |
| `rho_kk_N40_V0{-40,-100}_{lin,DMC}.pdf` | QMC \|ρ_{nn'}\|/N | `python/plots/plot_rho_k.py [--estimator DMC]` | `numpy_arrays_OBDM/momentum_*.npz` |
| `rho_kk_vs_cat_N40_V0-100_lin.pdf` | side by side, same colour scale: GP cat \|ρ_{nn'}\|/N and QMC 2·DMC − VMC at −100. Odd n − n' entries: 0 (cat), ≤ 3×10⁻³ (QMC); even entries agree to 0.006 | `python/plots/plot_rho_cat_compare.py` | `numpy_arrays_GP/gp_cn_N40.csv`, `numpy_arrays_OBDM/momentum_N40_V0kac-100.npz` |
| `g1sq_map_GP_N40_V0{-40,-100}.pdf` | GP cat \|g¹(x,x')\|² | `python/plots/plot_gp_g1.py` | `numpy_arrays_GP/gp_cn_N40.csv` |

The momentum data come from `python/obdm_momentum.py`. The GP inputs come from
`julia/gp/gp_scf.jl`.

**V0 = −200: unconverged, not used.** `gp_nk_N40_V0-200.pdf`, `gp_rho_cat_N40_V0-200.pdf`,
`rho_kk_N40_V0-200_lin.pdf` and `rho_kk_N40_V0-200_DMC.pdf` are still in `final_imag/`, but
V0 = −200 was dropped (section 8). These figures are kept only for the record.

**Earlier thesis figures (VMC / QMC, not part of this analysis), from `plots.ipynb`:**
`Sk_*`, `Skc_*`, `condensate_density_N*`, `g1sq_map_N{40,50}_*`, `lam0_*`, `rhos_*`.

---

## 10. Runs needed before publication

Not launched.

| run (N = 40) | settings | purpose |
|---|---|---|
| V0 = −80, −100 + OBDM | 2400 walkers, τ = 32, Δτ = 10⁻³ | converged check of the cat (λ±, overlaps, depletion) |
| V0 = −150 | 4800 walkers, τ = 32 | residual population-control bias |
| V0 = −60 | 2400 walkers, τ = 32 | protects the Δ fit and the two-mode % recovered |
| V0 = −20 | 1200 walkers, τ = 32 | below-threshold two-mode result |
