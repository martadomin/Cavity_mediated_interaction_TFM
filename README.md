# Quantum Monte Carlo study of systems interacting via long-range interactions mediated by a cavity

Code accompanying the manuscript *"Quantum Monte Carlo study of systems interacting via
long-range interactions mediated by a cavity"* (arXiv:[2601.10301](https://arxiv.org/abs/2601.10301),
published in Physical Review A DOI: [10.1103/6jcl-c1gt](https://doi.org/10.1103/6jcl-c1gt)).

The code studies 1D quantum gases: an ideal Bose gas, bosons with combined short- and
long-range interactions, and an ideal Fermi gas; each of them subject to cavity-mediated,
infinite-range interactions in a periodic box. The methods used are: Variational and Diffusion Monte
Carlo (VMC/DMC). It computes ground-state energies, density profiles $n(x)$, pair correlations
$g^{(2)}$, and the superfluid fraction via the Leggett bound.

## Repository structure

```
.
├── julia/
│   ├── src/                  # shared VMC/DMC core — single source of truth
│   │   ├── VMCCore.jl        #   top-level includer (fixes the include order)
│   │   ├── utils.jl          #   periodic geometry, contact-interaction root finding
│   │   ├── wavefunctions.jl  #   trial Ψ, log Ψ, and move-ratio updates
│   │   ├── moves.jl          #   particle-move proposals (local + global shift)
│   │   ├── energy.jl         #   local-energy estimators (log-space and direct)
│   │   └── sampler.jl        #   the VMC Metropolis loop (`metropolis`)
│   ├── vmc/
│   │   └── run_vmc.jl        # VMC driver: sets physical/simulation parameters, runs scans
│   └── dmc/
│       ├── dmc_core.jl       # drift force, DMC branching/walker logic (includes src/VMCCore.jl)
│       └── run_dmc.jl        # DMC driver: sets walker/time-step parameters, runs scans
├── python/
│   ├── imaginary_time_evolution.py   # generates the two-body cavity-mediated wavefunction
│   └── requirements.txt
└── (data output directories, e.g. numpy_arrays_VMC/, are created at
    runtime next to wherever a driver script is *run from* — see "Data paths"
    below; they are gitignored, not part of this tree)
```

`julia/src/` contains all the shared code (trial wavefunction, particle moves, energy
calculation) that both VMC and DMC use. Instead of each having its own copy for the
basic functions which are calculated in the same way (energy, and such), they both
include the same files from `src/`, so when you fix a bug or improve something, it
automatically works everywhere.

## Workflow

1. **Generate the two-body wavefunction.** Run `python/imaginary_time_evolution.py` for
   the desired cavity coupling `V0`. This produces the `.npy` grid consumed by
   `interpolated_wave_function` in `vmc_core.jl`.
2. **VMC.** Run `julia/vmc/run_vmc.jl` to sample the trial wavefunction and obtain
   variational energies, densities, and pair correlations for a scan of `(N, V0)`.
3. **DMC.** Run `julia/dmc/run_dmc.jl` for projector Monte Carlo refinement of the VMC
   estimates, using the same trial wavefunction as an importance-sampling guide.

### Data paths

Output arrays are organized by interaction type, particle number `N`, coupling `V0`, and
box length `L` (see the `base_dir`/`filename` construction near the top of each driver
script). **These paths are relative to the current working directory at the time you run
the script, not to the script's own location.** Run drivers from the directory where you
want `numpy_arrays_VMC/`, `numpy_arrays_DMC/`, etc. to appear (e.g. `cd julia/vmc &&
julia --project=../ run_vmc.jl`), or update the `base_dir`/`psi_path`/`E_path` strings to
absolute paths if you run from elsewhere.

## Environment setup

### Julia

From the `julia/` directory:

```julia
julia --project=. -e 'using Pkg; Pkg.add(["Dierckx", "StatsBase", "Plots", "LaTeXStrings", "ProgressMeter", "Roots", "NPZ"])'
```

This creates `Project.toml`/`Manifest.toml` pinning the exact package versions used.
Activate the environment before running a driver script:

```julia
julia --project=julia julia/vmc/run_vmc.jl
```

### Python

```bash
pip install -r python/requirements.txt
```

## Citation

If you use this code, please cite:

M. Domínguez-Navarro, A. Rojo-Francàs, B. Juliá-Díaz, and G. E. Astrakharchik,
"Quantum Monte Carlo study of systems interacting via long-range interactions mediated
by a cavity," Phys. Rev. A **114**, 033303 (2026).
https://doi.org/10.1103/6jcl-c1gt

## Authors

Marta Domínguez-Navarro (corresponding author, UPC), Grigori Astrakharchik
(UPC), Abel Rojo-Francàs (OIST), Bruno Juliá-Díaz (ICCUB & UB)
