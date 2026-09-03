# Quantum Monte Carlo study of systems interacting via long-range interactions mediated by a cavity

Code accompanying the manuscript *"Quantum Monte Carlo study of systems interacting via
long-range interactions mediated by a cavity"* (arXiv:[2601.10301](https://arxiv.org/abs/2601.10301),
submitted to Physical Review A).

The code studies 1D quantum gases — an ideal Bose gas, bosons with combined short- and
long-range interactions, and an ideal Fermi gas — subject to cavity-mediated,
infinite-range interactions in a periodic box, using Variational and Diffusion Monte
Carlo (VMC/DMC). It computes ground-state energies, density profiles, pair correlations
g⁽²⁾, and the superfluid fraction via the Leggett bound.

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
└── (data output directories, e.g. numpy_arrays_VMC_130726/, are created at
    runtime next to wherever a driver script is *run from* — see "Data paths"
    below; they are gitignored, not part of this tree)
```

`julia/src/` is the single source of truth for the trial wavefunction, particle moves,
and energy estimators; both `run_vmc.jl` and `dmc/dmc_core.jl` include it (via
`VMCCore.jl`) rather than each keeping their own copy, so a fix made once (e.g. to the
wavefunction or the kinetic-energy evaluation) is automatically picked up everywhere.
The five files under `src/` split along physical function, not just line count:
geometry/root-finding, wavefunction evaluation, MC move proposals, energy estimators,
and the sampler loop that ties them together — each can be read, tested, or modified on
its own. `VMCCore.jl` just includes them in dependency order; nothing else changed
functionally when the split happened (I diffed every resulting file against the
original to confirm).

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
want `numpy_arrays_VMC_.../`, `numpy_arrays_DMC/`, etc. to appear (e.g. `cd julia/vmc &&
julia --project=../ run_vmc.jl`), or update the `base_dir`/`psi_path`/`E_path` strings to
absolute paths if you run from elsewhere (e.g. from a SLURM submission script on
Correfoc).

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

Marta Domínguez-Navarro (corresponding author, ICCUB/UB, UPC), Grigori Astrakharchik
(UPC), Abel Rojo-Francàs (OIST), Bruno Juliá-Díaz (UB)
