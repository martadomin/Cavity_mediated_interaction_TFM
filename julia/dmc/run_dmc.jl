# DMC production run for bosons with the Kac-scaled cavity interaction.
#
# Standard workflow: Δτ and the number of walkers are fixed once, from the convergence
# study at the hardest point (convergence_dmc.jl), and used for every production point.
#
# For every (N, V0_kac) point:
#   1. VMC reference with the same Ψ_T and grids (energy, n(x), S(k), ρ₁).
#   2. Quadratic (second-order) DMC at the chosen Δτ (mixed estimators).
#   3. Extrapolated estimators 2·O_DMC − O_VMC for n(x), S(k), ρ₁.
#
# Usage (from the repository root):
#   julia -t auto julia/dmc/run_dmc.jl           # all points, one after the other
#   julia -t auto julia/dmc/run_dmc.jl 7         # only point 7 (0-based)
#   julia julia/dmc/run_dmc.jl --list            # print the list of points and exit
#   julia -t auto julia/dmc/run_dmc.jl --V0kac -150 --walkers 600 --tau 8 --vmc-sweeps 200000 [--dtau 5e-4]
#                                                # higher-statistics rerun of some couplings
#                                                # (new files: walkers/steps are in the names)
#   julia -t auto julia/dmc/run_dmc.jl --V0kac -25 --N 40 --walkers 1200 --tau 8 --branch
#                                                # single-branch Ψ_T from python/branch_trial.py: no Z₂ moves,
#                                                # output in numpy_arrays_DMC_branch/
#
# Inputs needed: numpy_arrays_VMC/wavefunction/V0=.../wavefunction_*.npy

ENV["GKSwstype"] = "100"     # headless plotting backend (Plots is loaded by the core files)

using Printf, NPZ, Statistics
using Base.Threads

include(joinpath(@__DIR__, "dmc_core.jl"))

# -----------------------------------------------
# Points and parameters
# -----------------------------------------------
N_list        = [15, 20, 25, 30, 35, 40]
V0_kac_list   = [-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0, -200.0]
density_val   = 1.0                     # L = N / density
Δτ            = 1e-3                    # convergence_dmc.jl, N=40, V0_kac=-100: bias < 0.1 (stat. error 0.45)
τ_total       = 3.0                     # imaginary time per DMC run
num_walkers   = 200                     # target population, from convergence_dmc.jl
burn_in_frac  = 0.2                     # equilibration
num_bins      = 100
obdm_every    = 10
z2_shift_prob = 0.1
quadratic     = true                    # second-order propagator
n_vmc_sweeps  = 10^5                    # VMC reference: sweeps (× N single-particle steps)
k_lat         = 2π
α             = 1.0
k_L           = 0.0

fermi_stats, reatto_chester, contact, long_range = false, false, false, true
importance_sampling = true

# --- command-line overrides: --V0kac a b c, --N a b, --walkers n, --tau t, --vmc-sweeps n ---
opts = Dict{String, Vector{String}}("points" => String[])   # bare integers before any flag = point indices
let key = "points"
    for a in ARGS
        if startswith(a, "--")
            key = a[3:end]; opts[key] = String[]
        else
            push!(opts[key], a)
        end
    end
end
haskey(opts, "V0kac")      && (V0_kac_list  = parse.(Float64, opts["V0kac"]))
haskey(opts, "N")          && (N_list       = parse.(Int, opts["N"]))
haskey(opts, "walkers")    && (num_walkers  = parse(Int, opts["walkers"][1]))
haskey(opts, "tau")        && (τ_total      = parse(Float64, opts["tau"][1]))
haskey(opts, "vmc-sweeps") && (n_vmc_sweeps = parse(Int, opts["vmc-sweeps"][1]))
haskey(opts, "dtau")       && (Δτ           = parse(Float64, opts["dtau"][1]))
branch_trial = haskey(opts, "branch")       # Ψ_T not invariant under the shift: no Z₂ moves in DMC or VMC
branch_trial && (z2_shift_prob = 0.0)
cat_trial = haskey(opts, "cat")             # Z₂-symmetric Ψ_T from python/cat_trial.py: same moves as the default trial
vmc_global_prob = branch_trial ? 0.0 : 0.2
point_args = opts["points"]

points = [(N, Vk) for Vk in V0_kac_list for N in N_list]

if "--list" in ARGS
    for (i, (N, Vk)) in enumerate(points)
        @printf("%3d  N=%2d  V0_kac=%7.1f  V0=%8.3f\n", i - 1, N, Vk, Vk / N)
    end
    println("$(length(points)) points")
    exit()
end

selected = isempty(point_args) ? points : [points[parse(Int, a) + 1] for a in point_args]
println("$(nthreads()) threads, running $(length(selected)) of $(length(points)) points, Δτ = $Δτ, " *
        "$num_walkers walkers, τ = $τ_total, VMC sweeps = $n_vmc_sweeps")

for (num_particles, V0_kac) in selected
    V0 = V0_kac / num_particles
    L = num_particles / density_val
    println("\n==================== N = $num_particles, V0_kac = $V0_kac (V0 = $V0), L = $L ====================")

    base = @sprintf "numpy_arrays_DMC%s/constant_density_%.1f" (branch_trial ? "_branch" : cat_trial ? "_cat" : "") density_val
    V0_dir = @sprintf "V0=%.3f" V0
    out(sub, name) = (p = joinpath(base, sub, V0_dir, name * ".npy"); mkpath(dirname(p)); p)
    tag_vmc = @sprintf("%d_%.3f_%.1f", num_particles, V0, L)
    num_steps = round(Int, τ_total / Δτ)
    core = @sprintf("%d_%.4f_%d_%d_%.3f_%.1f%s", num_walkers, Δτ, num_particles, num_steps, V0, L,
                    importance_sampling ? "_IS" : "")
    if isfile(joinpath(base, "obdm", V0_dir, "obdm_extrap_" * core * ".npy"))
        println("Already done (obdm_extrap_$core.npy exists), skipping")
        continue
    end

    # --- Trial wavefunction: periodic spline, identical to run_vmc.jl ---
    psi_path = branch_trial ?
        (@sprintf "numpy_arrays_VMC/wavefunction_branch/V0=%.3f/wavefunction_%.3f_%d.npy" V0 V0 num_particles) :
        cat_trial ?
        (@sprintf "numpy_arrays_VMC/wavefunction_cat/V0=%.3f/wavefunction_%.3f_%d.npy" V0 V0 num_particles) :
        (@sprintf "numpy_arrays_VMC/wavefunction/V0=%.3f/wavefunction_%.3f_%.1f.npy" V0 V0 1.0)
    isfile(psi_path) || error("Wavefunction file not found: $psi_path")
    ψ_2 = periodic_wave_function(NPZ.npzread(psi_path))
    k_contact = contact ? find_k_contact(L, -1.0) : 0.0

    # -----------------------------------------------
    # 1. VMC reference (same Ψ_T and grids as the DMC estimators)
    # -----------------------------------------------
    t_vmc = @elapsed begin
        E_vmc, _, SSF_vmc, hist_1d_vmc, _, _, E_arr_vmc, _, _, obdm_vmc, _, ncell_vmc = metropolis(
            num_particles, n_vmc_sweeps * num_particles, num_bins, 1.0, L, V0, k_lat,
            FastSpline2D(ψ_2), k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact;
            global_move_prob = vmc_global_prob, step_block = num_particles, compute_zoom = false,
            compute_energy = true, compute_ssf = true, compute_g2 = false, compute_obdm = true)
    end
    E_vmc_err = blocking_error(E_arr_vmc) * num_particles
    @printf("VMC: E = %.4f ± %.4f  (%.0f s)\n", E_vmc, E_vmc_err, t_vmc)
    NPZ.npzwrite(out("energy",  "energy_vmc_" * tag_vmc), [E_vmc, E_vmc_err])
    NPZ.npzwrite(out("hist_1d", "hist1d_vmc_" * tag_vmc), hist_1d_vmc)
    NPZ.npzwrite(out("SSF",     "SSF_vmc_"    * tag_vmc), SSF_vmc)
    NPZ.npzwrite(out("obdm",    "obdm_vmc_"   * tag_vmc), obdm_vmc)
    NPZ.npzwrite(out("n_cell",  "ncell_vmc_"  * tag_vmc), ncell_vmc)    # n(x mod λ)/n̄

    # -----------------------------------------------
    # 2. DMC
    # -----------------------------------------------
    println("\n--- DMC Δτ = $Δτ, $num_steps steps, $num_walkers walkers ---")
    t_dmc = @elapsed begin
        energies, energy, hist_1d, hist_2d, SSF, final_walkers, obdm, ncell = DMC_part_w_inter(
            num_particles, num_steps, num_walkers, num_bins, Δτ, E_vmc, α,
            V0, L, ψ_2, k_contact, long_range, fermi_stats, contact,
            importance_sampling, false;
            k_lat = k_lat, k_L = k_L, reatto_chester = reatto_chester,
            burn_in_frac = burn_in_frac, compute_obdm = true, obdm_every = obdm_every,
            z2_shift_prob = z2_shift_prob, quadratic = quadratic)
    end
    energy_err = blocking_error(energies[floor(Int, burn_in_frac * num_steps)+1:end])
    @printf("DMC: E = %.4f ± %.4f,  E/N = %.5f ± %.5f   (VMC: E/N = %.5f)  (%.0f s)\n",
            energy, energy_err, energy / num_particles, energy_err / num_particles, E_vmc / num_particles, t_dmc)

    NPZ.npzwrite(out("energies", "energies_" * core), energies)
    NPZ.npzwrite(out("energy",   "energy_"   * core), [energy, energy_err])
    NPZ.npzwrite(out("hist_1d",  "hist1d_"   * core), hist_1d)
    NPZ.npzwrite(out("hist_2d",  "hist2d_"   * core), hist_2d)
    NPZ.npzwrite(out("SSF",      "SSF_"      * core), SSF)
    NPZ.npzwrite(out("obdm",     "obdm_"     * core), obdm)
    NPZ.npzwrite(out("walkers",  "walkers_"  * core), final_walkers)
    NPZ.npzwrite(out("n_cell",   "ncell_"    * core), ncell)              # n(x mod λ)/n̄

    # -----------------------------------------------
    # 3. Extrapolated estimators 2·DMC − VMC
    # -----------------------------------------------
    NPZ.npzwrite(out("hist_1d", "hist1d_extrap_" * core), 2 .* hist_1d .- hist_1d_vmc)
    NPZ.npzwrite(out("SSF",     "SSF_extrap_"    * core), 2 .* SSF .- SSF_vmc)
    NPZ.npzwrite(out("obdm",    "obdm_extrap_"   * core), 2 .* obdm .- obdm_vmc)
    println("Saved to $base/*/$V0_dir/")
end
println("\nAll selected points done.")
