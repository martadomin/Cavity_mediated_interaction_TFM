# VMC energies for the (N, V0) points used in DMC (trial energies and VMC reference).
# Same physics, parameters and output paths as run_vmc.jl (constant density), but only
# the energy observables are computed and saved:
#   energy, energy_sq, energy_kin, energy_pot  -> means
#   energy_array                               -> E_L / N at each measurement (for error bars)
#
# Usage (from the repository root):  julia -t auto julia/vmc/run_vmc_energy.jl

using NPZ, Printf, ProgressMeter, Base.Threads, Statistics

include(joinpath(@__DIR__, "..", "src", "VMCCore.jl"))

density_val     = 1.0
N_list          = [15, 20, 25, 30, 35, 40]
V0_kac_list     = collect(0.0:-10.0:-250.0)
n_sweeps        = 10^5
k_lat           = 2π
α               = 1.0
k_L             = 0.0
long_range, fermi_stat, reatto_chester, contact = true, false, false, false
overwrite       = true           # false: skip (N, V0) points whose energy file already exists

combos = [(N, V0_kac / N) for N in N_list for V0_kac in V0_kac_list]
sort!(combos, by = c -> -c[1])    # largest N first for better load balancing
println("Total (N, V0) combinations: $(length(combos)), threads: $(nthreads())")

base_dir = @sprintf "numpy_arrays_VMC/constant_density_%.1f" density_val

@threads for (num_part, V0) in combos
    L = num_part / density_val
    V0_dir = @sprintf "V0=%.3f" V0
    path(sub) = joinpath(base_dir, sub, V0_dir, @sprintf("%s_%d_%.3f_%.1f.npy", sub, num_part, V0, L))

    if !overwrite && isfile(path("energy"))
        println("N=$num_part, V0=$V0: energy already exists, skipping")
        continue
    end

    # Trial wavefunction: periodic spline, identical to run_vmc.jl. FastSpline2D is the
    # same B-spline evaluated without allocations (one instance per task, so thread-safe).
    psi_path = @sprintf "numpy_arrays_VMC/wavefunction/V0=%.3f/wavefunction_%.3f_%.1f.npy" V0 V0 1.0
    isfile(psi_path) || error("Wavefunction file not found: $psi_path")
    local_interp = FastSpline2D(periodic_wave_function(npzread(psi_path)))

    elapsed = @elapsed begin
        E, E_sq, _, _, _, acc, E_array, E_kin, E_pot, _, _ = metropolis(
            num_part, n_sweeps * num_part, 100, 1.0, L, V0, k_lat,
            local_interp, k_L, 0.0, α, long_range, fermi_stat, reatto_chester, contact;
            global_move_prob = 0.2, step_block = num_part, compute_zoom = false,
            compute_energy = true, compute_ssf = false, compute_g2 = false, compute_obdm = false
        )
    end

    for (sub, val) in [("energy", E), ("energy_sq", E_sq), ("energy_kin", E_kin),
                       ("energy_pot", E_pot), ("energy_array", E_array)]
        mkpath(dirname(path(sub)))
        npzwrite(path(sub), val)
    end
    @printf("DONE N=%2d V0=%8.3f  E=%.6f  E/N=%.6f  acc=%.3f  (%.0f s)\n", num_part, V0, E, E / num_part, acc, elapsed)
end
println("All VMC energies done.")
