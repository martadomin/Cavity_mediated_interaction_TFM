using Printf, ProgressMeter, NPZ, DelimitedFiles
using Plots, LinearAlgebra, Random
using Base.Threads

include("DMC_our_system.jl") # Contains the DMC simulation function and utilities

print(Threads.nthreads(), " threads available.\n")

# -----------------------------------------------
# Global Simulation Parameters
# -----------------------------------------------
num_particles_array = [2]  # Particle numbers to loop over
Δτ = 1e-4                          # Imaginary time step
num_steps = 10^5                   # Total number of DMC steps
num_walkers = 1000                 # Number of walkers
α = 1.0                            # Fermi statistics strength
num_bins = 100                     # Binning for histogram observables

# -----------------------------------------------
# Simulation Flags
# -----------------------------------------------
fermi_stats = false
contact     = false
long_range  = true
importance_sampling = true
previous_walkers    = false

# -----------------------------------------------
# Outer Loop: Iterate over interaction strengths
# -----------------------------------------------
for V0 in collect(-10.0:20.0:10.0)
    println("Running simulations for V0 = $V0...\n")

    # Inner loop over particle numbers
    for num_particles in num_particles_array
        L = 5.0  # Fixed system size

        # --- Load or estimate initial trial energy ---
        E_path = @sprintf("numpy_arrays_VMC/constant_length_5.0/energy/V0=%.1f/energy_%d_%.1f_%.1f.npy", V0, num_particles, V0, L)
        if !isfile(E_path)
            # Estimate E_T as proportional to interaction strength and particle number squared
            E_T = -abs(V0)^2 * num_particles^2 / 2.0
        else
            E_T = NPZ.npzread(E_path)
        end
        println("Trial energy: ", E_T)

        # --- Define suffix for filenames ---
        IS_tag = importance_sampling ? "_IS" : ""

        # --- Base output directories ---
        base_folder_data = "numpy_arrays_DMC/energy_study_def"
        base_folder_imag = @sprintf "imag_DMC/energy_study_def/V0=%.1f" V0

        # Create base output folders if needed
        if !isdir(base_folder_data); mkpath(base_folder_data); end
        if !isdir(base_folder_imag); mkpath(base_folder_imag); end

        # --- Construct common filename stem ---
        filename_core = @sprintf("%d_%.4f_%d_%d%s", num_walkers, Δτ, num_particles, num_steps, IS_tag)

        # --- Output file paths ---
        energies_path   = base_folder_data * "/energies/V0=$(V0)/energies_" * filename_core * ".npy"
        energy_path     = base_folder_data * "/energy/V0=$(V0)/energy_"     * filename_core * ".npy"
        hist1d_path     = base_folder_data * "/hist_1d/V0=$(V0)/hist1d_"     * filename_core * ".npy"
        hist2d_path     = base_folder_data * "/hist_2d/V0=$(V0)/hist2d_"     * filename_core * ".npy"
        ssf_path        = base_folder_data * "/SSF/V0=$(V0)/SSF_"           * filename_core * ".npy"
        walker_path     = base_folder_data * "/walkers/V0=$(V0)/walkers_2_" * filename_core * ".npy"

        # --- Plot output paths ---
        histplot_path   = base_folder_imag * "/histogram_fs_"  * filename_core * ".png"
        energyplot_path = base_folder_imag * "/energy_"        * filename_core * ".png"
        ssfplot_path    = base_folder_imag * "/SSF_"           * filename_core * ".png"
        hist2dplot_path = base_folder_imag * "/hist2d_"        * filename_core * ".png"

        # Create output directories as needed
        for path in [energies_path, energy_path, hist1d_path, hist2d_path, ssf_path, walker_path,
                     histplot_path, energyplot_path, ssfplot_path, hist2dplot_path]
            dir_path = dirname(path)
            if !isdir(dir_path); mkpath(dir_path); end
        end

        println("Data will be saved to: $energies_path")
        println("Images will be saved to: $histplot_path")

        # -----------------------------------------------
        # Prepare trial wavefunction and contact strength
        # -----------------------------------------------
        k_contact = 0.0
        if contact
            k_contact = find_k_contact(L, -1.0)
        end

        ψ_2 = 0.0
        if long_range
            # Load 2D spline wavefunction ψ(x_i, x_j)
            L_unit = 1.0
            psi_path = @sprintf "numpy_arrays_VMC/wavefunction/V0=%.1f/wavefunction_%.1f_%.1f.npy" V0 V0 L_unit
            if !isfile(psi_path)
                error("Wavefunction file not found at path: $psi_path. Please generate the wavefunction data first.")
            end
            psi = NPZ.npzread(psi_path)
            x = collect(range(-L_unit/2, stop=L_unit/2, length= size(psi)[1] + 1))[1:end-1]
            ψ_2 = interpolated_wave_function(psi, x)
        end

        # -----------------------------------------------
        # Run DMC simulation
        # -----------------------------------------------
        println("Running DMC for $(num_particles) particles with Δτ = $(Δτ) and $(num_walkers) walkers... ")
        energies, energy, hist_1d, hist_2d, SSF, final_walkers = DMC_part_w_inter(
            num_particles, num_steps, num_walkers, num_bins, Δτ, E_T, α,
            V0, L, ψ_2, k_contact, long_range, fermi_stats, contact,
            importance_sampling, previous_walkers
        )
        println("Final energy for $num_walkers walkers with Δτ = $(Δτ): $(energy)")

        # -----------------------------------------------
        # Save output data arrays
        # -----------------------------------------------
        NPZ.npzwrite(energies_path, energies)
        NPZ.npzwrite(energy_path, energy)
        NPZ.npzwrite(hist1d_path, hist_1d)
        NPZ.npzwrite(hist2d_path, hist_2d)
        NPZ.npzwrite(ssf_path, SSF)
        NPZ.npzwrite(walker_path, final_walkers)

        # -----------------------------------------------
        # Save plots for observables
        # -----------------------------------------------
        bins_plot = range(-L/2, stop=L/2, length=num_bins + 1)
        bin_center = 0.5 .* (bins_plot[1:end-1] .+ bins_plot[2:end])

        # 1D density histogram
        plt_hist = plot(bin_center, hist_1d, label="DMC", lw=2)
        xlabel!("x"); ylabel!("Density")
        title!("Ground State Density of Particle in a Box (N = $num_particles)")
        savefig(plt_hist, histplot_path)

        # Energy evolution
        plt_energy = plot(energies, label="Estimated E₀", xlabel="Time step", ylabel="Energy", legend=:bottomright)
        savefig(plt_energy, energyplot_path)

        # Static Structure Factor
        k_vals_plot = (2π / L) * collect(1:5*L)
        plt_SSF = plot(k_vals_plot, real(SSF), label="Real part of SSF", xlabel="k", ylabel="SSF", legend=:bottomright, seriestype = :line, marker = :circle)
        savefig(plt_SSF, ssfplot_path)

        # 2D histogram of pair density
        plt_2d = heatmap(bin_center, bin_center, hist_2d, xlabel="x", ylabel="y", title="2D Histogram")
        savefig(plt_2d, hist2dplot_path)

        print(energyplot_path, " saved successfully.\n")
    end
end
# -----------------------------------------------