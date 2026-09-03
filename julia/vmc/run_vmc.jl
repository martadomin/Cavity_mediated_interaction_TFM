using NPZ, DelimitedFiles, Printf, ProgressMeter, Base.Threads, Plots, LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "VMCCore.jl"))

# 0 --> Density is kept constant
# 1 --> Length/Size of the system is kept constant
system_prop = 0

long_range      = true        
fermi_stat      = true    
reatto_chester  = false
contact         = false       
density_val     = 2.0
L_val           = 2.0

V0_scan_ranges = Dict(
    4 => [-3., -2., -1., 0, 1., 2., 3.],
)

combos = Tuple{Int, Float64}[]
for (N, V0_list) in V0_scan_ranges
    for V0 in V0_list
        push!(combos, (N, V0))
    end
end

println("Total (N, V0) combinations to simulate: $(length(combos))")
for (N, V0_list) in sort(collect(V0_scan_ranges), by = first)
    println("$N: ", V0_list, ",")
end

num_steps = 10^6
k_lat = 2π
α = 1.0
k_L = 0.0

@threads for (num_part, V0) in combos

    num_bins = 100

    filename = system_prop == 0 ? "constant_density" : "constant_length"
    filename *= !long_range ? "_no-interaction" : ""
    filename *= fermi_stat ? "_fermi-stat" : ""
    filename *= (!fermi_stat && reatto_chester) ? "_reatto-chester" : ""
    filename *= contact ? "_contact" : ""
    filename2 = system_prop == 0 ? density_val : L_val

    base_dir = @sprintf "numpy_arrays_VMC/%s_%.1f" filename filename2
    println(base_dir)

    if !isdir(base_dir)
        mkpath(base_dir)
    end

    for subdir in ["hist_1D", "hist_2D", "energy", "energy_sq", "SSF",
                   "energy_array", "energy_kin", "energy_pot"]
        dir_path = joinpath(base_dir, subdir, @sprintf "V0=%.3f" V0)
        if !isdir(dir_path)
            mkpath(dir_path)
        end
    end

    println("\n================ SIMULATION SUMMARY ================\n")
    println("Simulation type: $(system_prop == 0 ? "Constant density" : "Constant length")")
    if system_prop == 0
        println("Density: $density_val")
        println("Box length L (computed): $(num_part / density_val)")
    else
        println("Box length L: $L_val")
        println("Density (computed): $(num_part / L_val)")
    end
    println("Number of particles: $num_part")
    println("Monte Carlo steps: $num_steps")
    println("Interaction Strength V₀: $V0")
    println("Fermi statistics: $(fermi_stat ? "Yes" : "No")")
    println("Reatto-Chester wavefunction: $(reatto_chester ? "Yes" : "No")")
    println("Contact interaction: $(contact ? "Yes" : "No")")
    println("Output folder: $(filename)_$(filename2)")
    println("\n====================================================\n")

    elapsed = @elapsed begin

        L = system_prop == 0 ? num_part / density_val : L_val
        delta = 1.0

        local_interp = 0.0
        if long_range
            L_unit_cell = 1.0
            psi_path = @sprintf "numpy_arrays_VMC/wavefunction/V0=%.3f/wavefunction_%.3f_%.1f.npy" V0 V0 L_unit_cell
            if !isfile(psi_path)
                error("Wavefunction file not found: $psi_path. Please generate wavefunction data first.")
            end
            psi = npzread(psi_path)
            N_grid = size(psi)[1]
            x = collect(range(-L_unit_cell/2, stop=L_unit_cell/2, length=N_grid + 1))
            psi_full = zeros(N_grid + 1, N_grid + 1)
            psi_full[1:N_grid, 1:N_grid] .= psi
            psi_full[N_grid+1, 1:N_grid] .= psi[1, :]
            psi_full[1:N_grid, N_grid+1] .= psi[:, 1]
            psi_full[N_grid+1, N_grid+1]  = psi[1, 1]
            local_interp = interpolated_wave_function(psi_full, x)
        end

        k_contact = 0.0
        a = -1.0
        if contact
            k_contact = find_k_contact(L, a)
        end

        hist1D_path       = @sprintf "%s/hist_1D/V0=%.3f/hist_1D_%d_%.3f_%.1f.npy"           base_dir V0 num_part V0 L
        hist2D_path       = @sprintf "%s/hist_2D/V0=%.3f/hist_2D_%d_%.3f_%.1f.npy"           base_dir V0 num_part V0 L
        energy_path       = @sprintf "%s/energy/V0=%.3f/energy_%d_%.3f_%.1f.npy"             base_dir V0 num_part V0 L
        energy_sq_path    = @sprintf "%s/energy_sq/V0=%.3f/energy_sq_%d_%.3f_%.1f.npy"       base_dir V0 num_part V0 L
        ssf_path          = @sprintf "%s/SSF/V0=%.3f/SSF_%d_%.3f_%.1f.npy"                   base_dir V0 num_part V0 L
        energy_array_path = @sprintf "%s/energy_array/V0=%.3f/energy_array_%d_%.3f_%.1f.npy" base_dir V0 num_part V0 L
        energy_kin_path   = @sprintf "%s/energy_kin/V0=%.3f/energy_kin_%d_%.3f_%.1f.npy"     base_dir V0 num_part V0 L
        energy_pot_path   = @sprintf "%s/energy_pot/V0=%.3f/energy_pot_%d_%.3f_%.1f.npy"     base_dir V0 num_part V0 L

        E, E_sq, SSF, hist_1d, hist_2d, acceptance_ratio, E_array, E_kin, E_pot = metropolis(
            num_part, num_steps, num_bins, delta, L, V0, k_lat,
            local_interp, k_L, k_contact, α,
            long_range, fermi_stat, reatto_chester, contact
        )

        npzwrite(hist1D_path,       hist_1d)
        npzwrite(hist2D_path,       hist_2d)
        # npzwrite(energy_path,       E)
        # npzwrite(energy_sq_path,    E_sq)
        # npzwri(ssf_pateth,          SSF)
        # npzwrite(energy_array_path, E_array)
        # npzwrite(energy_kin_path,   E_kin)
        # npzwrite(energy_pot_path,   E_pot)

        println("\nAcceptance ratio for $num_part particles: $acceptance_ratio")
        println("Saved data for $num_part particles with V0 = $V0")
        println("Energy: $E")
        println("Energy squared: $E_sq")
        println("Kinetic Energy: $E_kin")
        println("Potential Energy: $E_pot")
        println("\n====================================================\n")
    end

    println("Elapsed time for N=$(num_part), V0=$(V0): $(elapsed) seconds")
    println("Simulation for N=$(num_part), V0=$(V0) completed successfully.")
end
