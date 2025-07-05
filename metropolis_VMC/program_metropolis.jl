using NPZ, DelimitedFiles, Printf, ProgressMeter, Base.Threads

# Load the Metropolis algorithm implementation with contact interaction
include("metropolis_with_contact.jl")

# ------------------- SIMULATION PARAMETERS -------------------

# System properties:
# system_prop = 0 → constant density
# system_prop = 1 → constant length
system_prop = 0

# Physical and simulation parameters
long_range      = true           # Enable cavity-mediated long-range interaction
fermi_stat      = true           # Enable Fermi statistics (antisymmetric wavefunction)
reatto_chester  = false          # Use Reatto-Chester form (only if fermi_stat is false)
contact         = false          # Enable contact interaction
density_val     = 1.0            # Target density (used only if constant density)
L_val           = 2.0            # Box length (used only if constant length)
V0_array = [-7.0, -2.0]          # Array of interaction strengths V₀ to simulate
println("V0_array: ", V0_array)

# Particle numbers to simulate (depending on whether V0 is attractive or repulsive)
num_part_array_neg  = [7]                  # For V₀ < 0
num_part_array_pos  = [19, 20, 21]         # For V₀ ≥ 0

# Number of Metropolis steps per simulation
num_steps       = 10^8

# Lattice momentum and interaction parameter
k_lat           = 2π
α               = 1.0
k_L             = 0.0                      # Only used in Reatto-Chester form

# ------------------- MAIN LOOP OVER V₀ VALUES -------------------

@threads for V0 in V0_array
    # Choose particle numbers depending on sign of V₀
    num_part_array = V0 < 0 ? num_part_array_neg : num_part_array_pos

    num_bins = 150  # Number of bins for histograms

    # --- Construct filename and directory path based on simulation parameters ---
    filename = system_prop == 0 ? "constant_density" : "constant_length"
    filename *= !long_range ? "_no-interaction" : ""
    filename *= fermi_stat ? "_fermi-stat" : ""
    filename *= (!fermi_stat && reatto_chester) ? "_reatto-chester" : ""
    filename *= contact ? "_contact" : ""
    filename2 = system_prop == 0 ? density_val : L_val

    # Output directory
    base_dir = @sprintf "numpy_arrays_VMC/%s_%.1f" filename filename2
    print(base_dir)

    # Create main directory if it doesn't exist
    if !isdir(base_dir)
        mkpath(base_dir)
    end

    # Create subdirectories for different types of data
    for subdir in ["hist_1D", "hist_2D", "energy", "energy_sq", "SSF", "energy_array"]
        dir_path = joinpath(base_dir, subdir, @sprintf "V0=%.1f" V0)
        if !isdir(dir_path)
            mkpath(dir_path)
        end
    end

    # --- Print simulation summary ---
    println("\n================ SIMULATION SUMMARY ================\n")
    println("Simulation type: $(system_prop == 0 ? "Constant density" : "Constant length")")
    if system_prop == 0
        println("Density: $density_val")
        println("Box length L (computed): $(num_part_array[1] / density_val)")
    else
        println("Box length L: $L_val")
        println("Density (computed): $(num_part_array[1] / L_val)")
    end
    println("Number of particles: $num_part_array")
    println("Monte Carlo steps: $num_steps")
    println("Interaction Strength V₀: $V0")
    println("Fermi statistics: $(fermi_stat ? "Yes" : "No")")
    println("Reatto-Chester wavefunction: $(reatto_chester ? "Yes" : "No")")
    println("Contact interaction: $(contact ? "Yes" : "No")")
    println("Output folder: $(filename)_$(filename2)")
    println("\n====================================================\n")

    # --- Run simulations for all selected particle numbers ---
    elapsed = @elapsed begin
        @threads for num_part in num_part_array

            # Compute box length based on constant density or fixed length
            L = system_prop == 0 ? num_part / density_val : L_val

            delta = L/2                 # Maximum move size
            psi_interp = 0.0            # Interpolated wavefunction (if needed)

            # Load long-range wavefunction if enabled
            if long_range
                L_unit_cell = 1.0
                psi_path = @sprintf "numpy_arrays_VMC/wavefunction/V0=%.1f/wavefunction_%.1f_%.1f.npy" V0 V0 L_unit_cell
                if !isfile(psi_path)
                    error("Wavefunction file not found at path: $psi_path. Please generate the wavefunction data first.")
                end
                psi = npzread(psi_path)
                x = collect(range(-L_unit_cell/2, stop=L_unit_cell/2, length= size(psi)[1] + 1))[1:end-1]
                psi_interp = interpolated_wave_function(psi, x)
            end

            # Compute contact interaction strength if enabled
            k_contact = 0.0
            a = -1.0
            if contact
                k_contact = find_k_contact(L::Float64, a::Float64)
            end

            # --- Output file paths ---
            hist1D_path = @sprintf "%s/hist_1D/V0=%.1f/hist_1D_%d_%.1f_%.1f.npy" base_dir V0 num_part V0 L
            hist2D_path = @sprintf "%s/hist_2D/V0=%.1f/hist_2D_%d_%.1f_%.1f.npy" base_dir V0 num_part V0 L
            energy_path = @sprintf "%s/energy/V0=%.1f/energy_%d_%.1f_%.1f.npy" base_dir V0 num_part V0 L
            energy_sq_path = @sprintf "%s/energy_sq/V0=%.1f/energy_sq_%d_%.1f_%.1f.npy" base_dir V0 num_part V0 L
            ssf_path = @sprintf "%s/SSF/V0=%.1f/SSF_%d_%.1f_%.1f.npy" base_dir V0 num_part V0 L
            energy_array_path = @sprintf "%s/energy_array/V0=%.1f/energy_array_%d_%.1f_%.1f.npy" base_dir V0 num_part V0 L
            energy_kin_array_path = @sprintf "%s/energy_kin_array/V0=%.1f/energy_kin_array_%d_%.1f_%.1f.npy" base_dir V0 num_part V0 L

            # --- Run the Metropolis VMC simulation ---
            E, E_sq, SSF, hist_1d, hist_2d, acceptance_ratio, E_array, E_kin_array = metropolis(
                num_part, num_steps, num_bins, delta, L, V0, k_lat,
                psi_interp, k_L, k_contact, α, long_range, fermi_stat, reatto_chester, contact
            )

            # Save simulation results
            npzwrite(hist1D_path, hist_1d)
            npzwrite(hist2D_path, hist_2d)
            npzwrite(energy_path, E)
            npzwrite(energy_sq_path, E_sq)
            npzwrite(ssf_path, SSF)
            npzwrite(energy_array_path, E_array)
            npzwrite(energy_kin_array_path, E_kin_array)

            # Print result summary
            println(hist1D_path)
            println("\nAcceptance ratio for $num_part particles: $acceptance_ratio")
            println("Saved data for $num_part particles with V0 = $V0")
            println("Energy: $(E)")
            println("Energy squared: $E_sq")
            println("\n====================================================\n")
        end
    end

    # Print total elapsed time
    println("Elapsed time for V0 = $(V0): $(elapsed) seconds")
    println("Simulation for V0 = $(V0) completed successfully.")
    println("Data saved in numpy_arrays_VMC/$filename/$filename2/")
end
