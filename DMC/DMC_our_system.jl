using Random, Plots, Statistics, ProgressMeter, NPZ, Dierckx, Printf, Roots
using Base.Threads

include("metropolis_with_contact.jl")  # Load trial wavefunction and energy logic

# ------------------------------------------
# Utility Functions
# ------------------------------------------

"""
    get_periodic_difference(x1, x2, L)

Computes the minimal image distance between two points in a 1D periodic box of length `L`.
Returns a value in [-L/2, L/2].
"""
function get_periodic_difference(x1::Float64, x2::Float64, L::Float64)::Float64
    diff = x1 - x2
    return mod(diff + L/2, L) - L/2
end

"""
    map_to_unit_cell(x)

Maps coordinate `x` to unit cell [-0.5, 0.5), useful for evaluating periodic wavefunctions.
"""
function map_to_unit_cell(x::Float64)::Float64
    return mod(x + 0.5, 1.0) - 0.5
end

# Global constants for diffusion
const ħ = 1.0
const m = 1.0
const D = ħ^2 / (2 * m)

"""
    find_k_contact(L, a)

Solves k·tan(kL/2) = -1/a for the first positive root.
Used for Bethe-Peierls contact interaction in periodic boxes.
"""
function find_k_contact(L::Float64, a::Float64)::Float64
    function equation(k)
        return k * tan(k * L / 2) + 1/a
    end
    b = 1e-6
    c = π / L - 1e-3
    return find_zero(equation, (b, c), Bisection(); rtol=1e-10)
end

"""
    interpolated_wave_function(psi, x)

Builds a bicubic spline interpolation of a 2D wavefunction grid.
"""
function interpolated_wave_function(psi::Matrix{Float64}, x::Vector{Float64})::Spline2D
    return Spline2D(x, x, psi; kx=3, ky=3, s=0)
end

# ------------------------------------------
# Drift Force Calculation
# ------------------------------------------

"""
    calculate_drift_force!(drift_force, x_diff, ...)

Calculates the drift force ∇log(Ψ) for each particle in a walker.

# Inputs:
- `drift_force`: vector to be filled with computed forces (modified in-place)
- `x_diff`: particle positions for the current walker
- `num_particles`: number of particles
- `L`: box length
- `α`: exponent for Fermi term
- `ψ_2`: spline-interpolated wavefunction (Spline2D)
- `k_contact`: contact interaction parameter
- `long_range`, `fermi_stats`, `contact`: which components to include

# Output:
- Returns `true` if successful, `false` if the wavefunction is zero or invalid at any pair.

# Notes:
- This function accumulates contributions from all enabled wavefunction components.
- Long-range part is accumulated and added only once per particle.
"""
function calculate_drift_force!(drift_force::Vector{Float64}, x_diff::Vector{Float64}, num_particles::Int64, L::Float64, α::Float64, ψ_2, k_contact::Float64, long_range::Bool, fermi_stats::Bool, contact::Bool)::Bool
    fill!(drift_force, 0.0)
    for k in 1:num_particles
        grad_logψ_2 = 0.0
        for j in 1:num_particles
            if j != k
                if fermi_stats
                    xkj = get_periodic_difference(x_diff[k], x_diff[j], L)
                    drift_force[k] += 2 * α * (π / L) * cot(π / L * xkj)
                end
                if long_range
                    x1 = map_to_unit_cell(x_diff[k])
                    x2 = map_to_unit_cell(x_diff[j])
                    ψ_val = evaluate(ψ_2, x1, x2)
                    if ψ_val <= 0
                        return false
                    end
                    dψ = derivative(ψ_2, x1, x2, nux=1, nuy=0)
                    grad_logψ_2 += dψ / ψ_val
                end
                if contact
                    sgn = sign(get_periodic_difference(x_diff[k], x_diff[j], L))
                    xkj = abs(get_periodic_difference(x_diff[k], x_diff[j], L))
                    ϕ = k_contact * (xkj - L/2)
                    drift_force[k] += 2 * sgn * (-k_contact * tan(ϕ))
                end
            end
        end
        if long_range
            drift_force[k] += 2 * grad_logψ_2
        end
    end
    return true
end

# ------------------------------------------
# Kinetic and Potential Energy
# ------------------------------------------

"""
    calculate_kinetic_energy(x, ...)

Calculates the total kinetic energy of a walker using the log-derivative form.

# Inputs:
- `x_coord`: particle positions
- `num_particles`: number of particles
- `L`: box length
- `α`: Fermi statistics exponent
- `ψ_2`: spline-interpolated wavefunction
- `k_contact`: contact parameter
- `long_range`, `fermi_stats`, `contact`: toggles for components

# Output:
- Returns kinetic energy, or `Inf` if the spline wavefunction becomes nonpositive.

# Notes:
- This function computes the gradient and Laplacian of log(Ψ) for each component.
"""
function calculate_kinetic_energy(x_coord::Vector{Float64}, num_particles::Int64, L::Float64, α::Float64, ψ_2, k_contact::Float64, long_range::Bool, fermi_stats::Bool, contact::Bool)::Float64
    E_kin = 0.0
    for k in 1:num_particles
        grad_logψ = lapl_logψ = grad_logψ_2 = lapl_logψ_2 = grad_logψ_3 = lapl_logψ_3 = 0.0
        for j in 1:num_particles
            if j != k
                if fermi_stats
                    xkj = get_periodic_difference(x_coord[k], x_coord[j], L)
                    arg = π / L * xkj
                    grad_logψ += α * (π / L) * cot(arg)
                    lapl_logψ += -α * (π / L)^2 * csc(arg)^2
                end
                if long_range
                    x1 = map_to_unit_cell(x_coord[k])
                    x2 = map_to_unit_cell(x_coord[j])
                    ψ_val = evaluate(ψ_2, x1, x2)
                    if ψ_val <= 0
                        return Inf
                    end
                    dψ = derivative(ψ_2, x1, x2, nux=1, nuy=0)
                    d2ψ = derivative(ψ_2, x1, x2, nux=2, nuy=0)
                    grad_2 = dψ / ψ_val
                    lap_2 = d2ψ / ψ_val
                    grad_logψ_2 += grad_2
                    lapl_logψ_2 += lap_2 - grad_2^2
                end
                if contact
                    sgn = sign(get_periodic_difference(x_coord[k], x_coord[j], L))
                    xkj = abs(get_periodic_difference(x_coord[k], x_coord[j], L))
                    ϕ = k_contact * (xkj - L/2)
                    grad_logψ_3 += -k_contact * tan(ϕ) * sgn
                    lapl_logψ_3 += -k_contact^2 / cos(ϕ)^2
                end
            end
        end
        if fermi_stats
            E_kin += -0.5 * (lapl_logψ + grad_logψ^2)
        end
        if long_range
            E_kin += -0.5 * (lapl_logψ_2 + grad_logψ_2^2)
        end
        if contact
            E_kin += -0.5 * (lapl_logψ_3 + grad_logψ_3^2)
        end
    end
    return E_kin
end

"""
    calculate_potential_energy(x, N, V0)

Evaluates total pairwise interaction energy using a cosine cavity-mediated potential.
Only unique i<j pairs are summed.
"""
function calculate_potential_energy(x_new::Vector{Float64}, num_particles::Int64, V0::Float64)::Float64
    V = 0.0
    for a in 1:num_particles
        for b in (a+1):num_particles
            V += V0 * cos(2π * x_new[a]) * cos(2π * x_new[b])
        end
    end
    return V
end

"""
    local_energy(x_coord, num_part, psi_interp, ...)

Computes the local energy E = T + V for a single walker using finite differences.

# Arguments
- `x_coord`: Vector of particle positions.
- `num_part`: Number of particles.
- `psi_interp`: Interpolated 2D spline wavefunction for long-range interactions.
- `V0`: Interaction strength.
- `k_lat`: Lattice wavevector for potential term.
- `L`: System size.
- `k_L`: Wavevector used in trial wavefunction.
- `k_contact`: Parameter for contact interaction.
- `fermi_stats`, `reatto_chester`, `contact`: Flags to enable trial wavefunction components.

# Returns
Tuple `(E_total, E_kin, E_pot)`:
- Total local energy.
- Kinetic energy (via second derivative of log Ψ).
- Potential energy (pairwise cosine interaction).
"""
function local_energy(
    x_coord::Vector{Float64}, num_part::Int, psi_interp,
    V0::Float64, k_lat::Float64, L::Float64, k_L::Float64, k_contact::Float64,
    fermi_stats::Bool, reatto_chester::Bool, contact::Bool
)::Tuple{Float64, Float64, Float64}

    # Evaluate the trial wavefunction at current configuration
    psi_current = trial_wave_function(x_coord, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)

    # Avoid division by zero (walker killed upstream)
    if psi_current == 0.0
        return 0.0, 0.0, 0.0
    end

    # --- Kinetic Energy via finite differences ---
    kinetic = 0.0
    dx = 1e-5
    @inbounds for i in 1:num_part
        x_plus = copy(x_coord); x_plus[i] += dx
        x_minus = copy(x_coord); x_minus[i] -= dx

        psi_plus = trial_wave_function(x_plus, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        psi_minus = trial_wave_function(x_minus, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)

        kinetic -= 0.5 * (psi_plus - 2 * psi_current + psi_minus) / (dx^2 * psi_current)
    end

    # --- Pairwise Potential Energy ---
    potential = 0.0
    @inbounds for i in 1:num_part
        @inbounds for j in (i + 1):num_part
            potential += V0 * cos(k_lat * x_coord[i]) * cos(k_lat * x_coord[j])
        end
    end

    return kinetic + potential, kinetic, potential
end

# ------------------------------------------
# DMC Simulation Core Function
# ------------------------------------------

"""
    DMC_part_w_inter(...)

Runs a Diffusion Monte Carlo simulation with optional importance sampling,
branching, histogramming, and structure factor calculation.

# Arguments
- `num_particles`: Number of particles in the system.
- `num_steps`: Total number of DMC steps.
- `num_walkers`: Initial number of walkers.
- `num_bins`: Number of histogram bins for observables.
- `Δτ`: Imaginary time step.
- `E_T`: Initial trial energy.
- `α`: Fermi statistics interaction strength.
- `V0`: Interaction strength for the long-range potential.
- `L`: System size.
- `ψ_2`: Interpolated 2D trial wavefunction (Spline2D).
- `k_contact`: Parameter from Bethe-Peierls contact condition.
- `long_range`, `fermi_stats`, `contact`: Enable corresponding interaction components.
- `importance_sampling`: Whether to use drift forces.
- `previous_walkers`: If true, load initial walker positions from disk.

# Returns
Tuple `(energies, energy, hist_1d, hist_2d, SSF, walker_positions)`:
- `energies`: Array of local energy estimates per step.
- `energy`: Average energy over the last 70% of steps.
- `hist_1d`: Normalized 1D particle density histogram.
- `hist_2d`: Normalized 2D pair correlation histogram.
- `SSF`: Static structure factor as a function of momentum.
- `walker_positions`: Final walker configurations.
"""
function DMC_part_w_inter(num_particles::Int64, num_steps::Int64, num_walkers::Int64, num_bins::Int64, Δτ::Float64, E_T::Float64, α::Float64, V0::Float64, L::Float64, ψ_2, k_contact::Float64, long_range::Bool,  fermi_stats::Bool, contact::Bool, importance_sampling::Bool, previous_walkers::Bool)::Tuple{Vector{Float64}, Float64, Vector{Float64}, Matrix{Float64}, Vector{ComplexF64}, Matrix{Float64}}

    # === Basic parameters ===
    dx = L / num_bins
    offset = -L / 2
    num_walkers_initial = num_walkers # Store initial number for population control
    energies = zeros(Float64, num_steps) # Local energy at each time step

    # === Walker initialization ===
    walker_positions = zeros(num_walkers, num_particles)

    if previous_walkers
        # Load previous walkers from disk
        walker_positions = NPZ.npzread("numpy_arrays_DMC/energy_study_def/walkers/V0=$(V0)/walkers_$(num_walkers)_$(Δτ)_$(num_particles)_$(num_steps)_IS.npy")
        num_walkers = size(walker_positions, 1)
    else
        # Initialize walkers uniformly in the box
        for w in 1:num_walkers, p in 1:num_particles
            walker_positions[w, p] = rand() * L - L / 2
        end
    end

    # === Thread-local preallocated buffers ===
    thread_local_energies_storage = [Float64[] for _ in 1:nthreads()]
    thread_new_positions_storage = [Vector{Float64}() for _ in 1:nthreads()]
    thread_hist_1d_storage = [zeros(Float64, num_bins) for _ in 1:nthreads()]
    thread_hist_2d_storage = [zeros(Float64, num_bins, num_bins) for _ in 1:nthreads()]
    thread_SSF_storage = [zeros(ComplexF64, 5 * Int(L)) for _ in 1:nthreads()]
    thread_rand_nums_buffer = [zeros(num_particles) for _ in 1:nthreads()]

    # === k-vectors for static structure factor ===
    final_point_k = 5 * L
    k_vec = (2π / L) * collect(1:1:final_point_k)

    iter_val = Float64[]
    E_values = Float64[]

    # === Accumulated observables ===
    hist_1d_total = zeros(Float64, num_bins)
    hist_2d_total = zeros(Float64, num_bins, num_bins)
    SSF_total = zeros(ComplexF64, length(k_vec))
    total_walkers_sampled = 0

    diffusion_scale = sqrt(2 * D * Δτ)

    # === Main DMC loop ===
    @showprogress for step in 1:num_steps
        # Clear thread-local buffers
        for i in 1:nthreads()
            empty!(thread_local_energies_storage[i])
            empty!(thread_new_positions_storage[i])
            fill!(thread_hist_1d_storage[i], 0.0)
            fill!(thread_hist_2d_storage[i], 0.0)
            fill!(thread_SSF_storage[i], complex(0.0, 0.0))
        end

        # === Allocate temp arrays for walkers ===
        thread_drift_force = [zeros(num_particles) for _ in 1:nthreads()]
        thread_x_diff = [zeros(num_particles) for _ in 1:nthreads()]
        thread_x_new = [zeros(num_particles) for _ in 1:nthreads()]

        @threads for w in 1:num_walkers
            tid = threadid()
            current_drift_force = thread_drift_force[tid]
            current_x_diff = thread_x_diff[tid]
            current_x_new = thread_x_new[tid]
            rand_nums = thread_rand_nums_buffer[tid]
            current_hist_2d = thread_hist_2d_storage[tid]
            current_SSF = thread_SSF_storage[tid]

            x_current_walker = walker_positions[w, :]

            # --- Diffusion step ---
            randn!(rand_nums)
            @. current_x_diff = mod(x_current_walker + rand_nums * diffusion_scale + L/2, L) - L/2

            # --- Drift force (if IS) ---
            if importance_sampling
                success_drift = calculate_drift_force!(current_drift_force, current_x_diff, num_particles, L, α, ψ_2, k_contact, long_range, fermi_stats, contact)
            else
                current_drift_force .= 0.0
                success_drift = true
            end

            local E_loc::Float64
            if !success_drift
                E_loc = Inf
            else
                # --- New proposal ---
                if importance_sampling
                    @. current_x_new = mod(current_x_diff + D * current_drift_force * Δτ + L/2, L) - L/2
                else
                    current_x_new .= current_x_diff
                end

                # --- Energy evaluation ---
                E_kin = importance_sampling ? calculate_kinetic_energy(current_x_new, num_particles, L, α, ψ_2, k_contact, long_range, fermi_stats, contact) : 0.0
                V = long_range ? calculate_potential_energy(current_x_new, num_particles, V0) : 0.0
                E_loc = E_kin + V
            end

            push!(thread_local_energies_storage[tid], E_loc)

            # --- Branching step ---
            M = (!isinf(E_loc) && !isnan(E_loc)) ? floor(Int, exp(-(E_loc - E_T) * Δτ) + rand()) : 0

            if M > 0
                # --- Histogram update ---
                for a in 1:num_particles
                    bin = min(num_bins, max(1, Int(floor((current_x_new[a] + L/2) / L * num_bins)) + 1))
                    thread_hist_1d_storage[tid][bin] += M
                end

                # --- Pair density (2D histogram) ---
                for a in 1:num_particles, b in (a+1):num_particles
                    bx = min(num_bins, max(1, Int(floor((current_x_new[a] + L/2) / L * num_bins)) + 1))
                    by = min(num_bins, max(1, Int(floor((current_x_new[b] + L/2) / L * num_bins)) + 1))
                    current_hist_2d[bx, by] += M
                    current_hist_2d[by, bx] += M
                end

                # --- Structure factor ---
                for a in 1:num_particles, b in 1:num_particles
                    current_SSF .+= M .* exp.(im .* (current_x_new[a] - current_x_new[b]) .* k_vec)
                end

                # --- Store surviving walker ---
                for _ in 1:M
                    append!(thread_new_positions_storage[tid], current_x_new)
                end
            end
        end

        # === Aggregate step results ===
        local_energies = reduce(vcat, thread_local_energies_storage)
        new_positions_flat = reduce(vcat, thread_new_positions_storage)
        hist_1d_step = sum(thread_hist_1d_storage)
        hist_2d_step = sum(thread_hist_2d_storage)
        SSF_step = sum(thread_SSF_storage)

        if isempty(new_positions_flat)
            error("All walkers died. Try smaller Δτ or adjust E_T.")
        end

        # === Prepare next step ===
        num_walkers = length(new_positions_flat) ÷ num_particles
        walker_positions = reshape(new_positions_flat, (num_particles, num_walkers))'

        hist_1d_total .+= hist_1d_step
        hist_2d_total .+= hist_2d_step
        SSF_total .+= SSF_step
        total_walkers_sampled += num_walkers

        # === Population control ===
        valid_energies = filter(e -> !isinf(e) && !isnan(e), local_energies)
        if isempty(valid_energies)
            error("All walkers encountered unphysical configurations.")
        end

        E_loc_mean = mean(valid_energies)
        energies[step] = E_loc_mean
        E_T = E_loc_mean - (1 / Δτ) * log(num_walkers / num_walkers_initial)

        push!(iter_val, step)
        push!(E_values, E_T)
    end

    # === Normalize histograms and SSF ===
    total_hist_1d_weight = sum(hist_1d_total) * dx
    hist_1d_normalized = total_hist_1d_weight > 0 ? hist_1d_total ./ total_hist_1d_weight : zeros(Float64, num_bins)

    total_hist_2d_weight = sum(hist_2d_total) * dx^2
    hist_2d_normalized = total_hist_2d_weight > 0 ? hist_2d_total ./ total_hist_2d_weight : zeros(Float64, num_bins, num_bins)

    SSF_normalized = total_walkers_sampled > 0 && num_particles > 0 ? SSF_total ./ (num_particles * total_walkers_sampled) : zeros(ComplexF64, length(k_vec))

    # === Final energy ===
    start_idx = max(1, 3 * num_steps ÷ 10^3)
    energy = start_idx > length(energies) ? NaN : mean(energies[start_idx:end])

    @printf("Normalization of 1D histogram: %.4f\n", sum(hist_1d_normalized) * dx)
    @printf("Normalization of 2D histogram: %.4f\n", sum(hist_2d_normalized) * dx^2)

    return energies, energy, hist_1d_normalized, hist_2d_normalized, SSF_normalized, walker_positions
end
