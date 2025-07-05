using Random, LinearAlgebra, Dierckx, StatsBase, Plots, Base.Threads, LaTeXStrings, ProgressMeter, Roots

# ------------------------------------------
# Utility Functions
# ------------------------------------------
"""
    get_periodic_difference(x1::Float64, x2::Float64, L::Float64) -> Float64

Computes the minimum-image (periodic) difference between two points in a 1D periodic box.

# Input:
- `x1::Float64`: Position of the first point.
- `x2::Float64`: Position of the second point.
- `L::Float64`: Length of the periodic box.

# Output:
- `Float64`: The difference `(x1 - x2)`, mapped to the interval [-L/2, L/2].

# Notes
Useful for applying periodic boundary conditions and minimum-image convention in simulations.
"""
function get_periodic_difference(x1::Float64, x2::Float64, L::Float64)::Float64
    diff = x1 - x2
    # Shift to [0, L), then to [-L/2, L/2]
    return mod(diff + L/2, L) - L/2
end


"""
    map_to_unit_cell(x::Float64) -> Float64

Maps a coordinate `x` to the canonical unit cell [-0.5, 0.5) for systems with unit length.

# Input:
- `x::Float64`: Coordinate to map.

# Output:
- `Float64`: `x` mapped to [-0.5, 0.5).

# Notes
Used to enforce periodicity and symmetry in simulations with unit box length.
"""
function map_to_unit_cell(x::Float64)::Float64
    return mod(x + 0.5, 1.0) - 0.5
end

"""
    find_k_contact(L::Float64, a::Float64) -> Float64

Finds the first positive solution `k` to the transcendental equation:
    k * tan(k L / 2) = -1/a

# Input:
- `L::Float64`: Length of the periodic box.
- `a::Float64`: Scattering length (contact interaction parameter).

# Output:
- `Float64`: The first positive solution `k` (in the interval (0, π/L)).

# Notes
This is used to construct the Bethe-Peierls pair wave function for contact interactions with periodic boundary conditions.
"""
function find_k_contact(L::Float64, a::Float64)::Float64
    function equation(k)
        return k * tan(k * L / 2) + 1/a
    end
    b = 1e-6
    c = π / L - 1e-3
    return find_zero(equation, (b, c), Bisection(); rtol=1e-10)
end

# ------------------------------------------
# Creating the Wavefunction
# ------------------------------------------

"""
    interpolated_wave_function(psi::Matrix{Float64}, x::Vector{Float64}) -> Spline2D

Interpolates a two-body wave function matrix `psi` onto a finer 2D grid using bicubic splines.

# Input:
- `psi::Matrix{Float64}`: 2D matrix of wave function values, defined on a grid of points.
- `x::Vector{Float64}`: Grid points corresponding to both axes of `psi`.

# Output:
- `Spline2D`: A bicubic spline object for smooth evaluation at arbitrary (x1, x2).

# Usage:
Constructs a continuous representation of the numerically obtained two-body wave function,
which can be efficiently evaluated during Monte Carlo sampling.
"""
function interpolated_wave_function(psi::Matrix{Float64}, x::Vector{Float64})::Spline2D
    return Spline2D(x, x, psi; kx=4, ky=4, s=0)
end

"""
    trial_wave_function(x_coord, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact) -> Float64

Evaluates the total trial wave function for a set of particle positions, possibly including 
Fermi statistics, Reatto-Chester (Jastrow) correlations, contact interaction, and/or long-range cavity-mediated interaction.

# Input:
- `x_coord::Vector{Float64}`: Vector of particle positions.
- `num_part::Int`: Number of particles.
- `psi_interp`: Interpolated two-body wave function (e.g. Spline2D object).
- `L::Float64`: Box length (system size).
- `k_L::Float64`: Reatto-Chester parameter (Jastrow exponent).
- `k_contact::Float64`: Parameter for the contact (Bethe-Peierls) term.
- `α::Float64`: Exponent for Fermi statistics factor.
- `long_range::Bool`: Whether to include long-range (cavity-mediated) term.
- `fermi_stats::Bool`: Whether to include Fermi statistics factor.
- `reatto_chester::Bool`: Whether to include Reatto-Chester factor.
- `contact::Bool`: Whether to include contact interaction factor.

# Output:
- `Float64`: Value of the total trial wave function for the given configuration.

# Notes
Loops over all unique pairs (i < j) and multiplies together the selected two-body terms.
The long-range term uses the interpolated two-body wave function.
"""
function trial_wave_function(
    x_coord::Vector{Float64}, num_part::Int, psi_interp,
    L::Float64, k_L::Float64, k_contact::Float64, α::Float64,
    long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool
)::Float64
    Psi_tot = 1.0
    @inbounds for i in 1:num_part
        @inbounds for j in (i + 1):num_part
            if fermi_stats
                # Fermi statistics: node at coincident positions, exponent α
                Psi_tot *= (sin((π/L) * (x_coord[i] - x_coord[j])))^α
            end
            if reatto_chester
                # Jastrow-like (Reatto-Chester) factor
                Psi_tot *= abs(sin((π/L) * (x_coord[i] - x_coord[j])))^k_L
            end
            if contact
                # Bethe-Peierls contact interaction
                dist_mod = abs(get_periodic_difference(x_coord[i], x_coord[j], L)) - L/2
                Psi_tot *= cos(k_contact * dist_mod)
            end
            if long_range
                # Cavity-mediated long-range interaction, interpolated on unit cell
                x_coord_per_1 = map_to_unit_cell(x_coord[i])
                x_coord_per_2 = map_to_unit_cell(x_coord[j])
                Psi_tot *= evaluate(psi_interp, x_coord_per_1, x_coord_per_2)
            end
        end
    end
    return Psi_tot
end

"""
    update_wave_function_after_move(x_coord, x_new, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact) -> Float64

Computes the ratio Ψ_new/Ψ_old of trial wave functions for a move, 
efficiently updating only the necessary pair terms.

# Input:
- `x_coord::Vector{Float64}`: Old configuration (positions).
- `x_new::Vector{Float64}`: New configuration (positions; typically only one coordinate differs).
- `num_part::Int`: Number of particles.
- `psi_interp`: Interpolated two-body wave function (Spline2D).
- `L::Float64`, `k_L::Float64`, `k_contact::Float64`, `α::Float64`: Same as in `trial_wave_function`.
- `long_range::Bool`, `fermi_stats::Bool`, `reatto_chester::Bool`, `contact::Bool`: Term selection.

# Output:
- `Float64`: The ratio Ψ_new / Ψ_old for the proposed move.

# Notes
Loops only over pairs involving the moved particle(s) for efficiency.
"""
function update_wave_function_after_move(
    x_coord::Vector{Float64}, x_new::Vector{Float64}, num_part::Int,
    psi_interp, L::Float64, k_L::Float64, k_contact::Float64, α::Float64,
    long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool
)::Float64
    psi_ratio = 1.0
    @inbounds for i in 1:num_part
        if x_coord[i] != x_new[i]
            @inbounds for j in 1:num_part
                if i != j
                    if fermi_stats
                        # Fermi statistics: node at coincident positions, exponent α
                        psi_ratio *= (sin((π/L) * (x_new[i] - x_new[j])) / sin((π/L) * (x_coord[i] - x_coord[j])))^α
                    end
                    if reatto_chester
                        # Jastrow-like (Reatto-Chester) factor
                        psi_ratio *= abs(sin((π/L) * (x_new[i] - x_new[j])))^k_L / abs(sin((π/L) * (x_coord[i] - x_coord[j])))^k_L
                    end
                    if contact
                        # Bethe-Peierls contact interaction
                        dist_mod_new = abs(get_periodic_difference(x_new[i], x_new[j], L)) - L/2
                        dist_mod_old = abs(get_periodic_difference(x_coord[i], x_coord[j], L)) - L/2
                        psi_ratio *= cos(k_contact * dist_mod_new) / cos(k_contact * dist_mod_old)
                    end
                    if long_range
                        # Cavity-mediated long-range interaction, interpolated on unit cell
                        x_new_per_1 = map_to_unit_cell(x_new[i])
                        x_new_per_2 = map_to_unit_cell(x_new[j])
                        x_coord_per_1 = map_to_unit_cell(x_coord[i])
                        x_coord_per_2 = map_to_unit_cell(x_coord[j])
                        psi_ratio *= evaluate(psi_interp, x_new_per_1, x_new_per_2) / evaluate(psi_interp, x_coord_per_1, x_coord_per_2)
                    end
                end
            end
        end
    end
    return psi_ratio
end

# ------------------------------------------
# System Initialization and Particle Moves
# ------------------------------------------

"""
    random_initial_config(num_part::Int, L::Float64) -> Vector{Float64}

Generates a random initial configuration of `num_part` particles uniformly distributed in a 1D periodic box of length `L`.

# Input:
- `num_part::Int`: Number of particles.
- `L::Float64`: Length of the simulation box.

# Output:
- `Vector{Float64}`: Positions of all particles, each in the interval [-L/2, L/2].

# Notes
Particle positions are initialized randomly and independently with uniform probability over the full simulation box.
"""
function random_initial_config(num_part::Int, L::Float64)::Vector{Float64}
    return L .* rand(num_part) .- L/2  # Random initial configuration in the range [-L/2, L/2]
end

"""
    move_one_part(x_coord::Vector{Float64}, num_part::Int, delta::Float64, L::Float64) -> Tuple{Vector{Float64}, Int}

Proposes a random move for a single particle by displacing it within [-delta, delta] and applying periodic boundary conditions.

# Input:
- `x_coord::Vector{Float64}`: Current positions of all particles.
- `num_part::Int`: Number of particles.
- `delta::Float64`: Maximum displacement (half-width of the move interval).
- `L::Float64`: Length of the simulation box.

# Output:
- `Vector{Float64}`: New positions after the proposed move (with periodicity).
- `Int`: Index of the particle that was moved.

# Notes
A particle is selected at random and displaced by a random amount in [-delta, delta]. The new position is wrapped to the periodic box using the minimum image convention.
"""
function move_one_part(x_coord::Vector{Float64}, num_part::Int, delta::Float64, L::Float64)::Tuple{Vector{Float64}, Int}
    x_coord_new = copy(x_coord)
    idx = rand(1:num_part)  # Choose a random particle to move
    x_coord_new[idx] += rand() * (2 * delta) - delta  # Displacement in [-delta, delta]
    x_coord_new[idx] = get_periodic_difference(x_coord_new[idx], 0.0, L)  # Apply periodic boundary conditions
    return x_coord_new, idx
end

# ========== Energy Calculations ==========

"""
    interaction_energy(x_coord, num_part, V0, k_lat) -> Float64

Calculates the total interaction energy for a set of particles, each pair interacting via a cavity-mediated cosine potential.

# Input:
- `x_coord::Vector{Float64}`: Positions of all particles.
- `num_part::Int`: Number of particles.
- `V0::Float64`: Interaction strength.
- `k_lat::Float64`: Lattice/cavity wavevector.

# Output:
- `Float64`: The total potential energy, sum over all unique pairs.

# Notes
The interaction is given by V0 * cos(k_lat * x1) * cos(k_lat * x2) for each pair (i < j).
"""
function interaction_energy(x_coord::Vector{Float64}, num_part::Int, V0::Float64, k_lat::Float64)::Float64
    potential = 0.0
    @inbounds for i in 1:num_part
        @inbounds for j in (i + 1):num_part
            potential += V0 * cos(k_lat * x_coord[i]) * cos(k_lat * x_coord[j])
        end
    end
    return potential
end

"""
    kinetic_energy_log_form(x_coord, num_part, psi_interp, k_contact, L, α, long_range, fermi_stats, contact) -> Float64

Evaluates the kinetic energy using the logarithmic-derivative (local energy) form, supporting combinations of contact, Fermi, and long-range terms.

# Input:
- `x_coord::Vector{Float64}`: Positions of all particles.
- `num_part::Int`: Number of particles.
- `psi_interp`: Interpolated two-body wave function (Spline2D or similar).
- `k_contact::Float64`: Parameter for Bethe-Peierls contact term.
- `L::Float64`: Box length.
- `α::Float64`: Fermi statistics exponent.
- `long_range::Bool`, `fermi_stats::Bool`, `contact::Bool`: Toggles for each term.

# Output:
- `Float64`: Total kinetic energy.

# Notes
Each active contribution (contact, Fermi, long-range) is summed using its logarithmic derivatives, for improved numerical stability in Monte Carlo.
"""
function kinetic_energy_log_form(
    x_coord::Vector{Float64}, num_part::Int, psi_interp, k_contact::Float64,
    L::Float64, α::Float64, long_range::Bool, fermi_stats::Bool, contact::Bool
)::Float64
    E_kin = 0.0
    ψ_2 = psi_interp

    if contact
        # Kinetic energy: contact term
        for k in 1:num_part
            grad_logψ_3 = 0.0
            lapl_logψ_3 = 0.0
            for j in 1:num_part
                if j != k
                    sgn = sign(get_periodic_difference(x_coord[k], x_coord[j], L))
                    xkj = abs(get_periodic_difference(x_coord[k], x_coord[j], L))
                    ϕ = k_contact * (xkj - L/2)
                    grad_logψ_3 += -k_contact * tan(ϕ) * sgn
                    lapl_logψ_3 += -k_contact^2 / cos(ϕ)^2
                end
            end
            E_kin += -0.5 * (lapl_logψ_3 + grad_logψ_3^2)
        end
    end

    if fermi_stats
        # Kinetic energy: Fermi statistics term
        for k in 1:num_part
            grad_logψ = 0.0
            lapl_logψ = 0.0
            for j in 1:num_part
                if j != k
                    xkj = get_periodic_difference(x_coord[k], x_coord[j], L)
                    arg = π / L * xkj
                    grad_logψ += α * (π / L) * cot(arg)
                    lapl_logψ += -α * (π / L)^2 * csc(arg)^2
                end
            end
            E_kin += -0.5 * (lapl_logψ + grad_logψ^2)
        end
    end

    if long_range
        # Kinetic energy: long-range/spline term
        for k in 1:num_part
            grad_logψ_2 = 0.0
            lapl_logψ_2 = 0.0
            for j in 1:num_part
                if j != k
                    x_coord_per_1 = map_to_unit_cell(x_coord[k])
                    x_coord_per_2 = map_to_unit_cell(x_coord[j])
                    ψ_val = evaluate(ψ_2, x_coord_per_1, x_coord_per_2)
                    dψ = derivative(ψ_2, x_coord_per_1, x_coord_per_2, nux=1, nuy=0)
                    d2ψ = derivative(ψ_2, x_coord_per_1, x_coord_per_2, nux=2, nuy=0)
                    if ψ_val > 0
                        grad_2 = dψ / ψ_val
                        lap_2 = d2ψ / ψ_val
                        grad_logψ_2 += grad_2
                        lapl_logψ_2  += lap_2 - grad_2^2
                    end
                end
            end
            E_kin += -0.5 * (lapl_logψ_2 + grad_logψ_2^2)
        end
    end

    return E_kin
end

"""
    local_energy_log(x_coord, num_part, psi_interp, V0, k_lat, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        -> Tuple{Float64, Float64, Float64}

Computes the total, kinetic, and potential energies for a configuration using the logarithmic-derivative kinetic form.

# Input:
- All positions, model, and toggle parameters as above.

# Output:
- `(E_tot, E_kin, E_pot)`: Total energy, kinetic, and potential.

# Notes
If long_range is true, the cavity-mediated potential is used; otherwise potential is zero.
"""
function local_energy_log(
    x_coord::Vector{Float64}, num_part::Int, psi_interp,
    V0::Float64, k_lat::Float64, L::Float64, k_L::Float64, k_contact::Float64, α::Float64,
    long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool
)::Tuple{Float64, Float64, Float64}

    kinetic = kinetic_energy_log_form(x_coord, num_part, psi_interp, k_contact, L, α, long_range, fermi_stats, contact)
    potential = 0.0
    if long_range
        potential = interaction_energy(x_coord, num_part, V0, k_lat)
    end

    return kinetic + potential, kinetic, potential
end

"""
    local_energy(x_coord, num_part, psi_interp, V0, k_lat, L, k_L, k_contact, fermi_stats, reatto_chester, contact)
        -> Tuple{Float64, Float64, Float64}

Computes the local energy for a given configuration via finite-difference kinetic energy and direct evaluation of the interaction energy.

# Input:
- All positions, model, and toggle parameters as above.

# Output:
- `(E_tot, E_kin, E_pot)`: Total, kinetic, and potential energies.

# Notes
The kinetic energy is estimated via central finite differences; the potential is a double sum over all unique pairs.
If the wave function is zero, Outputs zeros for all energies.
"""
function local_energy(
    x_coord::Vector{Float64}, num_part::Int, psi_interp,
    V0::Float64, k_lat::Float64, L::Float64, k_L::Float64, k_contact::Float64,
    fermi_stats::Bool, reatto_chester::Bool, contact::Bool
)::Tuple{Float64, Float64, Float64}
    psi_current = trial_wave_function(x_coord, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
    if psi_current == 0.0
        return 0.0, 0.0, 0.0
    end

    kinetic = 0.0
    dx = 1e-5

    @inbounds for i in 1:num_part
        x_plus = copy(x_coord); x_plus[i] += dx
        x_minus = copy(x_coord); x_minus[i] -= dx

        psi_plus = trial_wave_function(x_plus, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        psi_minus = trial_wave_function(x_minus, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)

        kinetic -= 0.5 * (psi_plus - 2 * psi_current + psi_minus) / (dx^2 * psi_current)
    end

    potential = 0.0
    @inbounds for i in 1:num_part
        @inbounds for j in (i + 1):num_part
            potential += V0 * cos(k_lat * x_coord[i]) * cos(k_lat * x_coord[j])
        end
    end

    return kinetic + potential, kinetic, potential
end

# ------------------------------------------
# Final Metropolis Implementation
# ------------------------------------------

"""
    metropolis(num_part, num_steps, num_bins, delta, L, V0, k_lat, psi_interp, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        -> Tuple{Float64, Float64, Vector{ComplexF64}, Vector{Float64}, Matrix{Float64}, Float64, Vector{Float64}}

Runs a full Metropolis Monte Carlo simulation for a 1D quantum system with customizable two-body wavefunction structure and interactions.

# Input:
- `num_part::Int`: Number of particles.
- `num_steps::Int`: Number of Metropolis steps.
- `num_bins::Int`: Number of bins for histograms (density, pair).
- `delta::Float64`: Maximum displacement for particle moves.
- `L::Float64`: Length of the periodic simulation box.
- `V0::Float64`: Strength of the cavity-mediated potential.
- `k_lat::Float64`: Lattice/cavity wavevector.
- `psi_interp`: Interpolated two-body wavefunction (e.g. Spline2D).
- `k_L::Float64`: Reatto-Chester (Jastrow) exponent.
- `k_contact::Float64`: Bethe-Peierls contact parameter.
- `α::Float64`: Exponent for Fermi statistics.
- `long_range::Bool`: Enable/disable long-range (cavity) term.
- `fermi_stats::Bool`: Enable/disable Fermi statistics term.
- `reatto_chester::Bool`: Enable/disable Jastrow/RC term.
- `contact::Bool`: Enable/disable contact term.

# Output:
- `Float64`: Mean total energy per configuration (E_tot / n_uncorr).
- `Float64`: Mean squared total energy per configuration.
- `Vector{ComplexF64}`: Static structure factor S(k) (for a range of k values).
- `Vector{Float64}`: Normalized 1D density histogram (n(x)).
- `Matrix{Float64}`: Normalized 2D pair density histogram (g2(x, x')).
- `Float64`: Acceptance ratio of proposed moves.
- `Vector{Float64}`: Block-averaged local energy per particle, for convergence diagnostics.

# Notes
- Uses block averaging (`step_block`) for energy and structure factor sampling to reduce autocorrelation.
- Particle positions are stored and binned in [-L/2, L/2].
- Output: density and pair correlation histograms normalized as probability densities.
- The function displays a plot of the energy evolution over Monte Carlo steps.

# Usage
Call this function to simulate the equilibrium properties of the system, and to extract observables such as energy, density profiles, g2, and structure factor.

# Example
```julia
E, E2, SSF, n_x, g2_xx, acc_ratio, E_trace = metropolis(8, 10^6, 100, 0.05, 1.0, 1.0, 2π, psi_interp, 2.0, 1.0, 1.0, true, true, false, false)
"""
function metropolis(num_part::Int, num_steps::Int, num_bins::Int, delta::Float64, L::Float64, V0::Float64, k_lat::Float64, psi_interp, k_L::Float64, k_contact::Float64, α::Float64, long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool)::Tuple{Float64, Float64, Vector{ComplexF64}, Vector{Float64}, Matrix{Float64}, Float64, Vector{Float64}, Vector{Vector{Float64}}}
    acceptance_ratio = 0.0
    n_uncorr = 0
    bins = range(-L/2, stop=L/2, length=num_bins+1)
    hist_1d = zeros(Float64, num_bins)
    hist_2d = zeros(Float64, num_bins, num_bins)
    dx = L / num_bins
    step_block = num_steps ÷ num_steps
    E_tot = 0.0
    E_sq = 0.0
    E_local_values = Vector{Float64}(undef, (num_steps÷step_block))
    iter_val = Vector{Float64}(undef, (num_steps÷step_block))
    idx_plot = 1
    configurations = Vector{Vector{Float64}}(undef, num_steps ÷ step_block)

    final_point = 5*L
    k = (2*π/L) * collect(1:1:final_point)
    SSF = zeros(ComplexF64, length(k))

    plt = plot(title=L"Evolution of Local Energy", xlabel=L"Step", ylabel=L"Local Energy", legend=false)

    x_coord = random_initial_config(num_part, L)
    psi_old_val = trial_wave_function(x_coord, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)

    progress = Progress(num_steps; desc="Running Metropolis $num_part...", showspeed=true)
    for i in 1:num_steps
        next!(progress)  # Update progress bar
        x_new, moved_idx = move_one_part(x_coord, num_part, delta, L)

        x_new_mapped = copy(x_new)
        x_new_mapped[moved_idx] = map_to_unit_cell(x_new[moved_idx])

        if x_new_mapped[moved_idx] == 0.5 || x_new_mapped[moved_idx] == -0.5
            println(x_new_mapped[moved_idx])
        end

        psi_ratio = update_wave_function_after_move(x_coord, x_new, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        psi_new_val = psi_old_val * psi_ratio
    
        ϵ = 1e-300  # tiny epsilon to prevent log(0)
        logψ_old = log(abs(psi_old_val) + ϵ)
        logψ_new = log(abs(psi_new_val) + ϵ)

        logw = 2 * (logψ_new - logψ_old)

        if log(rand()) < logw
            x_coord = x_new
            psi_old_val = psi_new_val
            acceptance_ratio += 1
        end
    
        for x in x_coord
            bin_idx = min(num_bins, max(1, Int(floor((x + L/2) / L * num_bins)) + 1))
            hist_1d[bin_idx] += 1
        end
    
        for i in 1:num_part
            for j in (i + 1):num_part
                bin_x = min(num_bins, max(1, Int(floor((x_coord[i] + L/2) / L * num_bins)) + 1))
                bin_y = min(num_bins, max(1, Int(floor((x_coord[j] + L/2) / L * num_bins)) + 1))

                hist_2d[bin_x, bin_y] += 1
            end
        end
    
        if i % step_block == 0

            E_local, E_kinetic, E_potential = local_energy_log(x_coord, num_part, psi_interp, V0, k_lat, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
            # E_local, E_kinetic, E_potential = local_energy(x_coord, num_part, psi_interp, V0, k_lat, L, k_L, k_contact, fermi_stats, reatto_chester, contact)
            E_tot += E_local
            E_sq += E_local^2
            n_uncorr += 1

            if isnan(E_local)
                @warn "NaN detected at step $i: E_local = $E_local"
                continue  # Skip this iteration to avoid polluting data
            end

            for a in 1:num_part, b in 1:num_part
                SSF .+= exp.(im * (x_coord[a] - x_coord[b]) .* k)
            end
    
            iter_val[idx_plot] = i
            E_local_values[idx_plot] = E_local / num_part
            configurations[idx_plot] = copy(x_coord)
            idx_plot += 1
        end
    end

    hist_1d ./= (sum(hist_1d) * dx)
    hist_2d ./= (sum(hist_2d) * dx^2)
    SSF ./= n_uncorr

    plot!(plt, iter_val, E_local_values, label=L"E", color=:blue)
    display(plt)

    return E_tot / n_uncorr, E_sq / n_uncorr, SSF, hist_1d, hist_2d, acceptance_ratio / num_steps, E_local_values, configurations
end
