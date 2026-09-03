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

function trial_log_wave_function(
    x_coord::Vector{Float64}, num_part::Int, psi_interp,
    L::Float64, k_L::Float64, k_contact::Float64, α::Float64,
    long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool
)::Float64
    logpsi = 0.0
    @inbounds for i in 1:num_part
        @inbounds for j in (i + 1):num_part
            if fermi_stats
                logpsi += α * log(abs(sin((π/L) * (x_coord[i] - x_coord[j]))))
            end
            if reatto_chester
                logpsi += k_L * log(abs(sin((π/L) * (x_coord[i] - x_coord[j]))))
            end
            if contact
                dist_mod = abs(get_periodic_difference(x_coord[i], x_coord[j], L)) - L/2
                logpsi += log(abs(cos(k_contact * dist_mod)))
            end
            if long_range
                x1 = map_to_unit_cell(x_coord[i])
                x2 = map_to_unit_cell(x_coord[j])
                logpsi += log(abs(evaluate(psi_interp, x1, x2)))
            end
        end
    end
    return logpsi
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

function update_log_wave_function_after_move(
    x_coord::Vector{Float64}, x_new::Vector{Float64}, num_part::Int,
    psi_interp, L::Float64, k_L::Float64, k_contact::Float64, α::Float64,
    long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool
)::Float64
    log_ratio = 0.0
    @inbounds for i in 1:num_part
        if x_coord[i] != x_new[i]
            @inbounds for j in 1:num_part
                if i != j
                    if fermi_stats
                        log_ratio += α * (log(abs(sin((π/L)*(x_new[i]-x_new[j])))) -
                                           log(abs(sin((π/L)*(x_coord[i]-x_coord[j])))))
                    end
                    if reatto_chester
                        log_ratio += k_L * (log(abs(sin((π/L)*(x_new[i]-x_new[j])))) -
                                             log(abs(sin((π/L)*(x_coord[i]-x_coord[j])))))
                    end
                    if contact
                        dist_new = abs(get_periodic_difference(x_new[i], x_new[j], L)) - L/2
                        dist_old = abs(get_periodic_difference(x_coord[i], x_coord[j], L)) - L/2
                        log_ratio += log(abs(cos(k_contact*dist_new))) - log(abs(cos(k_contact*dist_old)))
                    end
                    if long_range
                        x_new_1 = map_to_unit_cell(x_new[i]);   x_new_2 = map_to_unit_cell(x_new[j])
                        x_old_1 = map_to_unit_cell(x_coord[i]); x_old_2 = map_to_unit_cell(x_coord[j])
                        log_ratio += log(abs(evaluate(psi_interp, x_new_1, x_new_2))) -
                                     log(abs(evaluate(psi_interp, x_old_1, x_old_2)))
                    end
                end
            end
        end
    end
    return log_ratio
end

