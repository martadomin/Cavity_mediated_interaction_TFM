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
    int_strength = V0
    @inbounds for i in 1:num_part
        @inbounds for j in (i + 1):num_part
            potential += int_strength * cos(k_lat * x_coord[i]) * cos(k_lat * x_coord[j])
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
    for k in 1:num_part
        grad_total = 0.0
        lapl_total = 0.0

        for j in 1:num_part
            if j != k
                if contact
                    sgn = sign(get_periodic_difference(x_coord[k], x_coord[j], L))
                    xkj = abs(get_periodic_difference(x_coord[k], x_coord[j], L))
                    ϕ = k_contact * (xkj - L/2)
                    grad_total += -k_contact * tan(ϕ) * sgn
                    lapl_total += -k_contact^2 / cos(ϕ)^2
                end

                if fermi_stats
                    xkj = get_periodic_difference(x_coord[k], x_coord[j], L)
                    arg = π / L * xkj
                    grad_total += α * (π / L) * cot(arg)
                    lapl_total += -α * (π / L)^2 * csc(arg)^2
                end

                if long_range
                    x1 = map_to_unit_cell(x_coord[k])
                    x2 = map_to_unit_cell(x_coord[j])
                    ψ_val = evaluate(psi_interp, x1, x2)
                    dψ   = derivative(psi_interp, x1, x2, nux=1, nuy=0)
                    d2ψ  = derivative(psi_interp, x1, x2, nux=2, nuy=0)
                    if ψ_val > 0.0
                        grad_lr = dψ / ψ_val
                        lapl_lr = d2ψ / ψ_val - grad_lr^2
                        grad_total += grad_lr
                        lapl_total += lapl_lr
                    end
                end
            end
        end

        # Cross-terms now included automatically via grad_total^2
        E_kin += -0.5 * (lapl_total + grad_total^2)
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

