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

"""
    global_shift_move(x_coord::Vector{Float64}, L::Float64, shift_period::Float64) -> Vector{Float64}

Proposes a collective move: every particle is shifted by exactly half the cavity
period, x_i -> x_i + shift_period/2 (mod L). This is the Z2 symmetry operation
of the cavity potential (cos(k_lat*x) -> -cos(k_lat*x) for every particle), so
the interaction energy is exactly unchanged by this move — only the trial
wavefunction ratio determines acceptance.

# Notes
`shift_period` should be `2π/k_lat` (the physical period of the cavity potential),
not L. For k_lat = 2π this is 1.0, matching the L_unit_cell used in the
wavefunction interpolation.
"""
function global_shift_move(x_coord::Vector{Float64}, L::Float64, shift_period::Float64)::Vector{Float64}
    shift = shift_period / 2
    return [get_periodic_difference(x + shift, 0.0, L) for x in x_coord]
end
