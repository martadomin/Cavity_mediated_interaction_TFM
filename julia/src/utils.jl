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
