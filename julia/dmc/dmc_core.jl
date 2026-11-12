using Random, Plots, Statistics, ProgressMeter, NPZ, Dierckx, Printf, Roots
using Base.Threads

include(joinpath(@__DIR__, "..", "src", "VMCCore.jl"))
# NOTE (flagged, not silently resolved): this used to be a *second*, locally
# redefined copy of get_periodic_difference, map_to_unit_cell, find_k_contact,
# and interpolated_wave_function. The local interpolated_wave_function used
# kx=3, ky=3 (cubic spline), while src/wavefunctions.jl (formerly part of
# metropolis_with_contact.jl) uses kx=4, ky=4 (quartic). Because the local
# copy was defined *after* the include, it silently overrode the shared one,
# so DMC and VMC were evaluating the trial wavefunction with different spline
# orders. Removing the duplicate now makes DMC use the same kx=4, ky=4 spline
# as VMC. Confirm this is the intended interpolation order for DMC before
# trusting existing DMC results that pre-date this change.

# Global constants for diffusion
const ħ = 1.0
const m = 1.0
const D = ħ^2 / (2 * m)

# ------------------------------------------
# Drift Force Calculation
# ------------------------------------------

"""
    calculate_drift_force!(drift_force, x_diff, ...)

Calculates the drift force F = 2∇log(Ψ) for each particle in a walker.

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

Calculates the total kinetic energy of a walker using the log-derivative form,
T = -½ Σ_k [∂²_k log Ψ + (∂_k log Ψ)²].

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
- The gradients of all enabled components are summed *before* squaring, so the
  cross terms between components are included (same as `kinetic_energy_log_form`
  in src/energy.jl, which is used by VMC).
"""
function calculate_kinetic_energy(x_coord::Vector{Float64}, num_particles::Int64, L::Float64, α::Float64, ψ_2, k_contact::Float64, long_range::Bool, fermi_stats::Bool, contact::Bool)::Float64
    E_kin = 0.0
    for k in 1:num_particles
        grad_total = 0.0
        lapl_total = 0.0
        for j in 1:num_particles
            if j != k
                if fermi_stats
                    xkj = get_periodic_difference(x_coord[k], x_coord[j], L)
                    arg = π / L * xkj
                    grad_total += α * (π / L) * cot(arg)
                    lapl_total += -α * (π / L)^2 * csc(arg)^2
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
                    grad_total += grad_2
                    lapl_total += d2ψ / ψ_val - grad_2^2
                end
                if contact
                    sgn = sign(get_periodic_difference(x_coord[k], x_coord[j], L))
                    xkj = abs(get_periodic_difference(x_coord[k], x_coord[j], L))
                    ϕ = k_contact * (xkj - L/2)
                    grad_total += -k_contact * tan(ϕ) * sgn
                    lapl_total += -k_contact^2 / cos(ϕ)^2
                end
            end
        end
        E_kin += -0.5 * (lapl_total + grad_total^2)
    end
    return E_kin
end

"""
    calculate_potential_energy(x, N, V0, k_lat)

Evaluates total pairwise interaction energy using the cosine cavity-mediated potential
V0 cos(k_lat x_a) cos(k_lat x_b). Only unique a<b pairs are summed.
"""
function calculate_potential_energy(x_new::Vector{Float64}, num_particles::Int64, V0::Float64, k_lat::Float64)::Float64
    V = 0.0
    for a in 1:num_particles
        for b in (a+1):num_particles
            V += V0 * cos(k_lat * x_new[a]) * cos(k_lat * x_new[b])
        end
    end
    return V
end

"""
    dmc_local_energy(x, ...)

Local energy E_L = T + V of a walker. T is only included with importance sampling
(without it the walkers sample Φ₀ itself and only V enters the branching).
Returns `Inf` for configurations where Ψ_T ≤ 0.
"""
function dmc_local_energy(x::Vector{Float64}, num_particles::Int64, L::Float64, α::Float64, V0::Float64, k_lat::Float64, ψ_2, k_contact::Float64, long_range::Bool, fermi_stats::Bool, contact::Bool, importance_sampling::Bool)::Float64
    E_kin = importance_sampling ? calculate_kinetic_energy(x, num_particles, L, α, ψ_2, k_contact, long_range, fermi_stats, contact) : 0.0
    V = long_range ? calculate_potential_energy(x, num_particles, V0, k_lat) : 0.0
    return E_kin + V
end

"""
    fermi_sign(x, N, L) -> Int

Sign of Π_{i<j} sin(π(x_i − x_j)/L), the nodal structure of the fermionic trial
wavefunction. Compared between the old (wrapped) configuration and the *unwrapped*
new one, so a change of sign means two particles crossed (a node was crossed).
"""
function fermi_sign(x::Vector{Float64}, num_particles::Int, L::Float64)::Int
    s = 1
    @inbounds for i in 1:num_particles, j in (i + 1):num_particles
        sin(π / L * (x[i] - x[j])) < 0 && (s = -s)
    end
    return s
end

"""
    blocking_error(x; min_blocks=32) -> Float64

Standard error of the mean of a correlated time series (Flyvbjerg-Petersen blocking):
the series is repeatedly halved by averaging neighbours and the naive error is taken at
each level; the returned value is the largest one with at least `min_blocks` blocks
(the plateau, if the series is long enough to reach it).
"""
function blocking_error(x::AbstractVector{Float64}; min_blocks::Int = 32)::Float64
    y = collect(x)
    err = std(y) / sqrt(length(y))
    while length(y) ÷ 2 >= min_blocks
        n = length(y) ÷ 2
        y = 0.5 .* (y[1:2:2n] .+ y[2:2:2n])
        err = max(err, std(y) / sqrt(n))
    end
    return err
end

"""
    z2_invariance_error(num_particles, L, k_lat, ψ_2, k_L, k_contact, α, long_range, fermi_stats,
                        reatto_chester, contact; n_test=200) -> Float64

Largest |log Ψ_T(x + λ/2) − log Ψ_T(x)| over `n_test` random configurations, λ = 2π/k_lat.
The DMC Z₂ shift is only unbiased if Ψ_T is invariant under the shift (H always is), so
this should be at the level of the spline accuracy (~1e-5 to 2e-3 for the 20×20 grids).
"""
function z2_invariance_error(num_particles::Int, L::Float64, k_lat::Float64, ψ_2, k_L::Float64, k_contact::Float64, α::Float64,
                             long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool; n_test::Int = 200)::Float64
    max_err = 0.0
    for _ in 1:n_test
        x = random_initial_config(num_particles, L)
        x_s = global_shift_move(x, L, 2π / k_lat)
        lp   = trial_log_wave_function(x,   num_particles, ψ_2, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        lp_s = trial_log_wave_function(x_s, num_particles, ψ_2, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        if isfinite(lp) && isfinite(lp_s)
            max_err = max(max_err, abs(lp_s - lp))
        end
    end
    return max_err
end

# local_energy(x_coord, num_part, psi_interp, V0, k_lat, L, k_L, k_contact,
#              fermi_stats, reatto_chester, contact) is used here unchanged.
# It was previously redefined verbatim (identical body) in this file; the
# duplicate has been removed and the shared definition from src/energy.jl is
# used instead. NOTE (flagged, not silently resolved): this function reads
# `α` and `long_range` from the enclosing scope rather than taking them as
# arguments, so its behavior depends on whatever values those globals hold
# in the script that includes this file. That was true before this
# reorganization too; it is not something this restructuring changed, but
# it is worth confirming `α`/`long_range` are set as intended in run_dmc.jl
# before each run.

# ------------------------------------------
# DMC Simulation Core Function
# ------------------------------------------

"""
    DMC_part_w_inter(...; k_lat=2π, k_L=0.0, reatto_chester=false, walker_file="",
                     burn_in_frac=0.3, pop_feedback=0.1, compute_obdm=true, obdm_every=10)

Runs a Diffusion Monte Carlo simulation with optional importance sampling,
branching, histogramming, structure factor and one-body density matrix.

Propagation with importance sampling (drift velocity v = D F, F = 2∇log Ψ_T):
- `quadratic = true` (default, second-order DMC): drift over Δτ/2 with a Heun (RK2) step,
  x_a = x + v(x)Δτ/2,  x_m = x + ½[v(x) + v(x_a)]Δτ/2;  Gaussian diffusion x_m += √(2DΔτ) η;
  drift over Δτ/2 again from x_m with the same Heun step. Time-step error O(Δτ²).
- `quadratic = false` (linear DMC): x' = x + v(x)Δτ + √(2DΔτ) η. Time-step error O(Δτ).
Then branching with weight exp(-(½[E_L(x) + E_L(x')] - E_T) Δτ), no accept/reject.
E_T and the energy of each step use the mean E_L of the post-branching population.
For fermions, walkers that cross a node (x_i = x_j) are killed (fixed node, exact in 1D).

All observables measured here are *mixed* estimators ⟨Ψ_T|Ô|Φ₀⟩. For quantities that
do not commute with H (n(x), S(k), ρ₁) use the extrapolation 2·DMC − VMC.

# Arguments
- `num_particles`: Number of particles in the system.
- `num_steps`: Total number of DMC steps.
- `num_walkers`: Target number of walkers.
- `num_bins`: Number of histogram bins (full box [-L/2, L/2)).
- `Δτ`: Imaginary time step.
- `E_T`: Initial trial energy.
- `α`: Fermi statistics interaction strength.
- `V0`: Interaction strength for the long-range potential (bare, per pair).
- `L`: System size.
- `ψ_2`: Interpolated 2D trial wavefunction (Spline2D).
- `k_contact`: Parameter from Bethe-Peierls contact condition.
- `long_range`, `fermi_stats`, `contact`: Enable corresponding interaction components.
- `importance_sampling`: Whether to use drift forces.
- `previous_walkers`: If true, load initial walker positions from `walker_file`.

# Keyword arguments
- `k_lat`: Cavity wavevector (default 2π).
- `k_L`, `reatto_chester`: Only used by the OBDM estimator (Ψ_T ratio).
- `walker_file`: .npy file with initial walkers (num_walkers × num_particles).
- `burn_in_frac`: Fraction of steps (equilibration) discarded before measuring anything (default 0.2).
- `quadratic`: Second-order (quadratic) propagator (default) or linear one (`false`).
- `pop_feedback`: Population control strength g in E_T = ⟨E_L⟩ − (g/Δτ) log(N_w/N_w⁰).
- `compute_obdm`: Accumulate the one-body density matrix.
- `obdm_every`: Measure the OBDM every this many steps (it costs N Ψ_T ratios per walker).
- `z2_shift_prob`: Per-walker, per-step probability of the global shift x → x + λ/2 (λ = 2π/k_lat).
  H and Ψ_T are both invariant under it, so it leaves the DMC distribution Ψ_T Φ₀ unchanged
  (no weight, no accept/reject) while letting walkers visit both density-wave branches,
  which diffusion alone cannot do at strong coupling. Any value in (0, 1] is unbiased;
  0 disables it. It is switched off automatically if Ψ_T is found not to be invariant.
- `vmc_init_sweeps`: Metropolis sweeps per walker (sampling |Ψ_T|²) before the DMC starts,
  when walkers are not loaded from file. 0 keeps the uniform initial distribution.
- `verbose_every`: Print walker number, ⟨E_L⟩ and E_T every this many steps (0 = off).
- `n_cell_bins`: Bins of the density folded into one cavity period (x mod λ), accumulated
  every step for every particle (input for the Leggett bound, which needs the density minima).

# Returns
Tuple `(energies, energy, hist_1d, hist_2d, SSF, walker_positions, obdm, n_cell)`:
- `energies`: Mean local energy of the post-branching population at each step.
- `energy`: Average energy after equilibration (error: `blocking_error` on `energies`).
- `hist_1d`: Mixed 1D density, normalized to unit integral.
- `hist_2d`: Mixed pair density, normalized to unit integral.
- `SSF`: Mixed static structure factor, k = 2πn/L, n = 1 … 5L.
- `walker_positions`: Final walker configurations.
- `obdm`: Mixed ρ₁(x, x') on the same grid as VMC `metropolis`
  (nb = max(num_bins, 25·L·k_lat/2π)), normalized so ∫ρ₁(x,x)dx = N.
- `n_cell`: Mixed density folded into one cavity period, x ∈ [-λ/2, λ/2), normalized to
  mean 1 (n(x)/n̄).
"""
function DMC_part_w_inter(num_particles::Int64, num_steps::Int64, num_walkers::Int64, num_bins::Int64, Δτ::Float64, E_T::Float64, α::Float64, V0::Float64, L::Float64, ψ_2, k_contact::Float64, long_range::Bool, fermi_stats::Bool, contact::Bool, importance_sampling::Bool, previous_walkers::Bool;
                          k_lat::Float64 = 2π, k_L::Float64 = 0.0, reatto_chester::Bool = false,
                          walker_file::String = "", burn_in_frac::Float64 = 0.2, pop_feedback::Float64 = 0.1,
                          quadratic::Bool = true,
                          compute_obdm::Bool = true, obdm_every::Int = 10, z2_shift_prob::Float64 = 0.1,
                          vmc_init_sweeps::Int = 200, verbose_every::Int = 0, n_cell_bins::Int = 200
                          )::Tuple{Vector{Float64}, Float64, Vector{Float64}, Matrix{Float64}, Vector{ComplexF64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

    # === Basic parameters ===
    dx = L / num_bins
    bin_of(x) = min(num_bins, max(1, Int(floor((x + L/2) / L * num_bins)) + 1))
    num_walkers_initial = num_walkers # Target number for population control
    energies = zeros(Float64, num_steps) # Mean local energy at each time step
    n_burn = floor(Int, burn_in_frac * num_steps)

    # === Z₂ shift: x → x + λ/2 must map the periodic box onto itself and leave Ψ_T invariant ===
    z2_shift = π / k_lat
    if z2_shift_prob > 0
        if !isapprox(mod(L, z2_shift), 0.0; atol=1e-10) && !isapprox(mod(L, z2_shift), z2_shift; atol=1e-10)
            @warn "L = $L is not a multiple of λ/2 = $z2_shift; disabling the Z₂ shift."
            z2_shift_prob = 0.0
        else
            z2_err = z2_invariance_error(num_particles, L, k_lat, ψ_2, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
            @printf("Z₂ invariance of Ψ_T: max |Δ log Ψ_T| = %.2e\n", z2_err)
            # ~1e-3 is the interpolation error of the spline itself (H is exactly invariant)
            if z2_err > 1e-2
                @warn "Ψ_T is not Z₂ invariant (max |Δ log Ψ_T| = $z2_err); the shift would bias DMC, disabling it."
                z2_shift_prob = 0.0
            end
        end
    end

    # === Walker initialization ===
    if previous_walkers
        isfile(walker_file) || error("Walker file not found: $walker_file")
        walker_positions = Matrix{Float64}(NPZ.npzread(walker_file))
        size(walker_positions, 2) == num_particles || error("Walker file has $(size(walker_positions, 2)) particles, expected $num_particles")
        num_walkers = size(walker_positions, 1)
    else
        # Initialize walkers uniformly in the box; burn-in removes the bias
        walker_positions = rand(num_walkers, num_particles) .* L .- L / 2
    end

    # Start the walkers from |Ψ_T|² rather than uniform: a few Metropolis sweeps per walker.
    # At strong coupling uniform walkers have E_L ≫ E_T and the population would collapse.
    if !previous_walkers && vmc_init_sweeps > 0
        ψ_init = [ψ_2 isa Spline2D ? FastSpline2D(ψ_2) : ψ_2 for _ in 1:nthreads()]
        δ_init = 0.25 * 2π / k_lat
        @threads :static for w in 1:num_walkers
            ψ_t = ψ_init[threadid()]
            x = walker_positions[w, :]
            x_try = copy(x)
            for _ in 1:(vmc_init_sweeps * num_particles)
                i = rand(1:num_particles)
                x_try[i] = mod(x[i] + δ_init * (2rand() - 1) + L/2, L) - L/2
                log_ratio = update_log_wave_function_after_move(x, x_try, num_particles, ψ_t, L, k_L, k_contact, α,
                                                                long_range, fermi_stats, reatto_chester, contact)
                if log(rand()) < 2 * log_ratio
                    x[i] = x_try[i]
                else
                    x_try[i] = x[i]
                end
            end
            walker_positions[w, :] .= x
        end
    end

    # Per-walker cache at the current position: E_L and drift force F (both reused next step)
    ψ_serial = ψ_2 isa Spline2D ? FastSpline2D(ψ_2) : ψ_2
    walker_E = zeros(num_walkers)
    walker_F = zeros(num_walkers, num_particles)
    F_buf = zeros(num_particles)
    for w in 1:num_walkers
        x = walker_positions[w, :]
        walker_E[w] = dmc_local_energy(x, num_particles, L, α, V0, k_lat, ψ_serial, k_contact,
                                       long_range, fermi_stats, contact, importance_sampling)
        if importance_sampling
            if calculate_drift_force!(F_buf, x, num_particles, L, α, ψ_serial, k_contact, long_range, fermi_stats, contact)
                walker_F[w, :] .= F_buf
            else
                walker_E[w] = Inf
            end
        end
    end

    # === k-vectors for static structure factor ===
    k_vec = (2π / L) .* collect(1:floor(Int, 5L))

    # === One-body density matrix grid (identical to VMC `metropolis`) ===
    nb_obdm = max(num_bins, round(Int, 25 * L * k_lat / (2π)))

    # === Thread-local preallocated buffers ===
    nt = nthreads()
    thread_new_positions_storage = [Float64[] for _ in 1:nt]
    thread_new_E_storage = [Float64[] for _ in 1:nt]
    thread_new_F_storage = [Float64[] for _ in 1:nt]
    thread_hist_1d_storage = [zeros(Float64, num_bins) for _ in 1:nt]
    thread_cell_storage = [zeros(Float64, n_cell_bins) for _ in 1:nt]
    λ_cav = 2π / k_lat
    cell_bin(x) = min(n_cell_bins, Int(floor(mod(x + λ_cav / 2, λ_cav) / λ_cav * n_cell_bins)) + 1)
    thread_hist_2d_storage = [zeros(Float64, num_bins, num_bins) for _ in 1:nt]
    thread_SSF_storage = [zeros(ComplexF64, length(k_vec)) for _ in 1:nt]
    thread_obdm_storage = [zeros(Float64, compute_obdm ? nb_obdm : 1, compute_obdm ? nb_obdm : 1) for _ in 1:nt]
    thread_obdm_weight = zeros(Float64, nt)
    thread_rand_nums_buffer = [zeros(num_particles) for _ in 1:nt]
    thread_drift_force = [zeros(num_particles) for _ in 1:nt]
    thread_x_old = [zeros(num_particles) for _ in 1:nt]
    thread_x_new = [zeros(num_particles) for _ in 1:nt]
    thread_x_tmp = [zeros(num_particles) for _ in 1:nt]
    thread_F_new = [zeros(num_particles) for _ in 1:nt]
    thread_x_a = [zeros(num_particles) for _ in 1:nt]      # quadratic propagator: intermediate points
    thread_x_mid = [zeros(num_particles) for _ in 1:nt]
    thread_F_a = [zeros(num_particles) for _ in 1:nt]
    thread_F_mid = [zeros(num_particles) for _ in 1:nt]
    # One allocation-free copy of the trial-wavefunction spline per thread (same B-spline as ψ_2)
    thread_ψ = [ψ_2 isa Spline2D ? FastSpline2D(ψ_2) : ψ_2 for _ in 1:nt]

    # === Accumulated observables ===
    hist_1d_total = zeros(Float64, num_bins)
    hist_2d_total = zeros(Float64, num_bins, num_bins)
    SSF_total = zeros(ComplexF64, length(k_vec))
    total_walkers_sampled = 0

    diffusion_scale = sqrt(2 * D * Δτ)

    # === Main DMC loop ===
    @showprogress for step in 1:num_steps
        measure = step > n_burn
        measure_obdm = compute_obdm && measure && (step % obdm_every == 0)

        # Clear per-step thread-local buffers (observable accumulators persist)
        for i in 1:nt
            empty!(thread_new_positions_storage[i])
            empty!(thread_new_E_storage[i])
            empty!(thread_new_F_storage[i])
        end

        # :static so that threadid() is a stable index into the buffers
        @threads :static for w in 1:num_walkers
            tid = threadid()
            ψ_t = thread_ψ[tid]
            F = thread_drift_force[tid]
            F_new = thread_F_new[tid]
            x_old = thread_x_old[tid]
            x_new = thread_x_new[tid]
            rand_nums = thread_rand_nums_buffer[tid]

            for p in 1:num_particles
                x_old[p] = walker_positions[w, p]
                F[p] = walker_F[w, p]
            end
            E_old = walker_E[w]
            E_new = Inf

            # --- Drift + diffusion. Coordinates stay unwrapped until the end: the drift and
            # Ψ_T only use periodic differences / map_to_unit_cell, and the node check needs
            # the continuous path. ---
            randn!(rand_nums)
            ok = isfinite(E_old)
            if quadratic
                x_a, x_mid, F_a, F_mid = thread_x_a[tid], thread_x_mid[tid], thread_F_a[tid], thread_F_mid[tid]
                # drift over Δτ/2 (Heun / RK2)
                @. x_a = x_old + D * F * (Δτ / 2)
                ok = ok && calculate_drift_force!(F_a, x_a, num_particles, L, α, ψ_t, k_contact, long_range, fermi_stats, contact)
                @. x_mid = x_old + 0.5 * D * (F + F_a) * (Δτ / 2)
                # Gaussian diffusion over Δτ
                @. x_mid += diffusion_scale * rand_nums
                # drift over Δτ/2 (Heun / RK2)
                ok = ok && calculate_drift_force!(F_mid, x_mid, num_particles, L, α, ψ_t, k_contact, long_range, fermi_stats, contact)
                @. x_a = x_mid + D * F_mid * (Δτ / 2)
                ok = ok && calculate_drift_force!(F_a, x_a, num_particles, L, α, ψ_t, k_contact, long_range, fermi_stats, contact)
                @. x_new = x_mid + 0.5 * D * (F_mid + F_a) * (Δτ / 2)
            else
                @. x_new = x_old + D * Δτ * F + diffusion_scale * rand_nums
            end
            # fixed node (exact in 1D): a fermionic walker that crosses x_i = x_j is killed
            crossed_node = fermi_stats && fermi_sign(x_new, num_particles, L) != fermi_sign(x_old, num_particles, L)
            @. x_new = mod(x_new + L/2, L) - L/2

            # --- Drift at the new position (cached for the next step) and local energy ---
            ok = ok && !crossed_node
            if ok && importance_sampling
                ok = calculate_drift_force!(F_new, x_new, num_particles, L, α, ψ_t, k_contact, long_range, fermi_stats, contact)
            elseif !importance_sampling
                fill!(F_new, 0.0)
            end
            if ok
                E_new = dmc_local_energy(x_new, num_particles, L, α, V0, k_lat, ψ_t, k_contact,
                                         long_range, fermi_stats, contact, importance_sampling)
            end

            # --- Z₂ shift (symmetry of H and Ψ_T, so E_L and F are unchanged) ---
            if z2_shift_prob > 0 && rand() < z2_shift_prob
                @. x_new = mod(x_new + z2_shift + L/2, L) - L/2
            end

            # --- Branching with the symmetrized local energy ---
            M = isfinite(E_new) ? floor(Int, exp(-(0.5 * (E_old + E_new) - E_T) * Δτ) + rand()) : 0

            if M > 0
                if measure
                    # --- Density ---
                    for a in 1:num_particles
                        thread_hist_1d_storage[tid][bin_of(x_new[a])] += M
                        thread_cell_storage[tid][cell_bin(x_new[a])] += M
                    end

                    # --- Pair density (2D histogram) ---
                    for a in 1:num_particles, b in (a+1):num_particles
                        bx = bin_of(x_new[a]); by = bin_of(x_new[b])
                        thread_hist_2d_storage[tid][bx, by] += M
                        thread_hist_2d_storage[tid][by, bx] += M
                    end

                    # --- Structure factor: Σ_ab e^{ik(x_a-x_b)} = |ρ_k|², ρ_k = Σ_a e^{ik x_a} ---
                    SSF_t = thread_SSF_storage[tid]
                    for ik in eachindex(k_vec)
                        re = 0.0; imag_part = 0.0
                        for a in 1:num_particles
                            s, c = sincos(k_vec[ik] * x_new[a])
                            re += c; imag_part += s
                        end
                        SSF_t[ik] += M * (re^2 + imag_part^2)
                    end
                end

                # --- One-body density matrix (McMillan estimator with Ψ_T ratios) ---
                if measure_obdm
                    accumulate_obdm!(thread_obdm_storage[tid], x_new, thread_x_tmp[tid], num_particles, L, nb_obdm, ψ_t,
                                     k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact, Float64(M))
                    thread_obdm_weight[tid] += M
                end

                # --- Store surviving walker(s) ---
                for _ in 1:M
                    append!(thread_new_positions_storage[tid], x_new)
                    push!(thread_new_E_storage[tid], E_new)
                    append!(thread_new_F_storage[tid], F_new)
                end
            end
        end

        # === Aggregate step results ===
        new_positions_flat = reduce(vcat, thread_new_positions_storage)

        if isempty(new_positions_flat)
            error("All walkers died. Try smaller Δτ or adjust E_T.")
        end

        # === Prepare next step ===
        num_walkers = length(new_positions_flat) ÷ num_particles
        walker_positions = Matrix(reshape(new_positions_flat, (num_particles, num_walkers))')
        walker_E = reduce(vcat, thread_new_E_storage)
        walker_F = Matrix(reshape(reduce(vcat, thread_new_F_storage), (num_particles, num_walkers))')

        if measure
            total_walkers_sampled += num_walkers
        end

        # === Energy of this step and population control (post-branching population:
        # a walker with M copies appears M times, so the plain mean carries the weights) ===
        E_loc_mean = mean(walker_E)
        energies[step] = E_loc_mean
        E_T = E_loc_mean - (pop_feedback / Δτ) * log(num_walkers / num_walkers_initial)

        if verbose_every > 0 && step % verbose_every == 0
            @printf("step %6d  walkers %6d  ⟨E_L⟩ %12.4f  E_T %12.4f\n", step, num_walkers, E_loc_mean, E_T)
        end
    end

    # === Reduce thread-local observables ===
    hist_1d_total .= sum(thread_hist_1d_storage)
    hist_2d_total .= sum(thread_hist_2d_storage)
    SSF_total .= sum(thread_SSF_storage)

    # === Normalize histograms and SSF ===
    total_hist_1d_weight = sum(hist_1d_total) * dx
    hist_1d_normalized = total_hist_1d_weight > 0 ? hist_1d_total ./ total_hist_1d_weight : zeros(Float64, num_bins)

    total_hist_2d_weight = sum(hist_2d_total) * dx^2
    hist_2d_normalized = total_hist_2d_weight > 0 ? hist_2d_total ./ total_hist_2d_weight : zeros(Float64, num_bins, num_bins)

    SSF_normalized = total_walkers_sampled > 0 && num_particles > 0 ? SSF_total ./ (num_particles * total_walkers_sampled) : zeros(ComplexF64, length(k_vec))

    # ρ₁(x,x') normalized so ∫ρ₁(x,x)dx = N (same convention as VMC `metropolis`)
    obdm = sum(thread_obdm_storage)
    W_obdm = sum(thread_obdm_weight)
    if compute_obdm && W_obdm > 0
        dx_obdm = L / nb_obdm
        obdm .*= L / (W_obdm * dx_obdm^2)
        obdm .= (obdm .+ obdm') ./ 2          # Ψ_T real ⇒ ρ₁ symmetric
    end

    # === Final energy (after burn-in) ===
    energy = n_burn >= num_steps ? NaN : mean(energies[n_burn+1:end])

    @printf("Normalization of 1D histogram: %.4f\n", sum(hist_1d_normalized) * dx)
    @printf("Normalization of 2D histogram: %.4f\n", sum(hist_2d_normalized) * dx^2)
    if compute_obdm && W_obdm > 0
        @printf("OBDM trace ∫ρ₁(x,x)dx: %.4f (expected N = %d)\n", sum(obdm[i, i] for i in 1:nb_obdm) * L / nb_obdm, num_particles)
    end

    n_cell = sum(thread_cell_storage)
    n_cell = sum(n_cell) > 0 ? n_cell ./ mean(n_cell) : n_cell

    return energies, energy, hist_1d_normalized, hist_2d_normalized, SSF_normalized, walker_positions, obdm, n_cell
end
