# ------------------------------------------
# One-body density matrix
# ------------------------------------------

"""
    accumulate_obdm!(G, x_coord, x_tmp, num_part, L, nb_obdm, psi_interp, k_L, k_contact, α,
                     long_range, fermi_stats, reatto_chester, contact, weight=1.0)

McMillan estimator for the one-body density matrix. For each particle i, displaces it
to a uniform x' ∈ [-L/2, L/2) and adds weight·Ψ_T(x',…)/Ψ_T(x_i,…) to G[bin(x_i), bin(x')].
`weight` is 1 in VMC and the walker multiplicity M in DMC. The configuration is not
modified. Bosons only: the log-ratio uses abs, so the sign of Ψ_T is dropped.
"""
function accumulate_obdm!(G::Matrix{Float64}, x_coord::Vector{Float64}, x_tmp::Vector{Float64},
                          num_part::Int, L::Float64, nb_obdm::Int, psi_interp,
                          k_L::Float64, k_contact::Float64, α::Float64,
                          long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool,
                          weight::Float64 = 1.0)
    copyto!(x_tmp, x_coord)
    for i in 1:num_part
        xp = L * rand() - L/2
        x_tmp[i] = xp
        log_ratio = update_log_wave_function_after_move(x_coord, x_tmp, num_part, psi_interp, L,
                        k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        x_tmp[i] = x_coord[i]
        a = min(nb_obdm, max(1, Int(floor((x_coord[i] + L/2) / L * nb_obdm)) + 1))
        b = min(nb_obdm, max(1, Int(floor((xp        + L/2) / L * nb_obdm)) + 1))
        G[a, b] += weight * exp(log_ratio)
    end
    return nothing
end

# ------------------------------------------
# Final Metropolis Implementation
# ------------------------------------------

"""
    metropolis(num_part, num_steps, num_bins, delta, L, V0, k_lat, psi_interp, k_L, k_contact, α,
               long_range, fermi_stats, reatto_chester, contact;
               global_move_prob=0.2, step_block=10, compute_zoom=true,
               compute_energy=true, compute_ssf=false, compute_g2=false, compute_obdm=true,
               show_plot=false)

Runs a Metropolis VMC simulation of a 1D periodic system with a pair-product trial wavefunction.

# Input:
- `num_part::Int`: Number of particles.
- `num_steps::Int`: Number of Metropolis steps (single-particle or global moves).
- `num_bins::Int`: Number of bins for n(x) and g2.
- `delta::Float64`: Maximum displacement for single-particle moves.
- `L::Float64`: Length of the periodic box.
- `V0::Float64`: Bare (per-pair) strength of the cavity-mediated potential.
- `k_lat::Float64`: Cavity wavevector.
- `psi_interp`: Interpolated two-body wavefunction (Spline2D).
- `k_L`, `k_contact`, `α`: Reatto-Chester exponent, Bethe-Peierls parameter, Fermi exponent.
- `long_range`, `fermi_stats`, `reatto_chester`, `contact`: Trial-wavefunction components.

# Keyword arguments:
- `global_move_prob`: Probability of proposing the global Z₂ shift x → x + λ/2 (default 0.2).
  Set to 0.0 to sample a single density-wave branch.
- `step_block`: Steps between measurements. Use `num_part` for one measurement per sweep.
- `compute_zoom`: If `true`, n(x) and g2 are binned over one period [-0.5, 0.5);
  otherwise over the full box [-L/2, L/2).
- `compute_energy`, `compute_ssf`, `compute_g2`, `compute_obdm`: Switch observables on/off.
  Disabled observables are returned as zeros.
- `show_plot`: Plot the local-energy trace (avoid inside `@threads`).

# Output (in this order):
1. `Float64`: Mean local energy ⟨E_L⟩.
2. `Float64`: Mean ⟨E_L²⟩.
3. `Vector{ComplexF64}`: Static structure factor S(k), k = 2πn/L, n = 1 … 5L.
4. `Vector{Float64}`: n(x), normalized to unit integral over the binned interval.
5. `Matrix{Float64}`: Pair distribution g2(x, x'), normalized to unit integral.
6. `Float64`: Acceptance ratio (single-particle + global accepts) / num_steps.
7. `Vector{Float64}`: Local energy per particle at each measurement.
8. `Float64`: Mean kinetic energy.
9. `Float64`: Mean potential energy.
10. `Matrix{Float64}`: One-body density matrix ρ₁(x, x') on [-L/2, L/2)², with ∫ρ₁(x,x)dx = N.
11. `Vector{Float64}`: Z₂ order parameter Θ = (1/N) Σᵢ cos(k_lat xᵢ) at each measurement.
12. `Vector{Float64}`: Density folded into one cavity period (x mod λ, `n_cell_bins` bins),
    every step, normalized to mean 1 (input for the Leggett bound).

# Example
```julia
out = metropolis(10, 10^6, 100, 1.0, 10.0, -0.5, 2π, psi_interp, 0.0, 0.0, 1.0,
                 true, false, false, false; step_block=10, compute_energy=false)
obdm, Θ = out[10], out[11]
```
"""
function metropolis(num_part::Int, num_steps::Int, num_bins::Int, delta::Float64, L::Float64,
                    V0::Float64, k_lat::Float64, psi_interp, k_L::Float64, k_contact::Float64, α::Float64,
                    long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool;
                    global_move_prob::Float64 = 0.2, step_block::Int = 10, compute_zoom::Bool = true,
                    compute_energy::Bool = false, compute_ssf::Bool = true, compute_g2::Bool = false,
                    compute_obdm::Bool = true, show_plot::Bool = false, n_cell_bins::Int = 200)

    # ---------------- Binning for n(x) and g2 ----------------
    x_min    = compute_zoom ? -0.5 : -L/2
    x_range  = compute_zoom ?  1.0 :  L
    dx       = x_range / num_bins
    bin_of(x) = min(num_bins, max(1, Int(floor((x - x_min) / x_range * num_bins)) + 1))
    in_range(x) = x_min <= x < x_min + x_range

    hist_1d = zeros(Float64, num_bins)
    hist_2d = zeros(Float64, num_bins, num_bins)
    hist_cell = zeros(Float64, n_cell_bins)                # density folded into one cavity period
    λ_cav = 2π / k_lat
    cell_bin(x) = min(n_cell_bins, Int(floor(mod(x + λ_cav / 2, λ_cav) / λ_cav * n_cell_bins)) + 1)

    # ---------------- Measurement storage ----------------
    n_meas_max     = num_steps ÷ step_block
    E_local_values = zeros(Float64, n_meas_max)
    Θ_values       = zeros(Float64, n_meas_max)
    idx_meas       = 0

    E_tot = 0.0; E_sq = 0.0; E_kin = 0.0; E_pot = 0.0
    n_energy = 0

    k_ssf  = (2π / L) .* collect(1:floor(Int, 5L))
    SSF    = zeros(ComplexF64, length(k_ssf))
    n_ssf  = 0

    # One-body density matrix: own grid over the full box, ≥ 25 bins per cavity period
    nb_obdm = max(num_bins, round(Int, 25 * L * k_lat / (2π)))
    obdm    = zeros(Float64, compute_obdm ? nb_obdm : 1, compute_obdm ? nb_obdm : 1)
    n_obdm  = 0

    # ---------------- Initial configuration and equilibration ----------------
    shift_period = 2π / k_lat          # physical period of the cavity potential
    x_coord = random_initial_config(num_part, L)
    x_tmp   = similar(x_coord)
    logpsi_old = trial_log_wave_function(x_coord, num_part, psi_interp, L, k_L, k_contact, α,
                                         long_range, fermi_stats, reatto_chester, contact)

    n_equil = max(num_steps ÷ 5, num_part * 1000)
    for _ in 1:n_equil
        x_new, _ = move_one_part(x_coord, num_part, delta, L)
        log_ratio = update_log_wave_function_after_move(x_coord, x_new, num_part, psi_interp, L,
                        k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
        if log(rand()) < 2 * log_ratio
            x_coord = x_new
            logpsi_old += log_ratio
        end
    end

    # ---------------- Main Metropolis loop ----------------
    acc_single = 0
    global_move_attempts = 0
    global_move_accepts  = 0

    progress = Progress(num_steps; desc="Running Metropolis N=$num_part...", showspeed=true)
    for step in 1:num_steps
        next!(progress)

        if rand() < global_move_prob
            global_move_attempts += 1
            x_new = global_shift_move(x_coord, L, shift_period)
            logpsi_new = trial_log_wave_function(x_new, num_part, psi_interp, L, k_L, k_contact, α,
                                                 long_range, fermi_stats, reatto_chester, contact)
            if log(rand()) < 2 * (logpsi_new - logpsi_old)
                x_coord = x_new
                logpsi_old = logpsi_new
                global_move_accepts += 1
            end
        else
            x_new, _ = move_one_part(x_coord, num_part, delta, L)
            log_ratio = update_log_wave_function_after_move(x_coord, x_new, num_part, psi_interp, L,
                            k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
            if log(rand()) < 2 * log_ratio
                x_coord = x_new
                logpsi_old += log_ratio
                acc_single += 1
            end
        end

        # ---- n(x): every step ----
        for x in x_coord
            in_range(x) && (hist_1d[bin_of(x)] += 1)
            hist_cell[cell_bin(x)] += 1
        end

        # ---- g2(x, x'): every step, O(N²) ----
        if compute_g2
            for a in 1:num_part, b in (a + 1):num_part
                xa = x_coord[a]; xb = x_coord[b]
                if in_range(xa) && in_range(xb)
                    hist_2d[bin_of(xa), bin_of(xb)] += 1
                end
            end
        end

        # ---- Measurements every step_block steps ----
        if step % step_block == 0
            idx_meas += 1
            Θ_values[idx_meas] = sum(cos.(k_lat .* x_coord)) / num_part

            if compute_energy
                E_local, E_kinetic, E_potential = local_energy_log(x_coord, num_part, psi_interp, V0, k_lat, L,
                                                      k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
                if isnan(E_local)
                    @warn "NaN detected at step $step: E_local = $E_local"
                else
                    E_tot += E_local
                    E_sq  += E_local^2
                    E_kin += E_kinetic
                    E_pot += E_potential
                    n_energy += 1
                    E_local_values[idx_meas] = E_local / num_part
                end
            end

            if compute_ssf
                # Σ_ab e^{ik(x_a − x_b)} = |ρ_k|², ρ_k = Σ_a e^{ik x_a}: O(N) per k instead of O(N²)
                for ik in eachindex(k_ssf)
                    re = 0.0; imag_part = 0.0
                    for a in 1:num_part
                        s, c = sincos(k_ssf[ik] * x_coord[a])
                        re += c; imag_part += s
                    end
                    SSF[ik] += re^2 + imag_part^2
                end
                n_ssf += 1
            end

            if compute_obdm
                accumulate_obdm!(obdm, x_coord, x_tmp, num_part, L, nb_obdm, psi_interp,
                                 k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
                n_obdm += 1
            end
        end
    end

    println("N=$num_part, V0=$V0: global moves attempted $global_move_attempts, accepted $global_move_accepts")

    # ---------------- Normalization ----------------
    total_1d = sum(hist_1d)
    total_1d > 0 && (hist_1d ./= (total_1d * dx))

    total_2d = sum(hist_2d)
    total_2d > 0 && (hist_2d ./= (total_2d * dx^2))

    n_ssf > 0 && (SSF ./= n_ssf)

    # ρ₁(x,x') = <ψ†(x)ψ(x')>, normalized so ∫ρ₁(x,x)dx = N
    if n_obdm > 0
        dx_obdm = L / nb_obdm
        obdm  .*= L / (n_obdm * dx_obdm^2)
        obdm   .= (obdm .+ obdm') ./ 2          # Ψ_T real ⇒ ρ₁ symmetric
    end

    E_mean    = n_energy > 0 ? E_tot / n_energy : NaN
    E_sq_mean = n_energy > 0 ? E_sq  / n_energy : NaN
    E_kin_m   = n_energy > 0 ? E_kin / n_energy : NaN
    E_pot_m   = n_energy > 0 ? E_pot / n_energy : NaN
    acceptance_ratio = (acc_single + global_move_accepts) / num_steps

    if show_plot && compute_energy
        plt = plot(1:idx_meas, E_local_values[1:idx_meas], title="Evolution of Local Energy",
                   xlabel="Measurement", ylabel="E_L / N", legend=false, color=:blue)
        display(plt)
    end

    sum(hist_cell) > 0 && (hist_cell ./= sum(hist_cell) / n_cell_bins)

    return E_mean, E_sq_mean, SSF, hist_1d, hist_2d, acceptance_ratio,
           E_local_values[1:idx_meas], E_kin_m, E_pot_m, obdm, Θ_values[1:idx_meas], hist_cell
end