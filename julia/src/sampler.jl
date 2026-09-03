# ------------------------------------------
# Final Metropolis Implementation
# ------------------------------------------

"""
    metropolis(num_part, num_steps, num_bins, delta, L, V0, k_lat, psi_interp, k_L, k_contact, α,
               long_range, fermi_stats, reatto_chester, contact;
               global_move_prob=0.2, step_block=10, compute_zoom=true)
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
- `global_move_prob::Float64`: Probability of proposing a global Z₂ shift move (default 0.2).
- `step_block::Int`: Number of Metropolis steps between energy/SSF samples (default 10).
- `compute_zoom::Bool`: If `true` (default), also accumulate the zoomed-in 1D and 2D
  histograms (`hist_zoom`, `hist_2d_zoom`) over `[-0.5, 0.5)`, on top of the zoomed-out
  ones (`hist_1d`, `hist_2d`) over the full box. Set to `false` to skip that extra pass
  over all particle pairs when the zoomed diagnostics aren't needed — `hist_zoom` and
  `hist_2d_zoom` are then returned as all-zero arrays of the same shape.

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
E, E2, SSF, n_x, g2_xx, acc_ratio, E_trace = metropolis(8, 10^6, 100, 0.05, 1.0, 1.0, 2π, psi_interp, 2.0, 1.0, 1.0, true, true, false, false; step_block=20, compute_zoom=false)
```
"""

function metropolis(num_part::Int, num_steps::Int, num_bins::Int, delta::Float64, L::Float64,
                    V0::Float64, k_lat::Float64, psi_interp, k_L::Float64, k_contact::Float64, α::Float64,
                    long_range::Bool, fermi_stats::Bool, reatto_chester::Bool, contact::Bool,
                    global_move_prob::Float64 = 0.2, step_block::Int = 10, compute_zoom::Bool = true)
                    ::Tuple{Float64, Float64, Vector{ComplexF64}, Vector{Float64}, Matrix{Float64}, Float64, Vector{Float64}, Float64, Float64}
    acceptance_ratio = 0.0
    n_uncorr = 0
    bins = range(-L/2, stop=L/2, length=num_bins+1)
    hist_1d = zeros(Float64, num_bins)
    hist_2d = zeros(Float64, num_bins, num_bins)
    dx = L / num_bins
    E_tot = 0.0
    E_sq = 0.0
    E_kin = 0.0
    E_pot = 0.0
    E_local_values = Vector{Float64}(undef, (num_steps÷step_block))
    E_kin_values = Vector{Float64}(undef, (num_steps÷step_block))
    iter_val = Vector{Float64}(undef, (num_steps÷step_block))
    idx_plot = 1
    Θ = 0.0
    zoom_half_width = 0.5
    dx_zoom = (2*zoom_half_width) / num_bins
    dx_zoom2 = dx_zoom
    shift_period = 2π / k_lat   # physical period of the cavity potential
    global_move_attempts = 0
    global_move_accepts  = 0

    final_point = 5*L
    k = (2*π/L) * collect(1:1:final_point)
    SSF = zeros(ComplexF64, length(k))

    plt = plot(title=L"Evolution of Local Energy", xlabel=L"Step", ylabel=L"Local Energy", legend=false)

    x_coord = random_initial_config(num_part, L)
    
    logpsi_old = trial_log_wave_function(x_coord, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
    n_equil = max(num_steps ÷ 5, num_part * 1000)
    min_log_psi = Inf

    for i in 1:n_equil
            x_new, _ = move_one_part(x_coord, num_part, delta, L)
            log_ratio = update_log_wave_function_after_move(x_coord, x_new, num_part,
                            psi_interp, L, k_L, k_contact, α,
                            long_range, fermi_stats, reatto_chester, contact)
            if log(rand()) < 2 * log_ratio
                x_coord = x_new
                logpsi_old += log_ratio
            end
        end
            
        progress = Progress(num_steps; desc="Running Metropolis $num_part...", showspeed=true)
        for i in 1:num_steps
            next!(progress)  # Update progress bar

            if rand() < global_move_prob
                global_move_attempts += 1
                x_new = global_shift_move(x_coord, L, shift_period)
                logpsi_new = trial_log_wave_function(x_new, num_part, psi_interp, L, k_L, k_contact, α,
                                                    long_range, fermi_stats, reatto_chester, contact)
                log_ratio = logpsi_new - logpsi_old
                if log(rand()) < 2 * log_ratio
                    x_coord = x_new
                    logpsi_old = logpsi_new
                    global_move_accepts += 1
                end
            else
                x_new, moved_idx = move_one_part(x_coord, num_part, delta, L)

                log_ratio = update_log_wave_function_after_move(x_coord, x_new, num_part, psi_interp, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
                logw = 2 * log_ratio

                if log(rand()) < logw
                    x_coord = x_new
                    logpsi_old += log_ratio
                    acceptance_ratio += 1
                end
            end

            # 1D density
            if compute_zoom
                for x in x_coord
                    if -zoom_half_width <= x < zoom_half_width
                        bin_idx_zoom = min(num_bins,
                                        max(1, Int(floor((x + zoom_half_width) / (2*zoom_half_width) * num_bins)) + 1))
                        hist_1d[bin_idx_zoom] += 1
                    end
                end

            else
                for x in x_coord
                    bin_idx = min(num_bins, max(1, Int(floor((x + L/2) / L * num_bins)) + 1))
                    hist_1d[bin_idx] += 1
                end
            end

            #2D density
            if compute_zoom
                for ii in 1:num_part
                    for jj in (ii + 1):num_part
                        xi = x_coord[ii]
                        xj = x_coord[jj]
                        if (-zoom_half_width <= xi < zoom_half_width) && (-zoom_half_width <= xj < zoom_half_width)
                            bin_x_zoom = min(num_bins,
                                            max(1, Int(floor((xi + zoom_half_width) / (2*zoom_half_width) * num_bins)) + 1))
                            bin_y_zoom = min(num_bins,
                                            max(1, Int(floor((xj + zoom_half_width) / (2*zoom_half_width) * num_bins)) + 1))
                            hist_2d[bin_x_zoom, bin_y_zoom] += 1
                        end
                    end
                end
            else
                for i in 1:num_part
                    for j in (i + 1):num_part
                        bin_x = min(num_bins, max(1, Int(floor((x_coord[i] + L/2) / L * num_bins)) + 1))
                        bin_y = min(num_bins, max(1, Int(floor((x_coord[j] + L/2) / L * num_bins)) + 1))

                        hist_2d[bin_x, bin_y] += 1
                    end
                end
            end

                if i % step_block == 0
                    E_local, E_kinetic, E_potential = local_energy_log(x_coord, num_part, psi_interp, V0, k_lat, L, k_L, k_contact, α, long_range, fermi_stats, reatto_chester, contact)
                    # E_local, E_kinetic, E_potential = local_energy(x_coord, num_part, psi_interp, V0, k_lat, L, k_L, k_contact, fermi_stats, reatto_chester, contact)
                    E_tot += E_local
                    E_sq += E_local^2
                    E_kin += E_kinetic
                    E_pot += E_potential
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
                    E_kin_values[idx_plot] = E_kinetic
                    idx_plot += 1
                end
            end
            println("Global Moves attempted:", global_move_attempts)
            println("Global Moves accepted:", global_move_accepts)

            total_1d = sum(hist_1d)
            total_2d = sum(hist_2d)



            if compute_zoom
                hist_1d ./= (total_1d * dx_zoom)
                hist_2d ./= (total_2d * dx_zoom2^2)
            else
                hist_1d ./= (total_1d * dx)
                hist_2d ./= (total_2d * dx^2)
            end
            
            SSF ./= n_uncorr

            plot!(plt, iter_val, E_local_values, label=L"E", color=:blue)
            display(plt)

    return E_tot / n_uncorr, E_sq / n_uncorr, SSF, hist_1d, hist_2d, (acceptance_ratio + global_move_accepts) / num_steps, E_local_values, E_kin / n_uncorr, E_pot / n_uncorr
end