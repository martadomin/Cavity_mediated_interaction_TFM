# Δτ and walker-number convergence tests for DMC.
#
# 1. Time-step scan at fixed num_walkers. With the quadratic (second-order) propagator
#    E(Δτ) = E(0) + b Δτ² (linear propagator: E(0) + b Δτ); the fit gives E(Δτ → 0) and
#    tells you which Δτ has a bias below the statistical error. That single Δτ is then
#    used for all production runs (run_dmc.jl).
# 2. Walker scan at fixed Δτ (set Δτ_walker_scan to the value chosen in step 1):
#    the population-control bias goes like 1/N_w; pick N_w where it is below the error.
#
# Each run covers the same total imaginary time τ_total, so smaller Δτ means more steps.
# Run this for a weak- and a strong-coupling point and the largest N you plan to use.
#
# Usage:  julia -t auto julia/dmc/convergence_dmc.jl   (from the repository root)

using Printf, NPZ, Plots, Statistics
using Base.Threads

include(joinpath(@__DIR__, "dmc_core.jl"))

# -----------------------------------------------
# Test points and scan parameters
# -----------------------------------------------
num_particles_array = [40]                    # hardest point: largest N used in DMC ...
V0_kac_list   = [-100.0]                      # ... and strongest coupling; bare V0 = V0_kac / N
density_val   = 1.0                           # L = N / density
k_lat         = 2π
α             = 1.0
num_bins      = 100
τ_total       = 3.0                           # imaginary time per run (relaxation time ~1/ω ≈ 0.01 at strong coupling)
burn_in_frac  = 0.2
z2_shift_prob = 0.1
quadratic     = true                          # second-order propagator (bias ∝ Δτ²)

Δτ_list          = [8e-3, 4e-3, 2e-3, 1e-3]
walkers_Δτ_scan  = 500
Δτ_walker_scan   = 2e-3
walkers_list     = [200, 500]

long_range, fermi_stats, reatto_chester, contact = true, false, false, false
importance_sampling = true
k_L = 0.0

"""
    weighted_linear_fit(x, y, σ) -> (a, b, σ_a, σ_b)

Least-squares fit y = a + b x with weights 1/σ².
"""
function weighted_linear_fit(x::Vector{Float64}, y::Vector{Float64}, σ::Vector{Float64})
    w = 1 ./ σ.^2
    S, Sx, Sy = sum(w), sum(w .* x), sum(w .* y)
    Sxx, Sxy = sum(w .* x.^2), sum(w .* x .* y)
    Δ = S * Sxx - Sx^2
    a = (Sxx * Sy - Sx * Sxy) / Δ
    b = (S * Sxy - Sx * Sy) / Δ
    return a, b, sqrt(Sxx / Δ), sqrt(S / Δ)
end

"""
    run_point(N, Δτ, N_w, E_T, ...) -> (E, σ_E, S(k_lat), final walker count)

One DMC run of total imaginary time τ_total; the OBDM is skipped for speed.
"""
function run_point(N, L, V0, ψ_2, k_contact, Δτ, N_w, E_T)
    num_steps = round(Int, τ_total / Δτ)
    energies, E, _, _, SSF, walkers, _ = DMC_part_w_inter(
        N, num_steps, N_w, num_bins, Δτ, E_T, α, V0, L, ψ_2, k_contact,
        long_range, fermi_stats, contact, importance_sampling, false;
        k_lat = k_lat, k_L = k_L, reatto_chester = reatto_chester,
        burn_in_frac = burn_in_frac, compute_obdm = false, z2_shift_prob = z2_shift_prob,
        quadratic = quadratic)
    σ_E = blocking_error(energies[floor(Int, burn_in_frac * num_steps)+1:end])
    S_klat = real(SSF[round(Int, L * k_lat / (2π))])   # S(k) at the cavity wavevector
    return E, σ_E, S_klat, size(walkers, 1)
end

for V0_kac in V0_kac_list, N in num_particles_array
    V0 = V0_kac / N
    L = N / density_val
    k_contact = contact ? find_k_contact(L, -1.0) : 0.0

    # --- Trial wavefunction: periodic spline, identical to run_vmc.jl / run_dmc.jl ---
    ψ_2 = 0.0
    if long_range
        psi_path = @sprintf "numpy_arrays_VMC/wavefunction/V0=%.3f/wavefunction_%.3f_%.1f.npy" V0 V0 1.0
        isfile(psi_path) || error("Wavefunction file not found: $psi_path")
        ψ_2 = periodic_wave_function(NPZ.npzread(psi_path))
    end

    # --- Initial trial energy: saved VMC energy (julia/vmc/run_vmc_energy.jl), else a short VMC run ---
    E_vmc_path = @sprintf "numpy_arrays_VMC/constant_density_%.1f/energy/V0=%.3f/energy_%d_%.3f_%.1f.npy" density_val V0 N V0 L
    if isfile(E_vmc_path)
        E_T = Float64(first(NPZ.npzread(E_vmc_path)))   # scalar or 0-d array
    else
        E_T = metropolis(N, 10^4 * N, num_bins, 1.0, L, V0, k_lat, ψ_2, k_L, k_contact, α,
                         long_range, fermi_stats, reatto_chester, contact;
                         step_block = N, compute_zoom = false, compute_energy = true,
                         compute_ssf = false, compute_obdm = false)[1]
    end
    println("\nN = $N, V0 = $V0, L = $L, VMC energy = $E_T", isfile(E_vmc_path) ? " (from $E_vmc_path)" : " (short VMC run)")

    out_dir  = @sprintf "numpy_arrays_DMC/convergence/constant_density_%.1f/V0=%.3f" density_val V0
    plot_dir = @sprintf "imag_DMC/convergence/constant_density_%.1f/V0=%.3f" density_val V0
    mkpath(out_dir); mkpath(plot_dir)
    tag = @sprintf "%d_%.3f_%.1f_tau%.1f" N V0 L τ_total

    # -----------------------------------------------
    # 1. Time-step scan
    # -----------------------------------------------
    nΔ = length(Δτ_list)
    E_Δ, σ_Δ, S_Δ = zeros(nΔ), zeros(nΔ), zeros(nΔ)
    for (i, Δτ) in enumerate(Δτ_list)
        println("\n--- Δτ = $Δτ, N_w = $walkers_Δτ_scan ---")
        E_Δ[i], σ_Δ[i], S_Δ[i], _ = run_point(N, L, V0, ψ_2, k_contact, Δτ, walkers_Δτ_scan, E_T)
        @printf("E = %.6f ± %.6f,  S(k_lat) = %.4f\n", E_Δ[i], σ_Δ[i], S_Δ[i])
    end
    p = quadratic ? 2 : 1                       # bias ∝ Δτ^p
    a, b, σa, σb = weighted_linear_fit(Δτ_list .^ p, E_Δ, σ_Δ)
    @printf("\nFit E = E(0) + b·Δτ^%d: E(Δτ→0) = %.6f ± %.6f, b = %.4g ± %.4g\n", p, a, σa, b, σb)
    println("Δτ    bias |b·Δτ^$p|    stat. error")
    for (i, Δτ) in enumerate(Δτ_list)
        bias = abs(b * Δτ^p)
        @printf("%.1e   %.2e         %.2e %s\n", Δτ, bias, σ_Δ[i], bias < σ_Δ[i] ? "  ✓" : "")
    end

    NPZ.npzwrite(joinpath(out_dir, "dtau_scan_" * (quadratic ? "quad_" : "lin_") * tag * ".npz"),
                 Dict("dtau" => Δτ_list, "E" => E_Δ, "E_err" => σ_Δ, "S_klat" => S_Δ,
                      "fit" => [a, b, σa, σb], "power" => [Float64(p)], "num_walkers" => [walkers_Δτ_scan]))

    plt = scatter(Δτ_list, E_Δ, yerror=σ_Δ, label=quadratic ? "quadratic DMC" : "linear DMC", xlabel="Δτ", ylabel="E",
                  title="Time-step convergence (N=$N, V0=$(round(V0, digits=3)))", xlims=(0, 1.05 * maximum(Δτ_list)))
    xs = range(0, maximum(Δτ_list), length=50)
    plot!(plt, xs, a .+ b .* xs .^ p, label=@sprintf("fit, E(0) = %.5f ± %.5f", a, σa), ls=:dash)
    hline!(plt, [E_T], label="VMC", ls=:dot)
    savefig(plt, joinpath(plot_dir, "dtau_scan_" * (quadratic ? "quad_" : "lin_") * tag * ".png"))

    # -----------------------------------------------
    # 2. Walker-number scan
    # -----------------------------------------------
    nW = length(walkers_list)
    E_W, σ_W, S_W = zeros(nW), zeros(nW), zeros(nW)
    for (i, N_w) in enumerate(walkers_list)
        println("\n--- Δτ = $Δτ_walker_scan, N_w = $N_w ---")
        E_W[i], σ_W[i], S_W[i], _ = run_point(N, L, V0, ψ_2, k_contact, Δτ_walker_scan, N_w, E_T)
        @printf("E = %.6f ± %.6f,  S(k_lat) = %.4f\n", E_W[i], σ_W[i], S_W[i])
    end
    inv_W = 1 ./ Float64.(walkers_list)
    aW, bW, σaW, σbW = weighted_linear_fit(inv_W, E_W, σ_W)
    @printf("\nLinear fit in 1/N_w: E(N_w→∞) = %.6f ± %.6f, slope = %.4f ± %.4f\n", aW, σaW, bW, σbW)

    NPZ.npzwrite(joinpath(out_dir, "walker_scan_" * tag * ".npz"),
                 Dict("num_walkers" => Float64.(walkers_list), "E" => E_W, "E_err" => σ_W, "S_klat" => S_W,
                      "fit" => [aW, bW, σaW, σbW], "dtau" => [Δτ_walker_scan]))

    pltW = scatter(inv_W, E_W, yerror=σ_W, label="DMC", xlabel="1 / N_w", ylabel="E",
                   title="Walker convergence (N=$N, Δτ=$Δτ_walker_scan)", xlims=(0, 1.05 * maximum(inv_W)))
    xs = range(0, maximum(inv_W), length=50)
    plot!(pltW, xs, aW .+ bW .* xs, label=@sprintf("fit, E(∞) = %.5f ± %.5f", aW, σaW), ls=:dash)
    savefig(pltW, joinpath(plot_dir, "walker_scan_" * tag * ".png"))

    println("\nResults saved to $out_dir, plots to $plot_dir")
end
