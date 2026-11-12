# Two-mode (two-orbital) ground state in the split basis φ_A, φ_B of julia/gp/split_energy.jl.
#
# H = Σ_i (−½ ∂²_i) + (V0/N) Σ_{i<j} cos(2πx_i) cos(2πx_j)     (ħ = m = λ = 1, Kac-scaled V0)
#
# φ_A = (φ⁺ + φ⁻)/√2, φ_B = (φ⁺ − φ⁻)/√2, φ± = even/odd-n parts of the GP orbital (normalised).
# In the basis |n_A, N − n_A⟩ (exact projection of H on span{φ_A, φ_B}, since ⟨A|cos|B⟩ = 0):
#   H(n_A, n_A)     = N T + V0/(2N) Θ'² [(2n_A − N)² − N]
#   H(n_A+1, n_A)   = t √((n_A+1)(N−n_A)),   t = ⟨A|T|B⟩ = 2π² Σ (−1)^n n² (c^A_n)²
# The GP product state lies in this space, so E2 ≤ E_GP(N); E2 ≥ E_exact (variational).
# Uniform GP (no odd part, below threshold): φ⁻ = cos 2πx, i.e. c⁻ = (δ_{n,1} + δ_{n,−1})/√2.
#
# GP orbitals: solve_gp from julia/gp/gp_scf.jl (definitions only; checked against gp_cn_N40.csv).
#
# Usage (repo root):  julia julia/gp/two_mode.jl
# Output: numpy_arrays_GP/two_mode.csv, two_mode_dense_N40.csv, two_mode_weights_N40.csv
#         (figures: python/plots/plot_two_mode.py)

using LinearAlgebra, Roots, Printf, DelimitedFiles, NPZ

# ---- definitions only from gp_scf.jl (const, struct, function); its top-level code is skipped
let ex = Meta.parseall(read(joinpath(@__DIR__, "gp_scf.jl"), String))
    for e in ex.args
        e isa Expr || continue
        if e.head in (:const, :struct, :function) ||
           (e.head == :(=) && e.args[1] isa Expr && e.args[1].head == :call)
            Core.eval(Main, e)
        end
    end
end

const V0_RUNS = [-20.0, -25.0, -30.0, -35.0, -40.0, -50.0, -60.0, -80.0, -100.0, -150.0]
const N_OTHER = [15, 20, 25, 30, 35]          # added to the table where a τ ≥ 32 DMC run exists
const V0_DIP = [-21.0, -22.0, -23.0]          # extra N = 40 DMC points at the two-mode dip (table only)
const NSET = -30:30

# ---- best available DMC energy (longest imaginary time, then most walkers)
function dmc_energy(N, V0kac)
    V0 = V0kac / N
    dir = @sprintf "numpy_arrays_DMC/constant_density_1.0/energy/V0=%.3f" V0
    isdir(dir) || return nothing
    pat = Regex(raw"^energy_(\d+)_([\d.]+)_" * string(N) * raw"_(\d+)_(-?[\d.]+)_([\d.]+)_IS\.npy$")
    best = nothing
    for f in readdir(dir)
        m = match(pat, f); m === nothing && continue
        w, dt, st = parse(Int, m[1]), parse(Float64, m[2]), parse(Int, m[3])
        key = (round(st * dt, digits = 6), w)
        (best === nothing || key > best[1]) && (best = (key, joinpath(dir, f), w, st * dt))
    end
    best === nothing && return nothing
    E, err = NPZ.npzread(best[2])[1:2]
    return E / N, err / N, best[3], best[4]
end

# ---- QMC OBDM occupations at N = 40 (python/obdm_momentum.py): estimator 0 = VMC, 1 = DMC, 2 = lin
function qmc_lambdas()
    f = "numpy_arrays_OBDM/summary_N40.csv"
    d = Dict{Tuple{Float64,Int},Tuple{Float64,Float64}}()
    isfile(f) || return d
    m = readdlm(f, ','; skipstart = 1)
    for i in axes(m, 1)
        d[(m[i, 1], Int(m[i, 2]))] = (m[i, 6], m[i, 7])
    end
    return d
end

# ---- split basis and two-mode problem
function two_mode(V0, N)
    st = solve_gp(V0, N)
    c, n = st.c, collect(NSET)
    cp = c .* iseven.(n)
    cm = c .* isodd.(n)
    norm(cm) < 1e-12 && (cm = Float64.(abs.(n) .== 1))       # uniform GP: φ⁻ = cos 2πx
    cp ./= norm(cp); cm ./= norm(cm)
    cA = (cp .+ cm) ./ sqrt(2)
    T = 2π^2 * sum(n .^ 2 .* cA .^ 2)
    t = 2π^2 * sum((-1.0) .^ n .* n .^ 2 .* cA .^ 2)
    Θ = sum(cA[1:end-1] .* cA[2:end])
    r = solve_two_mode(T, t, Θ, V0, N)
    return (; E2 = r.E2, egp = st.e, s = st.s, T, t, Θ, w = r.w, λp = r.λp, λm = r.λm, c)
end

"""
Ground state of the two-mode Hamiltonian for given (T, t, Θ'):
    H(n_A, n_A) = N T + V0/(2N) Θ'² [(2n_A − N)² − N],  H(n_A+1, n_A) = t √((n_A+1)(N−n_A)).
Returns E2/N, the weights |ψ(n_A)|², λ±/N, g = ⟨a_A†a_B⟩ and W = V0/(2N) ⟨(2n_A − N)² − N⟩
(so that E2 = λ₊ε₊ + λ₋ε₋ + W Θ'² by Hellmann–Feynman).
"""
function solve_two_mode(T, t, Θ, V0, N)
    nA = 0:N
    d = [N * T + V0 / (2N) * Θ^2 * ((2k - N)^2 - N) for k in nA]
    o = [t * sqrt((k + 1) * (N - k)) for k in 0:N-1]
    # H commutes with n_A ↔ N − n_A and its ground state is the symmetric one (t < 0). Deep in the
    # phase the symmetric/antisymmetric splitting is ∝ t^N, below machine precision, so diagonalise
    # in the symmetric sector: basis (|k⟩ + |N−k⟩)/√2, k < N/2, plus |N/2⟩ for even N.
    H = Matrix(SymTridiagonal(d, o))
    K = 0:N÷2
    U = zeros(N + 1, length(K))
    for (j, k) in enumerate(K)
        k == N - k ? (U[k+1, j] = 1.0) : (U[k+1, j] = U[N-k+1, j] = 1 / sqrt(2))
    end
    F = eigen(Symmetric(U' * H * U))
    ψ = U * F.vectors[:, 1]
    Efull = eigmin(Symmetric(H))
    abs(F.values[1] - Efull) < 1e-8 * max(1, abs(Efull)) || @warn "symmetric-sector ground state above full spectrum" V0 N
    ρAA = sum(abs2.(ψ) .* nA)
    ρAB = sum(ψ[2:end] .* ψ[1:end-1] .* sqrt.((nA[1:end-1] .+ 1) .* (N .- nA[1:end-1])))
    λ = eigvals(Symmetric([ρAA ρAB; ρAB N-ρAA]))
    W = V0 / (2N) * sum(abs2.(ψ) .* [(2k - N)^2 - N for k in nA])
    return (; E2 = F.values[1] / N, w = abs2.(ψ), λp = λ[2] / N, λm = λ[1] / N, g = ρAB, W)
end

# ---------------------------------------------------------------- checks
println("Checks")
ok = Ref(true)
chk(name, cond, info = "") = (@printf("  [%s] %s %s\n", cond ? "PASS" : "FAIL", name, info); ok[] &= cond)

r0 = two_mode(0.0, 40)
chk("V0 = 0: all particles in φ⁺ (λ+ = N), E2 = 0", abs(r0.λp - 1) < 1e-10 && abs(r0.E2) < 1e-10,
    @sprintf("(λ+/N = %.12f, E2/N = %.1e)", r0.λp, r0.E2))
CN40 = readdlm("numpy_arrays_GP/gp_cn_N40.csv", ','; skipstart = 1)
dc = maximum(maximum(abs.(two_mode(V, 40).c .- CN40[findfirst(isapprox(V), CN40[:, 1]), 2:end])) for V in V0_RUNS)
chk("GP orbitals = gp_cn_N40.csv", dc < 1e-12, @sprintf("(max |Δc| = %.1e)", dc))
r = two_mode(-150.0, 40)
wend = r.w[1] + r.w[end]
chk("V0 = −150: weight at n_A ∈ {0, N}", wend > 0.5, @sprintf("(w(0) + w(N) = %.4f)", wend))
chk("V0 = −150: λ± ≈ N(1 ± s)/2", abs(r.λp - (1 + r.s) / 2) < 0.01,
    @sprintf("(λ+/N = %.5f, (1+s)/2 = %.5f; λ−/N = %.5f, (1−s)/2 = %.5f)", r.λp, (1 + r.s) / 2, r.λm, (1 - r.s) / 2))
chk("V0 = −150: E2 ≈ E_GP", abs(r.E2 - r.egp) < 0.02, @sprintf("(E2/N − E_GP/N = %.2e)", r.E2 - r.egp))
viol = [(V, N) for V in V0_RUNS, N in vcat(N_OTHER, 40) if (x = two_mode(V, N); x.E2 > x.egp + 1e-10)]
chk("E2 ≤ E_GP(N) everywhere (GP state lies in the two-mode space)", isempty(viol), isempty(viol) ? "" : string(viol))

# ---------------------------------------------------------------- table
QL = qmc_lambdas()
rows = []
for N in vcat(40, N_OTHER), V in (N == 40 ? sort(vcat(V0_RUNS, V0_DIP), rev = true) : V0_RUNS)
    x = two_mode(V, N)
    d = dmc_energy(N, V)
    N != 40 && (d === nothing || d[4] < 32 - 1e-9) && continue
    q = [get(QL, (V, k), (NaN, NaN)) for k in 0:2]
    push!(rows, (V, N, x.egp, x.E2, d === nothing ? NaN : d[1], d === nothing ? NaN : d[2],
                 d === nothing ? NaN : d[1] - x.E2, x.λp, x.λm,
                 (N == 40 ? vcat(collect.(q)...) : fill(NaN, 6))...,
                 x.s, x.T, x.t, x.Θ, d === nothing ? 0 : d[3], d === nothing ? NaN : d[4]))
end
open("numpy_arrays_GP/two_mode.csv", "w") do io
    println(io, "V0,N,E_GP_per_N,E2_per_N,E_DMC_per_N,E_DMC_err,E_DMC_minus_E2,lambda_plus_2m,lambda_minus_2m," *
                "lambda0_VMC,lambda1_VMC,lambda0_DMC,lambda1_DMC,lambda0_lin,lambda1_lin,s,T,t,Theta_prime,DMC_walkers,DMC_tau")
    for r in rows; println(io, join([@sprintf("%.12g", v) for v in r], ",")); end
end

println(ok[] ? "All checks passed" : "SOME CHECKS FAILED")
println("\n    V0   N     E_GP/N       E2/N    E_DMC/N             E_DMC−E2          λ±/N (2-mode)     λ0,λ1/N VMC | DMC | lin   [DMC run]")
for r in rows
    dm = isnan(r[5]) ? "        —                          " :
         @sprintf("%9.4f±%.4f  %+8.4f±%.4f (%4.1fσ)", r[5], r[6], r[7], r[6], abs(r[7]) / r[6])
    ql = isnan(r[10]) ? "" : @sprintf("%.3f,%.3f | %.3f,%.3f | %.3f,%.3f", r[10:15]...)
    run = r[20] == 0 ? "" : @sprintf("[%dw τ=%g]", r[20], r[21])
    @printf("%6.1f %3d %10.4f %10.4f  %s   %.4f,%.4f   %s  %s\n", r[1], r[2], r[3], r[4], dm, r[8], r[9], ql, run)
end

# ---------------------------------------------------------------- figure data (N = 40)
open("numpy_arrays_GP/two_mode_dense_N40.csv", "w") do io
    println(io, "V0,E_GP_per_N,E2_per_N,lambda_plus_2m,lambda_minus_2m")
    for V in 0.0:-0.5:-60.0
        x = two_mode(V, 40)
        @printf(io, "%.1f,%.12g,%.12g,%.12g,%.12g\n", V, x.egp, x.E2, x.λp, x.λm)
    end
end
let Vw = [-25.0, -40.0, -60.0, -150.0], W = [two_mode(V, 40).w for V in Vw]
    open("numpy_arrays_GP/two_mode_weights_N40.csv", "w") do io
        println(io, "nA_over_N," * join([@sprintf("w_V0%.0f", V) for V in Vw], ","))
        for k in 0:40
            println(io, @sprintf("%.6f", k / 40), ",", join([@sprintf("%.12g", w[k+1]) for w in W], ","))
        end
    end
end
println("\nSaved numpy_arrays_GP/two_mode.csv, two_mode_dense_N40.csv, two_mode_weights_N40.csv")
