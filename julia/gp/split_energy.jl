# Two-orbital "split" (fragmented Fock state) energies built from the GP orbital.
#
# H = Σ_i (−½ ∂²_i) + (V0/N) Σ_{i<j} cos(2πx_i) cos(2πx_j)     (ħ = m = λ = 1, Kac-scaled V0)
#
# From the GP coefficients c_n (plane waves e^{i2πnx}):
#   c⁺ = even-n part of c, normalised;  c⁻ = odd-n part, normalised
#   φ_A = (φ⁺ + φ⁻)/√2,  φ_B = (φ⁺ − φ⁻)/√2   (orthonormal, φ_B(x) = φ_A(x + λ/2))
# cos couples n ↔ n±1 (even ↔ odd), so ⟨A|cos|B⟩ = 0 and the Fock state |n_A, n_B⟩ has
#   E(n_A)/N = T + V0/(2N²) Θ'² [(n_A − n_B)² − N],   T = 2π² Σ n² (c^A_n)²,  Θ' = Σ c^A_n c^A_{n+1}
#
# GP orbitals: N = 40 from numpy_arrays_GP/gp_cn_N40.csv; other N from solve_gp of
# julia/gp/gp_scf.jl (only its definitions are loaded — the script itself is not run).
#
# Usage (repo root):  julia julia/gp/split_energy.jl
# Output: numpy_arrays_GP/split_energy.csv (table), numpy_arrays_GP/split_energy_curves_N40.csv
#         (E(n_A)/N behind the figure), final_imag/split_energy_N40.pdf (via python/plots/plot_split.py)

using LinearAlgebra, Roots, Printf, DelimitedFiles, NPZ

# ---- definitions only from gp_scf.jl (const, struct, function); its top-level code is skipped
let ex = Meta.parseall(read(joinpath(@__DIR__, "gp_scf.jl"), String))
    for e in ex.args
        e isa Expr || continue
        h = e.head == :macrocall ? nothing : e.head
        if h in (:const, :struct, :function) ||
           (h == :(=) && e.args[1] isa Expr && e.args[1].head == :call)     # short-form functions
            Core.eval(Main, e)
        end
    end
end

const RUNS = vcat([(V, 40) for V in (-25.0, -30.0, -40.0, -60.0, -80.0, -100.0, -150.0)],
                  [(-150.0, N) for N in (15, 20, 25, 30)])

# ---- GP orbital c_n, n = -30..30
const CN40 = readdlm("numpy_arrays_GP/gp_cn_N40.csv", ','; skipstart = 1)
function gp_orbital(V0, N)
    if N == 40
        i = findfirst(isapprox(V0), CN40[:, 1])
        i === nothing || return Float64.(CN40[i, 2:end]), -30:30
    end
    st = solve_gp(V0, N)                       # nmax = 30 default in gp_scf.jl
    return st.c, -30:30
end

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

# ---- split energies
function split(c, n, V0, N)
    ev, od = iseven.(n), isodd.(n)
    cp = c .* ev; cp ./= norm(cp)
    cm = c .* od; cm ./= norm(cm)
    cA = (cp .+ cm) ./ sqrt(2)
    cB = (cp .- cm) ./ sqrt(2)
    cosmat(u, v) = (sum(u[1:end-1] .* v[2:end]) + sum(u[2:end] .* v[1:end-1])) / 2   # ⟨u|cos 2πx|v⟩
    T = 2π^2 * sum(n .^ 2 .* cA .^ 2)
    Θ = sum(cA[1:end-1] .* cA[2:end])
    E(nA) = T + V0 / (2N^2) * Θ^2 * ((2nA - N)^2 - N)
    return (; cA, cB, T, Θ, E, AB = dot(cA, cB), cosAB = cosmat(cA, cB), cosAA = cosmat(cA, cA),
              cosBB = cosmat(cB, cB), s = sum((-1.0) .^ n .* c .^ 2),
              TB = 2π^2 * sum(n .^ 2 .* cB .^ 2))
end

println("Checks")
rows, curves = [], Dict{Float64,Vector{Float64}}()
allok = true
for (V0, N) in RUNS
    c, n = gp_orbital(V0, N)
    egp = solve_gp(V0, N).e
    r = split(c, n, V0, N)
    Es = [r.E(nA) for nA in 0:N]
    ok = abs(r.AB) < 1e-12 && abs(r.cosAB) < 1e-12 && abs(r.cosAA + r.cosBB) < 1e-12 && abs(r.T - r.TB) < 1e-10 &&
         maximum(abs.(Es .- reverse(Es))) < 1e-12 &&
         argmin(Es) in (1, N + 1) && Es[1] ≈ minimum(Es) && Es[end] ≈ minimum(Es) &&
         Es[end] >= egp - 1e-12                                # φ_A^N is a product state: ≥ E_GP
    global allok &= ok
    @printf("  [%s] V0=%6.1f N=%2d  <A|B>=%.1e  <A|cos|B>=%.1e  sym=%.1e  min at nA∈{0,N}  E(all A)−E_GP = %.2e\n",
            ok ? "PASS" : "FAIL", V0, N, r.AB, r.cosAB, maximum(abs.(Es .- reverse(Es))), Es[end] - egp)
    nh = N ÷ 2                                                 # 50/50 (odd N: n_A = (N−1)/2)
    d = dmc_energy(N, V0)
    push!(rows, (V0, N, r.T, r.Θ, r.s, Es[end], Es[nh + 1], Es[nh + 1] - Es[end], N * (Es[nh + 1] - Es[end]), egp,
                 d === nothing ? NaN : d[1], d === nothing ? NaN : d[2],
                 d === nothing ? 0 : d[3], d === nothing ? NaN : d[4]))
    N == 40 && (curves[V0] = Es)
end
println(allok ? "All checks passed" : "SOME CHECKS FAILED")

open("numpy_arrays_GP/split_energy.csv", "w") do io
    println(io, "V0,N,T,Theta_prime,s,E_allA_per_N,E_5050_per_N,gap_per_N,gap_total,E_GP_per_N," *
                "E_DMC_per_N,E_DMC_err,DMC_walkers,DMC_tau")
    for r in rows
        @printf(io, "%.1f,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.10g,%.10g,%d,%.6g\n", r...)
    end
end
let Vs = sort(collect(keys(curves)), rev = true)
    open("numpy_arrays_GP/split_energy_curves_N40.csv", "w") do io
        println(io, "nA_over_N," * join([@sprintf("E_V0%.0f", V) for V in Vs], ","))
        for nA in 0:40
            println(io, @sprintf("%.6f", nA / 40), ",", join([@sprintf("%.12g", curves[V][nA+1]) for V in Vs], ","))
        end
    end
end

println("\n   V0   N        T       Θ'        s    E(all A)/N  E(50/50)/N   gap/N    gap(total)   E_GP/N      E_DMC/N")
for r in rows
    dmc = isnan(r[11]) ? "      —" : @sprintf("%9.4f±%.4f (%dw, τ=%g)", r[11], r[12], r[13], r[14])
    @printf("%6.1f %3d %9.4f %7.4f %8.5f %10.4f %10.4f %9.4f %10.2f %10.4f  %s\n",
            r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], dmc)
end
println("\nSaved numpy_arrays_GP/split_energy.csv and split_energy_curves_N40.csv")
