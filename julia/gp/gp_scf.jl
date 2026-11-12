# Gross-Pitaevskii (single-orbital Hartree) benchmark by self-consistent field for the
# 1D cavity Bose gas, H = sum_i -1/2 d_i^2 + (V0_kac/N) sum_{i<j} cos(kL x_i) cos(kL x_j).
#
# Product state phi^N, phi(x) = sum_n c_n exp(i n kL x)/sqrt(L), c_n real, sum c_n^2 = 1:
#   E/N = epsk sum_n n^2 c_n^2 + (V0/2) (N-1)/N Theta1^2,   Theta1 = sum_n c_n c_{n+1}
# One-body h[A]: diagonal epsk n^2, off-diagonal A/2, with A = V0 (N-1)/N Theta1.
# The nontrivial root of f(A) = A - V0 (N-1)/N Theta1(A) is found with Brent.
#
# Usage (from the repository root):  julia julia/gp/gp_scf.jl
# Output: numpy_arrays_GP/gp_scf_N<N>.csv and gp_scf_Ninf.csv (+ c_n files gp_cn_*.csv)

using LinearAlgebra, Roots, Printf, DelimitedFiles, NPZ

const EPSK = 2π^2                      # hbar^2 kL^2 / 2m in units eps = hbar^2/(m lambda^2)
const V0_GRID = [-15., -20., -25., -30., -35., -40., -50., -60., -80., -100., -150., -200.]

struct GPState
    c::Vector{Float64}                 # c_n, n = -nmax:nmax, c_0 > 0, sum c^2 = 1
    A::Float64
    theta1::Float64
    e::Float64                         # E/N
    s::Float64                         # overlap of the two Z2 branches
end

# lowest eigenvector of h[A] and Theta1(A)
function ground(A, nmax)
    n = -nmax:nmax
    H = SymTridiagonal(EPSK .* Float64.(n) .^ 2, fill(A / 2, 2nmax))
    c = eigen(H, 1:1).vectors[:, 1]
    c .*= sign(c[nmax + 1])
    return c, sum(c[1:end-1] .* c[2:end])
end

function solve_gp(V0, N; nmax = 30)
    g = isfinite(N) ? (N - 1) / N : 1.0
    n = -nmax:nmax
    f(A) = A - V0 * g * ground(A, nmax)[2]
    A = 0.0
    if V0 < 0 && f(V0 * g) * f(-1e-8) < 0
        A = find_zero(f, (V0 * g, -1e-8), Roots.Brent(); xatol = 1e-14, rtol = 1e-14)
    end                                # no sign change -> uniform state
    c, th = A == 0.0 ? (ground(0.0, nmax)[1], 0.0) : ground(A, nmax)
    e = EPSK * sum(n .^ 2 .* c .^ 2) + V0 / 2 * g * th^2
    s = sum((-1.0) .^ n .* c .^ 2)
    return GPState(c, A, th, e, s)
end

# OBDM occupations of the symmetric two-branch (cat) state
function cat_occupations(s, N)
    isfinite(N) || return (1 + s) / 2, (1 - s) / 2
    d = 2 * (1 + s^N)
    return (1 + s) * (1 + s^(N - 1)) / d, (1 - s) * (1 - s^(N - 1)) / d
end

# ---------------------------------------------------------------- DMC lookup (N = 40)
function dmc_energy(N, V0kac)
    V0 = V0kac / N
    dir = @sprintf "numpy_arrays_DMC/constant_density_1.0/energy/V0=%.3f" V0
    isdir(dir) || return nothing
    pat = Regex(raw"^energy_(\d+)_([\d.]+)_" * string(N) * raw"_(\d+)_(-?[\d.]+)_([\d.]+)(_IS)?\.npy$")
    fs = [joinpath(dir, f) for f in readdir(dir) if occursin(pat, f)]
    isempty(fs) && return nothing
    E, err = NPZ.npzread(argmax(mtime, fs))[1:2]
    return E / N, err / N
end

# N -> infinity DMC from thermodynamic_limit.py
function dmc_inf()
    f = "numpy_arrays_DMC/thermodynamic_limit/constant_density_1.0/e_inf_order1_nmin0.csv"
    isfile(f) || return Dict{Float64,Tuple{Float64,Float64}}()
    d = readdlm(f, ','; skipstart = 1)
    return Dict(d[i, 1] => (d[i, 2], d[i, 3]) for i in axes(d, 1))
end

# ---------------------------------------------------------------- checks
passed = Bool[]
function check(name, ok, detail = "")
    push!(passed, ok)
    @printf("[%s] %s %s\n", ok ? "PASS" : "FAIL", name, detail)
end

N = 40
println("epsk = $(round(EPSK, digits=4)), N = $N\n")

# 1. V0 = 0
st = solve_gp(0.0, N)
check("V0 = 0: c_n = delta_n0, E = 0, Theta1 = 0",
      st.c[31] ≈ 1 && abs(st.e) < 1e-14 && st.theta1 == 0, "(E=$(st.e))")

# 2. threshold
for (Nc, label) in ((N, "N=$N"), (Inf, "N=inf"))
    thr = EPSK * (isfinite(Nc) ? Nc / (Nc - 1) : 1.0)
    below = solve_gp(-thr * (1 - 1e-6), Nc).theta1
    above = solve_gp(-thr * (1 + 1e-3), Nc).theta1
    check("threshold $label at |V0| = $(round(thr, digits=4))", below == 0 && above > 0,
          "(Theta1 below = $below, above = $above)")
end

# 3. convergence in nmax
dmax = maximum(abs(solve_gp(V, N; nmax = 20).e - solve_gp(V, N; nmax = 40).e) for V in V0_GRID)
check("nmax 20 vs 40: |dE/N| < 1e-10", dmax < 1e-10, "(max diff = $dmax)")

# 4. E/N < 0 above threshold
neg = all(solve_gp(V, Nc).e < 0 for V in V0_GRID, Nc in (N, Inf))
check("E/N < 0 on the whole grid (below the uniform state)", neg)

# 5. variational ordering E_DMC <= E_GP (within 1 error bar)
einf = dmc_inf()
println("\nVariational ordering (E_DMC/N <= E_GP/N within 1 sigma):")
@printf("%8s %12s %12s %20s %12s %20s\n", "V0", "E_GP(40)", "E_GP(inf)", "E_DMC(N=40)", "", "E_DMC(N->inf)")
ord_ok = true
rows40 = Vector{Any}(); rowsinf = Vector{Any}()
for V in V0_GRID
    s40, sinf = solve_gp(V, N), solve_gp(V, Inf)
    d40 = dmc_energy(N, V)
    dinf = get(einf, V, nothing)
    ok40 = d40 === nothing || d40[1] <= s40.e + d40[2]
    okinf = dinf === nothing || dinf[1] <= sinf.e + dinf[2]
    global ord_ok &= ok40 && okinf
    @printf("%8.1f %12.5f %12.5f %12s %12s %12s %s\n", V, s40.e, sinf.e,
            d40 === nothing ? "n/a" : @sprintf("%.4f±%.4f", d40...), ok40 ? "ok" : "VIOLATED",
            dinf === nothing ? "n/a" : @sprintf("%.4f±%.4f", dinf...), okinf ? "ok" : "VIOLATED")
end
check("variational ordering E_DMC/N <= E_GP/N at every V0", ord_ok)

# ---------------------------------------------------------------- output
mkpath("numpy_arrays_GP")
for (Nc, tag) in ((N, "N$N"), (Inf, "Ninf"))
    sts = [solve_gp(V, Nc) for V in V0_GRID]
    occ = [cat_occupations(x.s, Nc) for x in sts]
    open("numpy_arrays_GP/gp_scf_$tag.csv", "w") do io
        println(io, "V0,A,Theta1,E_per_N,s,lambda_plus_per_N,lambda_minus_per_N")
        for (V, x, o) in zip(V0_GRID, sts, occ)
            @printf(io, "%.1f,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n", V, x.A, x.theta1, x.e, x.s, o[1], o[2])
        end
    end
    open("numpy_arrays_GP/gp_cn_$tag.csv", "w") do io     # one row per V0, columns n = -30..30
        println(io, "V0," * join(-30:30, ","))
        for (V, x) in zip(V0_GRID, sts)
            println(io, string(V) * "," * join(x.c, ","))
        end
    end
end
println("\nSaved numpy_arrays_GP/gp_scf_N$N.csv, gp_scf_Ninf.csv, gp_cn_*.csv")

# dense curves (1 eps steps) for plotting: E/N, Theta1, s, lambda_pm/N for N = 40 and N -> inf
let Vd = collect(0.0:-0.5:-210.0)
    open("numpy_arrays_GP/gp_dense.csv", "w") do io
        println(io, "V0,E_N40,Theta1_N40,s_N40,lp_N40,lm_N40,E_Ninf,Theta1_Ninf,s_Ninf,lp_Ninf,lm_Ninf")
        for V in Vd
            a, b = solve_gp(V, N), solve_gp(V, Inf)
            oa, ob = cat_occupations(a.s, N), cat_occupations(b.s, Inf)
            @printf(io, "%.1f,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n",
                    V, a.e, a.theta1, a.s, oa[1], oa[2], b.e, b.theta1, b.s, ob[1], ob[2])
        end
    end
end

all(passed) || (println("\nSOME CHECKS FAILED"); exit(1))
println("\nAll checks passed")
