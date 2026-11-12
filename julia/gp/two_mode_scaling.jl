# N dependence of the two-mode correction near threshold (GP orbitals, no optimisation).
#
# For each N, (E2 − E_GP)/N on a grid in x = V0/|V0|_c(N), |V0|_c(N) = 2π² N/(N−1) (GP threshold at that N),
# x ∈ [−3, −0.8], denser near x = −1. GP is re-solved at every (V0, N) (solve_gp from gp_scf.jl, with the
# (N−1)/N factor), and the two-mode Hamiltonian of two_mode.jl is built from the same orbitals:
#   H(n_A, n_A) = N T + V0/(2N) Θ'² [(2n_A − N)² − N],   H(n_A+1, n_A) = t √((n_A+1)(N−n_A)).
# Only the lowest eigenvalue is needed, so instead of the dense symmetric-sector diagonalisation of
# solve_two_mode (O(N³), too slow for N = 10⁴) it is taken from the tridiagonal H directly
# (LAPACK stebz, O(N)); the ground state is the symmetric one, and deep in the phase the symmetric /
# antisymmetric pair is degenerate to machine precision, which does not affect the eigenvalue.
# Checked against two_mode() (two_mode.jl) at N = 40, 100, 400.
#
# Output: numpy_arrays_GP/two_mode_scaling_curves.csv (x, V0 and (E2 − E_GP)/N per N),
#         numpy_arrays_GP/two_mode_scaling_summary.csv (dip position and depth, values at x = −1, −1.5, −3)
# Usage (repo root):  julia julia/gp/two_mode_scaling.jl      (figures: python/plots/plot_two_mode_scaling.py)

using LinearAlgebra, Roots, Printf, DelimitedFiles, NPZ

function load_definitions(file)
    for e in Meta.parseall(read(joinpath(@__DIR__, file), String)).args
        e isa Expr || continue
        e.head == :macrocall && e.args[1] == GlobalRef(Core, Symbol("@doc")) && (e = e.args[end])
        if e.head in (:const, :struct, :function) ||
           (e.head == :(=) && e.args[1] isa Expr && e.args[1].head == :call)
            Core.eval(Main, e)
        end
    end
end
load_definitions("gp_scf.jl")
load_definitions("two_mode.jl")

const NS = [40, 100, 400, 1000, 4000, 10000]
const XGRID = sort(unique(vcat(range(-3.0, -1.3, length = 60), range(-1.3, -1.05, length = 80),
                               range(-1.05, -0.95, length = 80), range(-0.95, -0.8, length = 20))))
const XFIX = [-1.0, -1.5, -3.0]

vc(N) = 2π^2 * N / (N - 1)

"(E2 − E_GP)/N at coupling V0 and particle number N (GP orbitals, same construction as two_mode())"
function delta2(V0, N)
    st = solve_gp(V0, N)
    c, n = st.c, collect(NSET)
    cp, cm = c .* iseven.(n), c .* isodd.(n)
    norm(cm) < 1e-12 && (cm = Float64.(abs.(n) .== 1))       # uniform GP: φ⁻ = cos 2πx
    cp ./= norm(cp); cm ./= norm(cm)
    cA = (cp .+ cm) ./ sqrt(2)
    T = 2π^2 * sum(n .^ 2 .* cA .^ 2)
    t = 2π^2 * sum((-1.0) .^ n .* n .^ 2 .* cA .^ 2)
    Θ = sum(cA[1:end-1] .* cA[2:end])
    d = [N * T + V0 / (2N) * Θ^2 * ((2k - N)^2 - N) for k in 0:N]
    o = [t * sqrt((k + 1) * (N - k)) for k in 0:N-1]
    E2 = eigvals(SymTridiagonal(d, o), 1:1)[1] / N
    return E2 - st.e, E2, st.e
end

"golden-section minimum of f on [a, b]"
function golden(f, a, b; tol = 1e-7)
    g = (sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol
        if fc < fd
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

# ---- check against two_mode() (dense symmetric-sector solver)
println("Check against two_mode.jl")
let worst = 0.0
    for N in (40, 100, 400), x in (-3.0, -1.5, -1.1, -1.0, -0.9)
        V = x * vc(N)
        r = two_mode(V, N)
        dd, E2, _ = delta2(V, N)
        worst = max(worst, abs(E2 - r.E2))
    end
    @printf("  max |E2(tridiagonal) − E2(two_mode)| = %.1e  %s\n", worst, worst < 1e-10 ? "[PASS]" : "[FAIL]")
end

# ---- curves
curves = zeros(length(XGRID), length(NS))
summary = []
for (j, N) in enumerate(NS)
    tN = @elapsed begin
        for (i, x) in enumerate(XGRID)
            curves[i, j] = delta2(x * vc(N), N)[1]
        end
        i0 = argmin(curves[:, j])
        lo, hi = XGRID[max(i0 - 1, 1)], XGRID[min(i0 + 1, length(XGRID))]
        xmin, dmin = golden(x -> delta2(x * vc(N), N)[1], lo, hi)
        fix = [delta2(x * vc(N), N)[1] for x in XFIX]
    end
    push!(summary, (N, vc(N), xmin, xmin * vc(N), dmin, fix...))
    @printf("N = %5d  |V0|_c = %.4f  x_min = %.5f (V0 = %.3f)  depth = %.4e   at x=-1: %.4e  -1.5: %.4e  -3: %.4e  (%.1f s)\n",
            N, vc(N), xmin, xmin * vc(N), dmin, fix..., tN)
end

# ---- slopes of log|Δ| vs log N at the fixed x
println("\nlog-log slopes d ln|E2 − E_GP|/N / d ln N")
S = reduce(vcat, [collect(r)' for r in summary])
for (k, x) in enumerate(XFIX)
    y = log.(abs.(S[:, 5+k])); lx = log.(S[:, 1])
    A = hcat(ones(length(lx)), lx)
    p = A \ y
    loc = diff(y) ./ diff(lx)
    @printf("  x = %5.1f: fit over all N: slope = %.3f;  local slopes: %s\n", x, p[2], join([@sprintf("%.3f", l) for l in loc], ", "))
end
let y = log.(abs.(S[:, 5])), lx = log.(S[:, 1]), p = hcat(ones(length(lx)), lx) \ y, loc = diff(y) ./ diff(lx)
    @printf("  dip depth : fit over all N: slope = %.3f;  local slopes: %s\n", p[2], join([@sprintf("%.3f", l) for l in loc], ", "))
end
let y = log.(abs.(S[:, 3] .+ 1)), lx = log.(S[:, 1]), p = hcat(ones(length(lx)), lx) \ y, loc = diff(y) ./ diff(lx)
    @printf("  |x_min+1| : fit over all N: slope = %.3f;  local slopes: %s\n", p[2], join([@sprintf("%.3f", l) for l in loc], ", "))
end

open("numpy_arrays_GP/two_mode_scaling_curves.csv", "w") do io
    println(io, "x," * join(["Delta_N$(N)" for N in NS], ","))
    for i in eachindex(XGRID)
        println(io, @sprintf("%.10g", XGRID[i]), ",", join([@sprintf("%.12g", curves[i, j]) for j in eachindex(NS)], ","))
    end
end
open("numpy_arrays_GP/two_mode_scaling_summary.csv", "w") do io
    println(io, "N,Vc,x_min,V0_min,depth,Delta_x-1,Delta_x-1.5,Delta_x-3")
    for r in summary; println(io, join([@sprintf("%.12g", v) for v in r], ",")); end
end
println("\nSaved numpy_arrays_GP/two_mode_scaling_curves.csv and two_mode_scaling_summary.csv")
