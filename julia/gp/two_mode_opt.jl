# Self-consistent (orbital-optimised) two-mode model.
#
# Same Hamiltonian as julia/gp/two_mode.jl, but the orbitals are varied instead of fixed from GP:
#   φ⁺ = Σ_{n even} d⁺_n e^{2πinx},  φ⁻ = Σ_{n odd} d⁻_n e^{2πinx},  d^±_{−n} = d^±_n real, Σ (d^±_n)² = 1,
#   χ_{A,B} = (φ⁺ ± φ⁻)/√2   (⟨χ_A|cos|χ_B⟩ = 0 for any such pair, so H2 below is the exact projection),
#   H2 = N T + t (a_A†a_B + h.c.) + V0/(2N) Θ'² [(n_A − n_B)² − N],
#   T = (ε₊ + ε₋)/2,  t = (ε₊ − ε₋)/2,  ε± = 2π² Σ n² (d^±_n)²,  Θ' = ½ Σ d⁺_n (d⁻_{n+1} + d⁻_{n−1}).
# E2(d⁺, d⁻) = lowest eigenvalue in the symmetric sector (solve_two_mode from two_mode.jl), minimised over
# d^± = v^±/|v^±| with independent v⁺_n (n = 0, 2, …) and v⁻_n (n = 1, 3, …), |n| ≤ NM.
# Since H2 is the projection of H on span{χ_A, χ_B}, E2_opt is a variational upper bound on E0 for every
# choice of orbitals, and E2_opt ≤ E2(GP orbitals) ≤ E_GP.
#
# Minimiser: BFGS with Armijo backtracking (no extra packages). Gradient by Hellmann–Feynman,
#   dE2 = n₊ dε₊ + n₋ dε₋ + 2 W Θ' dΘ',  n± = N/2 ± ⟨a_A†a_B⟩,  W = V0/(2N) ⟨(n_A − n_B)² − N⟩,
# projected on the tangent of the normalisation sphere (checked against finite differences below).
#
# Usage (repo root):  julia julia/gp/two_mode_opt.jl
# Output: numpy_arrays_GP/two_mode_opt.csv, two_mode_opt_dense_N40.csv, two_mode_opt_orbitals.csv
#         (figure: python/plots/plot_two_mode_opt.py)

using LinearAlgebra, Roots, Printf, DelimitedFiles, NPZ, Logging

# ---- definitions only (const, struct, function, documented function) from gp_scf.jl and two_mode.jl
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

const NM = 24                                  # |n| ≤ NM: 13 even, 12 odd coefficients (checked vs NM_CHECK)
const NM_CHECK = 32
const RUNS_N40 = [-20.0, -22.0, -25.0, -30.0, -40.0, -60.0]
const RUNS_NV = [(N, V) for N in (15, 20, 25, 30), V in (-25.0, -30.0)]

# ---- parameters ↔ orbitals
evens(nm) = 0:2:nm
odds(nm) = 1:2:nm
npar(nm) = length(evens(nm)) + length(odds(nm))

"full symmetric vectors on n = −nm … nm (unnormalised) from x = [v⁺ (n = 0, 2, …); v⁻ (n = 1, 3, …)]"
function orbitals(x, nm)
    P, M = zeros(2nm + 1), zeros(2nm + 1)
    ke = evens(nm)
    for (j, k) in enumerate(ke); P[nm+1+k] = P[nm+1-k] = x[j]; end
    for (j, k) in enumerate(odds(nm)); M[nm+1+k] = M[nm+1-k] = x[length(ke)+j]; end
    return P, M
end

"x from full coefficient vectors c on n = −nc … nc (even part → v⁺, odd part → v⁻), truncated at nm"
function params(cp, cm, nm)
    nc = (length(cp) - 1) ÷ 2
    c(v, k) = k <= nc ? v[nc+1+k] : 0.0
    vcat([c(cp, k) for k in evens(nm)], [c(cm, k) for k in odds(nm)])
end

up(v) = [v[2:end]; 0.0]                          # v_{n+1}
dn(v) = [0.0; v[1:end-1]]                        # v_{n−1}

function orbital_data(x, nm)
    P, M = orbitals(x, nm)
    p, m = P / norm(P), M / norm(M)
    n = collect(-nm:nm)
    εp, εm = 2π^2 * sum(n .^ 2 .* p .^ 2), 2π^2 * sum(n .^ 2 .* m .^ 2)
    Θ = 0.5 * sum(p .* (up(m) .+ dn(m)))
    return (; P, M, p, m, n, εp, εm, Θ)
end

"E2/N and its gradient with respect to x"
function energy_grad(x, V0, N, nm)
    o = orbital_data(x, nm)
    r = solve_two_mode((o.εp + o.εm) / 2, (o.εp - o.εm) / 2, o.Θ, V0, N)
    npl, nmi = N / 2 + r.g, N / 2 - r.g
    gp = npl * 4π^2 .* o.n .^ 2 .* o.p .+ r.W * o.Θ .* (up(o.m) .+ dn(o.m))
    gm = nmi * 4π^2 .* o.n .^ 2 .* o.m .+ r.W * o.Θ .* (up(o.p) .+ dn(o.p))
    GP = (gp .- o.p * dot(o.p, gp)) / norm(o.P)
    GM = (gm .- o.m * dot(o.m, gm)) / norm(o.M)
    g = vcat([k == 0 ? GP[nm+1] : GP[nm+1+k] + GP[nm+1-k] for k in evens(nm)],
             [GM[nm+1+k] + GM[nm+1-k] for k in odds(nm)])
    return r.E2, g / N, r, o
end

function bfgs(f, x0; gtol = 1e-10, maxit = 5000)
    x = copy(x0)
    E, g = f(x)
    H = Matrix(1.0I, length(x), length(x))
    it, stall = 0, 0
    # stop on a small gradient, or when E has stopped decreasing (ΔE < 1e-15 |E|) for 10 iterations
    while norm(g) > gtol * norm(x) && stall < 10 && it < maxit
        it += 1
        d = -H * g
        if dot(d, g) >= 0
            H = Matrix(1.0I, length(x), length(x)); d = -g
        end
        α, Enew, gnew = 1.0, E, g
        while true
            Enew, gnew = f(x + α * d)
            Enew <= E + 1e-4 * α * dot(d, g) && break
            α /= 2
            α < 1e-14 && return x, E, it            # no further decrease at machine precision
        end
        s, y = α * d, gnew - g
        sy = dot(s, y)
        if sy > 1e-16
            ρ = 1 / sy
            H = (I - ρ * s * y') * H * (I - ρ * y * s') + ρ * s * s'
        end
        stall = E - Enew < 1e-15 * max(1, abs(E)) ? stall + 1 : 0
        x, E, g = x + s, Enew, gnew
    end
    return x, E, it
end

function optimise(x0, V0, N, nm)
    # line-search trial points can have ε₊ > ε₋ (t > 0), where for odd N the full ground state is
    # antisymmetric and solve_two_mode warns; such points are simply rejected (higher E2). The final
    # state is checked separately (t < 0, symmetric-sector E2 = full-spectrum minimum).
    x, E, it = with_logger(NullLogger()) do
        bfgs(z -> energy_grad(z, V0, N, nm)[1:2], x0)
    end
    _, g, r, o = energy_grad(x, V0, N, nm)
    return (; x, E2 = E, it, gnorm = norm(g), λp = r.λp, λm = r.λm, r.w, o.p, o.m, o.εp, o.εm, o.Θ)
end

# ---- initial guesses
const N_GP = (length(NSET) - 1) ÷ 2
x_uniform(nm) = (x = zeros(npar(nm)); x[1] = 1.0; x[length(evens(nm))+1] = 1.0; x)   # φ⁺ = 1, φ⁻ = √2 cos 2πx
function x_gp(V0, N, nm)
    c = solve_gp(V0, N).c
    n = collect(NSET)
    cm = c .* isodd.(n)
    norm(cm) < 1e-12 && return x_uniform(nm)                                           # uniform GP
    return params(c .* iseven.(n), cm, nm)
end

"GP-orbital two-mode quantities on the same footing (ε±, Θ', orbitals) for comparison"
function gp_orbital_data(V0, N, nm)
    o = orbital_data(x_gp(V0, N, nm), nm)
    tm = two_mode(V0, N)                                   # E2 with GP orbitals on n = −30 … 30
    return (; o..., E2 = tm.E2, egp = tm.egp, λp = tm.λp, λm = tm.λm, s = tm.s)
end

"optimum from two starts: GP orbitals (uniform/cos below threshold) and the other guess"
function opt_point(V0, N; nm = NM)
    x1 = x_gp(V0, N, nm)
    x2 = x1 == x_uniform(nm) ? x_gp(-25.0, N, nm) : x_uniform(nm)     # below threshold: GP orbitals at −25
    a, b = optimise(x1, V0, N, nm), optimise(x2, V0, N, nm)
    best = a.E2 <= b.E2 ? a : b
    ov = dot(a.p, b.p)^2 * dot(a.m, b.m)^2
    return best, a, b, ov
end

# ---------------------------------------------------------------- checks
println("Checks")
ok = Ref(true)
chk(name, cond, info = "") = (@printf("  [%s] %s %s\n", cond ? "PASS" : "FAIL", name, info); ok[] &= cond)

let x = x_gp(-30.0, 40, NM) .+ 0.01 .* sin.(1:npar(NM)), h = 1e-6
    E, g = energy_grad(x, -30.0, 40, NM)
    gfd = [(energy_grad(x .+ h .* (1:npar(NM) .== j), -30.0, 40, NM)[1] -
            energy_grad(x .- h .* (1:npar(NM) .== j), -30.0, 40, NM)[1]) / 2h for j in 1:npar(NM)]
    chk("Hellmann–Feynman gradient = finite differences", maximum(abs.(g .- gfd)) < 1e-7,
        @sprintf("(max |Δ| = %.1e, |g| = %.2e)", maximum(abs.(g .- gfd)), norm(g)))
end
let gpo = gp_orbital_data(-30.0, 40, NM), E = energy_grad(x_gp(-30.0, 40, NM), -30.0, 40, NM)[1]
    chk("E2(GP orbitals, |n| ≤ $NM) = two_mode.jl", abs(E - gpo.E2) < 1e-12, @sprintf("(Δ = %.1e)", E - gpo.E2))
end
# (perturbation small enough to keep ε₊ < ε₋: an empty orbital has zero gradient, so a start with ε₊ > ε₋
#  stays at the stationary point "all particles in φ⁻ = √2 cos 2πx", E2/N = 2π²)
r0 = optimise(x_uniform(NM) .+ 0.05 .* cos.(1:npar(NM)) ./ (1:npar(NM)) .^ 2, 0.0, 40, NM)
chk("V0 = 0 (perturbed start): E2_opt = 0, all particles in φ⁺ = 1", abs(r0.E2) < 1e-10 && abs(r0.λp - 1) < 1e-8 &&
    abs(abs(r0.p[NM+1]) - 1) < 1e-6, @sprintf("(E2/N = %.1e, λ+/N = %.10f, |d⁺_0| = %.8f)", r0.E2, r0.λp, abs(r0.p[NM+1])))

# ---------------------------------------------------------------- runs
QL = qmc_lambdas()
pts = vcat([(40, V) for V in RUNS_N40], vec(RUNS_NV), [(40, -150.0)])
res = Dict()
println("\nOptimisation (two starts each; NM = $NM vs $NM_CHECK)")
println("    V0   N   E2_opt/N (start 1)   E2_opt/N (start 2)   |Δ|      overlap²   E2(NM=$NM_CHECK) − E2   BFGS it")
for (N, V) in pts
    best, a, b, ov = opt_point(V, N)
    c32 = opt_point(V, N; nm = NM_CHECK)[1]
    gpo = gp_orbital_data(V, N, NM)
    res[(N, V)] = (; best, a, b, ov, c32, gpo)
    @printf("%6.1f %3d   %.12f   %.12f   %.1e   %.8f   %+.1e          %d/%d\n",
            V, N, a.E2, b.E2, abs(a.E2 - b.E2), ov, c32.E2 - best.E2, a.it, b.it)
end

chk("two starts reach the same minimum (|ΔE2/N| < 1e-8)",
    all(abs(x.a.E2 - x.b.E2) < 1e-8 for x in values(res)),
    @sprintf("(max %.1e)", maximum(abs(x.a.E2 - x.b.E2) for x in values(res))))
chk("NM = $NM converged (|E2(NM=$NM_CHECK) − E2| < 1e-10)", all(abs(x.c32.E2 - x.best.E2) < 1e-10 for x in values(res)),
    @sprintf("(max %.1e)", maximum(abs(x.c32.E2 - x.best.E2) for x in values(res))))
function full_min(x, V0, N)
    T, t, Θ = (x.εp + x.εm) / 2, (x.εp - x.εm) / 2, x.Θ
    d = [N * T + V0 / (2N) * Θ^2 * ((2k - N)^2 - N) for k in 0:N]
    eigmin(SymTridiagonal(d, [t * sqrt((k + 1) * (N - k)) for k in 0:N-1])) / N
end
chk("optimum has ε₊ < ε₋ and E2 = full-spectrum minimum",
    all(x.best.εp < x.best.εm && abs(full_min(x.best, V, N) - x.best.E2) < 1e-10 for ((N, V), x) in res))
chk("E2_opt ≤ E2_GPorb", all(x.best.E2 <= x.gpo.E2 + 1e-12 for x in values(res)))
viol = []
for ((N, V), x) in res
    d = dmc_energy(N, V)
    d === nothing && continue
    x.best.E2 < d[1] - 2 * d[2] && push!(viol, (N, V, x.best.E2, d[1], d[2]))
end
chk("E2_opt ≥ E_DMC − 2σ (variational)", isempty(viol), isempty(viol) ? "" : string(viol))
for V in (-60.0, -150.0)
    x = res[(40, V)]
    chk("V0 = $V: E2_opt ≈ E2_GPorb ≈ E_GP", abs(x.best.E2 - x.gpo.E2) < 0.02 && abs(x.best.E2 - x.gpo.egp) < 0.02,
        @sprintf("(E_GP − E2_GPorb = %.2e, E2_GPorb − E2_opt = %.2e per particle)", x.gpo.egp - x.gpo.E2, x.gpo.E2 - x.best.E2))
end
println(ok[] ? "All checks passed" : "SOME CHECKS FAILED")

# ---------------------------------------------------------------- tables
frac(egp, e2, ed) = (egp - e2) / (egp - ed)
println("\nEnergies per particle; recovered fraction f = (E_GP − E2)/(E_GP − E_DMC)")
println("    V0   N     E_GP/N    E2_GPorb/N   E2_opt/N    E_DMC/N              f(GP orb)   f(opt)    [DMC run]")
rows = []
for (N, V) in pts
    x = res[(N, V)]
    d = dmc_energy(N, V)
    ed, ee, w, τ = d === nothing ? (NaN, NaN, 0, NaN) : d
    fg, fo = frac(x.gpo.egp, x.gpo.E2, ed), frac(x.gpo.egp, x.best.E2, ed)
    q = [get(QL, (V, k), (NaN, NaN)) for k in 1:2]
    push!(rows, (V, N, x.gpo.egp, x.gpo.E2, x.best.E2, ed, ee, fg, fo, x.gpo.λp, x.gpo.λm, x.best.λp, x.best.λm,
                 (N == 40 ? vcat(collect.(q)...) : fill(NaN, 4))...,
                 dot(x.best.p, x.gpo.p)^2, dot(x.best.m, x.gpo.m)^2,
                 x.gpo.εp, x.gpo.εm, x.gpo.Θ, x.best.εp, x.best.εm, x.best.Θ, w, τ))
    run = d === nothing ? "no DMC" : @sprintf("[%dw τ=%g]%s", w, τ, τ < 32 - 1e-9 ? " older run" : "")
    @printf("%6.1f %3d  %10.5f  %10.5f  %10.5f  %s   %7.3f   %7.3f   %s\n", V, N, x.gpo.egp, x.gpo.E2, x.best.E2,
            d === nothing ? "      —            " : @sprintf("%9.5f±%.5f", ed, ee), fg, fo, run)
end
open("numpy_arrays_GP/two_mode_opt.csv", "w") do io
    println(io, "V0,N,E_GP_per_N,E2_GPorb_per_N,E2_opt_per_N,E_DMC_per_N,E_DMC_err,frac_GPorb,frac_opt," *
                "lambda_plus_GPorb,lambda_minus_GPorb,lambda_plus_opt,lambda_minus_opt," *
                "lambda0_DMC,lambda1_DMC,lambda0_lin,lambda1_lin,overlap2_plus,overlap2_minus," *
                "eps_plus_GPorb,eps_minus_GPorb,Theta_prime_GPorb,eps_plus_opt,eps_minus_opt,Theta_prime_opt," *
                "DMC_walkers,DMC_tau")
    for r in rows; println(io, join([@sprintf("%.12g", v) for v in r], ",")); end
end

println("\nOccupations λ±/N: two-mode with GP orbitals | optimised | QMC (N = 40, DMC mixed | 2 DMC − VMC)")
for r in rows
    ql = isnan(r[14]) ? "" : @sprintf("%.4f,%.4f | %.4f,%.4f", r[14:17]...)
    @printf("%6.1f %3d   %.4f,%.4f | %.4f,%.4f | %s\n", r[1], r[2], r[10], r[11], r[12], r[13], ql)
end
println("\nOrbital change: |⟨φ±opt|φ±GP⟩|², ε±, Θ' (GP orbitals → optimised)")
for r in rows
    @printf("%6.1f %3d   %.6f %.6f   ε+ %8.4f → %8.4f   ε− %8.4f → %8.4f   Θ' %.5f → %.5f\n",
            r[1], r[2], r[18], r[19], r[20], r[23], r[21], r[24], r[22], r[25])
end

# ---------------------------------------------------------------- orbitals and dense curve (N = 40)
open("numpy_arrays_GP/two_mode_opt_orbitals.csv", "w") do io
    println(io, "V0,N,orbital,n,d_opt,d_GPorb")
    for (N, V) in pts, (lab, f) in (("plus", :p), ("minus", :m))
        x = res[(N, V)]
        a, b = getfield(x.best, f), getfield(x.gpo, f)
        sgn = sign(dot(a, b)) == 0 ? 1.0 : sign(dot(a, b))
        for k in 0:NM
            (lab == "plus") == iseven(k) || continue
            @printf(io, "%.1f,%d,%s,%d,%.12g,%.12g\n", V, N, lab, k, sgn * a[NM+1+k], b[NM+1+k])
        end
    end
end
open("numpy_arrays_GP/two_mode_opt_dense_N40.csv", "w") do io
    println(io, "V0,E_GP_per_N,E2_GPorb_per_N,E2_opt_per_N,lambda_plus_opt,lambda_minus_opt")
    xprev = nothing
    for V in 0.0:-0.5:-60.0
        tm = two_mode(V, 40)
        x = optimise(x_gp(V, 40, NM), V, 40, NM)
        if xprev !== nothing                                   # continuation from the previous point
            y = optimise(xprev, V, 40, NM)
            y.E2 < x.E2 - 1e-10 && (x = y)
        end
        xprev = x.x
        @printf(io, "%.1f,%.12g,%.12g,%.12g,%.12g,%.12g\n", V, tm.egp, tm.E2, x.E2, x.λp, x.λm)
    end
end
println("\nSaved numpy_arrays_GP/two_mode_opt.csv, two_mode_opt_dense_N40.csv, two_mode_opt_orbitals.csv")
