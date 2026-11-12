# ------------------------------------------
# Allocation-free evaluation of a Dierckx Spline2D
# ------------------------------------------
#
# Dierckx's scalar `derivative(spl, x, y)` allocates its FITPACK work arrays on every
# call (~4 KB), which dominates the DMC cost and serializes the threads on the GC.
# FastSpline2D evaluates the *same* tensor-product B-spline (same knots and
# coefficients) with preallocated scratch buffers. It agrees with Dierckx to rounding
# error. The buffers make one instance non-thread-safe: use one copy per thread.

struct FastSpline2D
    tx::Vector{Float64}
    ty::Vector{Float64}
    c::Matrix{Float64}        # c[i, j]: i-th B-spline in x, j-th in y
    kx::Int
    ky::Int
    Nx::Matrix{Float64}       # Nx[r+1, q+1] = degree-q x-basis, scratch
    Ny::Matrix{Float64}
    left::Vector{Float64}
    right::Vector{Float64}
    d::Vector{Float64}        # coefficients reduced along y, scratch
end

function FastSpline2D(spl::Spline2D)
    nbx = length(spl.tx) - spl.kx - 1
    nby = length(spl.ty) - spl.ky - 1
    # FITPACK layout: c[(i-1)*nby + j]
    c = Matrix(transpose(reshape(copy(spl.c), nby, nbx)))
    kmax = max(spl.kx, spl.ky)
    return FastSpline2D(copy(spl.tx), copy(spl.ty), c, spl.kx, spl.ky,
                        zeros(spl.kx + 1, spl.kx + 1), zeros(spl.ky + 1, spl.ky + 1),
                        zeros(kmax), zeros(kmax), zeros(spl.kx + 1))
end

# Knot span l with t[l] ≤ x < t[l+1], clamped to the valid range [k+1, n-k-1].
@inline knot_span(t::Vector{Float64}, k::Int, x::Float64) =
    clamp(searchsortedlast(t, x), k + 1, length(t) - k - 1)

"""
Fills N[r+1, q+1] = B_{l-q+r, q}(x) for all degrees q = 0..k (Cox-de Boor triangle).
"""
@inline function bspline_basis!(N::Matrix{Float64}, t::Vector{Float64}, k::Int, l::Int, x::Float64,
                                left::Vector{Float64}, right::Vector{Float64})
    @inbounds begin
        N[1, 1] = 1.0
        for q in 1:k
            left[q]  = x - t[l + 1 - q]
            right[q] = t[l + q] - x
            saved = 0.0
            for r in 0:(q - 1)
                temp = N[r + 1, q] / (right[r + 1] + left[q - r])
                N[r + 1, q + 1] = saved + right[r + 1] * temp
                saved = left[q - r] * temp
            end
            N[q + 1, q + 1] = saved
        end
    end
    return nothing
end

"""
    spline_eval(fs, x, y, nux) -> ∂ⁿᵘˣ_x s(x, y)   (nux = 0, 1, 2; no y-derivatives)
"""
function spline_eval(fs::FastSpline2D, x::Float64, y::Float64, nux::Int)::Float64
    kx, ky = fs.kx, fs.ky
    tx, ty = fs.tx, fs.ty
    lx = knot_span(tx, kx, x)
    ly = knot_span(ty, ky, y)
    bspline_basis!(fs.Nx, tx, kx, lx, x, fs.left, fs.right)
    bspline_basis!(fs.Ny, ty, ky, ly, y, fs.left, fs.right)

    @inbounds begin
        # d[r+1] = Σ_j c[lx-kx+r, j] B_j(y)
        for r in 0:kx
            i = lx - kx + r
            acc = 0.0
            for s in 0:ky
                acc += fs.c[i, ly - ky + s] * fs.Ny[s + 1, ky + 1]
            end
            fs.d[r + 1] = acc
        end

        if nux == 0
            val = 0.0
            for r in 0:kx
                val += fs.d[r + 1] * fs.Nx[r + 1, kx + 1]
            end
            return val
        elseif nux == 1
            # s' = Σ_i kx (d_i - d_{i-1}) / (t_{i+kx} - t_i) B_{i,kx-1},  i = lx-kx+1+r
            val = 0.0
            for r in 0:(kx - 1)
                i = lx - kx + 1 + r
                val += kx * (fs.d[r + 2] - fs.d[r + 1]) / (tx[i + kx] - tx[i]) * fs.Nx[r + 1, kx]
            end
            return val
        elseif nux == 2
            # second difference of the first-derivative coefficients, basis of degree kx-2
            val = 0.0
            for r in 0:(kx - 2)
                i = lx - kx + 2 + r
                d1_i   = kx * (fs.d[r + 3] - fs.d[r + 2]) / (tx[i + kx] - tx[i])
                d1_im1 = kx * (fs.d[r + 2] - fs.d[r + 1]) / (tx[i - 1 + kx] - tx[i - 1])
                val += (kx - 1) * (d1_i - d1_im1) / (tx[i + kx - 1] - tx[i]) * fs.Nx[r + 1, kx - 1]
            end
            return val
        else
            error("FastSpline2D: only nux = 0, 1, 2 are implemented")
        end
    end
end

# Same call signatures as Dierckx, so FastSpline2D drops in wherever a Spline2D is used.
Dierckx.evaluate(fs::FastSpline2D, x::Real, y::Real) = spline_eval(fs, Float64(x), Float64(y), 0)

function Dierckx.derivative(fs::FastSpline2D, x::Real, y::Real; nux::Int = 0, nuy::Int = 0)
    nuy == 0 || error("FastSpline2D: y-derivatives are not implemented")
    return spline_eval(fs, Float64(x), Float64(y), nux)
end
