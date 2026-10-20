# Lawson–Hanson active-set non-negative least squares: min ‖A x − b‖₂ s.t. x ≥ 0.
# Implemented here (rather than as a dependency) so the fit loop has no surprises.

"""
    best_candidate(w, passive, tol)

Index of the non-passive variable with the largest gradient above `tol`, or `nothing`.
"""
function best_candidate(w, passive, tol)
    best = tol
    jbest = nothing
    for j in eachindex(w)
        if !passive[j] && w[j] > best
            best = w[j]
            jbest = j
        end
    end
    return jbest
end


"""
    ls_passive(A, b, passive)

Unconstrained least squares restricted to the passive set (zeros elsewhere).
"""
function ls_passive(A, b, passive)
    T = float(promote_type(eltype(A), eltype(b)))
    s = zeros(T, length(passive))
    if any(passive)
        s[passive] = A[:, passive] \ b
    end
    return s
end


"""
    step_to_boundary(x, s, passive)

Largest step α ∈ [0,1] from `x` towards `s` that keeps every passive variable ≥ 0.
"""
function step_to_boundary(x, s, passive)
    α = one(eltype(x))
    for j in eachindex(x)
        if !passive[j] || s[j] > 0
            continue
        end

        d = x[j] - s[j]
        if d > 0
            α = min(α, x[j] / d)
        else
            α = min(α, zero(α))
        end
    end
    return α
end


"""
    drop_zeros!(passive, x)

Move variables that reached zero out of the passive set (and set them exactly to zero).
"""
function drop_zeros!(passive, x)
    tolx = 1e-12 * maximum(x; init=zero(eltype(x)))
    for j in eachindex(x)
        if passive[j] && x[j] <= tolx
            passive[j] = false
            x[j] = 0
        end
    end
    return passive
end


"""
    refine!(passive, x, A, b)

Inner loop: solve on the passive set, backing off until the solution is feasible.
"""
function refine!(passive, x, A, b)
    s = ls_passive(A, b, passive)
    while any(<=(0), view(s, passive))
        α = step_to_boundary(x, s, passive)
        x = x .+ α .* (s .- x)
        drop_zeros!(passive, x)
        s = ls_passive(A, b, passive)
    end
    return s
end


"""
    nnls(A, b; maxiter=3size(A, 2)) -> x

Non-negative least squares by the Lawson–Hanson active-set method.
"""
function nnls(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}; maxiter::Integer=3 * size(A, 2))
    m, n = size(A)
    if length(b) != m
        throw(DimensionMismatch("A has $m rows but b has $(length(b)) entries"))
    end
    if maxiter < 1
        throw(ArgumentError("maxiter must be ≥ 1"))
    end

    T = float(promote_type(eltype(A), eltype(b)))
    x = zeros(T, n)
    passive = falses(n)
    w = A' * b
    tol = 10 * eps(T) * max(m, n) * maximum(abs, w; init=zero(T))

    for _ in 1:maxiter
        j = best_candidate(w, passive, tol)
        if isnothing(j)
            return x
        end

        passive[j] = true
        x = refine!(passive, x, A, b)
        residual = b - A * x
        w = A' * residual
    end
    error("nnls did not converge in $maxiter iterations")
end
