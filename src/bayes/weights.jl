# Conditional posterior of the linear weights given the nonlinear parameters:
# N(μ, Λ⁻¹) with Λ = AᵀA + diag(1/τ²), μ = Λ⁻¹ Aᵀb. Optionally truncated to w ≥ 0.

"""
    WeightPosterior

Gaussian conditional posterior of the weights: `mean` μ, precision `prec` Λ and its
Cholesky factor `chol`. Build with `conditional_weights` or `weight_posterior`.
"""
struct WeightPosterior{T<:AbstractFloat}
    mean::Vector{T}
    prec::Matrix{T}
    chol::Cholesky{T,Matrix{T}}
end


"""
    conditional_weights(S, τ)

Gaussian conditional posterior of the weights (positivity ignored).
"""
function conditional_weights(S::CombinedStats, τ::AbstractVector)
    Λ = Matrix(ridge_precision(S, τ))
    F = cholesky(Symmetric(Λ))
    μ = F \ S.Atb
    return WeightPosterior(μ, Λ, F)
end


"""
    gaussian_draw(rng, wp)

One unconstrained draw: μ + U⁻¹ z with Λ = UᵀU and z standard normal.
"""
function gaussian_draw(rng, wp::WeightPosterior)
    z = randn(rng, length(wp.mean))
    return wp.mean .+ wp.chol.U \ z
end


"""
    draw_positive(rng, m, sd)

Truncated-normal draw on [0, ∞); falls back to 0 on numerical failure.
"""
function draw_positive(rng, m::Real, sd::Real)
    w = rand(rng, truncated(Normal(m, sd), 0.0, Inf))
    if !isfinite(w)
        return 0.0
    end
    return max(w, 0.0)
end


"""
    gibbs_sweep!(rng, w, wp)

One Gibbs sweep for the Gaussian truncated to the positive orthant. Coordinate j is
N(μ_j − Σ_{k≠j} Λ_jk (w_k − μ_k)/Λ_jj, 1/Λ_jj), truncated at zero.
"""
function gibbs_sweep!(rng, w::AbstractVector, wp::WeightPosterior)
    Λ = wp.prec
    μ = wp.mean
    δ = w .- μ
    for j in eachindex(w)
        λ = Λ[j, j]
        coupling = dot(view(Λ, :, j), δ) - λ * δ[j]
        conditional_mean = μ[j] - coupling / λ
        conditional_sd = 1 / sqrt(λ)
        w[j] = draw_positive(rng, conditional_mean, conditional_sd)
        δ[j] = w[j] - μ[j]
    end
    return w
end


"""
    sample_weights(rng, wp, nsamples; positive=true, burn=50, init=nothing)

Draw `(ncolumns, nsamples)` weight vectors from the conditional posterior. With
`positive=true` (default) the Gaussian is truncated to w ≥ 0 and sampled by Gibbs
sweeps started from `init` (default: the clipped mean), so successive columns are
correlated and weakly constrained directions mix slowly; raise `burn` and thin if needed.
`positive=false` gives independent unconstrained draws.
"""
function sample_weights(rng, wp::WeightPosterior, nsamples::Integer;
                        positive::Bool=true, burn::Integer=50, init=nothing)
    if nsamples < 1
        throw(ArgumentError("nsamples must be ≥ 1"))
    end
    if burn < 0
        throw(ArgumentError("burn must be ≥ 0"))
    end
    if !isnothing(init) && length(init) != length(wp.mean)
        throw(ArgumentError("init has $(length(init)) entries, there are $(length(wp.mean)) weights"))
    end

    out = Matrix{Float64}(undef, length(wp.mean), nsamples)
    if !positive
        for k in 1:nsamples
            out[:, k] = gaussian_draw(rng, wp)
        end
        return out
    end

    if isnothing(init)
        w = max.(wp.mean, 0.0)
    else
        w = max.(float.(init), 0.0)
    end
    for _ in 1:burn
        gibbs_sweep!(rng, w, wp)
    end
    for k in 1:nsamples
        out[:, k] = gibbs_sweep!(rng, w, wp)
    end
    return out
end
