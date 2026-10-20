# Analytic marginalisation of the linear weights.
#
# Data d ~ N(A w, diag((s_k σ_i)²)),  prior w_j ~ N(0, τ_j²)  (τ is a per-column scale).
# With rows divided by s_k σ_i the noise is unit, and with Λ = AᵀA + diag(1/τ²):
#     log p(d | θ, s, τ) = -½ [ n log 2π + log|Λ| + 2 Σ log τ_j + bᵀb − (Aᵀb)ᵀ Λ⁻¹ (Aᵀb) ]
#                          − Σ_i log(s_k σ_i)
# This costs one m×m Cholesky (m = number of columns) instead of an n×n solve.

"""
    ridge_precision(S, τ)

Posterior precision of the weights given the data: Λ = AᵀA + diag(1/τ²).
"""
function ridge_precision(S::CombinedStats, τ::AbstractVector)
    if length(τ) != length(S.Atb)
        throw(DimensionMismatch("τ has $(length(τ)) entries, there are $(length(S.Atb)) columns"))
    end

    prior_precision = Diagonal(1 ./ τ .^ 2)
    return S.AtA + prior_precision
end


"""
    marginal_loglike(S, τ)

Log marginal likelihood of the data with the linear weights integrated out against
independent zero-mean Gaussian priors of scales `τ` (one per column). Returns `-Inf`
if the precision matrix is not numerically positive definite.

The Gaussian prior ignores positivity (see `sample_weights` for the constrained
conditional) and acts as ridge regularisation: with very large `τ` the evidence is
dominated by the Occam factor, so choose `τ` deliberately or fit it hierarchically.
"""
function marginal_loglike(S::CombinedStats, τ::AbstractVector)
    if length(τ) != length(S.Atb)
        throw(DimensionMismatch("τ has $(length(τ)) entries, there are $(length(S.Atb)) columns"))
    end

    F = cholesky(Symmetric(ridge_precision(S, τ)); check=false)
    if !issuccess(F)
        return -Inf
    end

    quadratic = S.btb - dot(S.Atb, F \ S.Atb)
    log_determinant = logdet(F) + 2 * sum(log, τ)
    return -(S.n * log(2π) + log_determinant + quadratic) / 2 - S.logdetrows
end
