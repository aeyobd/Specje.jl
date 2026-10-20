# The posterior over the nonlinear parameters (and optional hyperparameters), with the
# linear weights marginalised out. Sampler coordinates u (free coordinates) are laid out as
#     [ u_model | log τ (per component, if fit_tau) | log s (per data block, if fit_noise) ]

"""
    BayesModel(model, data, θ0; priors=nothing, tau=nothing, tau_factor=5,
               fit_tau=false, fit_noise=false)

`model` is a `SpectralModel` or `JointModel`, `data` a `FitData`. `θ0` (physical
parameters) is only used to set the default weight-prior scales (`default_tau`).

- `fit_tau`   sample a prior scale τ per component (hierarchical regularisation)
- `fit_noise` sample a noise-scale factor s per data block (σ² = s² σ_obs²)

Implements the order-1 `LogDensityProblems` interface: gradients come from ForwardDiff
(`log_posterior_and_gradient`), so gradient-based samplers such as NUTS (AdvancedHMC,
DynamicHMC) can use a `BayesModel` directly.
"""
struct BayesModel{M,P<:Priors}
    model::M
    data::FitData{Float64}
    priors::P
    tau::Vector{Float64}
    fit_tau::Bool
    fit_noise::Bool

    function BayesModel(model::M, data::FitData{Float64}, priors::P, tau::AbstractVector{<:Real};
                        fit_tau::Bool=false, fit_noise::Bool=false) where {M,P<:Priors}
        ncomp = length(model_components(model))
        if length(priors.θ) != nparams(model)
            throw(ArgumentError("need $(nparams(model)) parameter priors, got $(length(priors.θ))"))
        end
        if length(tau) != ncomp
            throw(ArgumentError("need one τ per component ($ncomp)"))
        end
        if !all(>(0), tau)
            throw(ArgumentError("τ must be positive"))
        end
        if length(priors.logτ) != ncomp
            throw(ArgumentError("need one log τ prior per component"))
        end
        if length(priors.logs) != length(data.blocks)
            throw(ArgumentError("need one log s prior per data block ($(length(data.blocks)))"))
        end
        return new{M,P}(model, data, priors, Vector{Float64}(tau), fit_tau, fit_noise)
    end
end


function BayesModel(model, data::FitData, θ0::AbstractVector; priors=nothing,
                    tau_factor::Real=5.0, tau=nothing, fit_tau::Bool=false, fit_noise::Bool=false)
    check_theta(model, θ0)

    if isnothing(tau)
        τ = default_tau(model, data, θ0; factor=tau_factor)
    else
        τ = tau
    end
    if !all(>(0), τ)
        throw(ArgumentError("τ must be positive"))
    end

    full_priors = build_priors(priors, model, τ, length(data.blocks))
    return BayesModel(model, data, full_priors, τ; fit_tau=fit_tau, fit_noise=fit_noise)
end


# ---------------------------------------------------------------- layout
"""
    n_tau(bm)

Number of sampled weight-prior scales (one per component if `fit_tau`, else 0).
"""
function n_tau(bm::BayesModel)
    if bm.fit_tau
        return length(bm.tau)
    else
        return 0
    end
end


"""
    n_noise(bm)

Number of sampled noise scales (one per data block if `fit_noise`, else 0).
"""
function n_noise(bm::BayesModel)
    if bm.fit_noise
        return length(bm.data.blocks)
    else
        return 0
    end
end


"""
    ndim(bm)

Dimension of the free coordinates: model parameters plus sampled hyperparameters.
"""
function ndim(bm::BayesModel)
    return nparams(bm.model) + n_tau(bm) + n_noise(bm)
end


"""
    split_free(bm, u) -> (u_model, log τ, log s)

Split free coordinates into views of the model, log τ and log s parts.
"""
function split_free(bm::BayesModel, u::AbstractVector)
    if length(u) != ndim(bm)
        throw(ArgumentError("expected $(ndim(bm)) coordinates, got $(length(u))"))
    end

    np = nparams(bm.model)
    nt = n_tau(bm)
    u_model = view(u, 1:np)
    log_tau = view(u, np+1:np+nt)
    log_noise = view(u, np+nt+1:length(u))
    return u_model, log_tau, log_noise
end


"""
    to_physical(bm, u)

Physical parameters `θ`, weight-prior scales `τ` and noise scales `s` at free
coordinates `u`, as a named tuple.
"""
function to_physical(bm::BayesModel, u::AbstractVector)
    um, lτ, ls = split_free(bm, u)

    θ = from_free(bm.model, um)
    τ = bm.fit_tau ? exp.(lτ) : bm.tau
    s = bm.fit_noise ? exp.(ls) : ones(length(bm.data.blocks))
    return (θ=θ, τ=τ, s=s)
end


"""
    initial_free(bm, θ0)

Free coordinates for physical parameters `θ0` (τ at its fixed value, s = 1).
"""
function initial_free(bm::BayesModel, θ0::AbstractVector)
    u_model = to_free(bm.model, θ0)
    log_tau = bm.fit_tau ? log.(bm.tau) : Float64[]
    log_noise = bm.fit_noise ? zeros(length(bm.data.blocks)) : Float64[]
    return vcat(u_model, log_tau, log_noise)
end


# ---------------------------------------------------------------- density
"""
    column_scales(bm, τ)

The weight-prior scale of every design-matrix column: each component's τ repeated
over its columns.
"""
function column_scales(bm::BayesModel, τ::AbstractVector)
    components = model_components(bm.model)
    if length(τ) != length(components)
        throw(ArgumentError("need one τ per component ($(length(components)))"))
    end

    per_component = [fill(τc, ncolumns(c)) for (τc, c) in zip(τ, components)]
    return reduce(vcat, per_component)
end


"""
    weighted_stats(bm, θ, s)

Combined sufficient statistics of the data at parameters `θ` and noise scales `s`.
"""
function weighted_stats(bm::BayesModel, θ::AbstractVector, s::AbstractVector)
    A = design_matrix(bm.model, θ)
    blocks = block_stats_all(A, bm.data)
    return combine(blocks, s)
end


"""
    loglike(bm, θ, τ, s)

Marginal log likelihood at physical parameters (NaN-safe: invalid values give -Inf).
"""
function loglike(bm::BayesModel, θ::AbstractVector, τ::AbstractVector, s::AbstractVector)
    S = weighted_stats(bm, θ, s)
    ll = marginal_loglike(S, column_scales(bm, τ))
    if isnan(ll)
        return -Inf
    end
    return ll
end


"""
    log_posterior(bm, u)

Log posterior at free coordinates `u`: prior on the physical parameters, log-Jacobian
of the coordinate map, and the weight-marginalised likelihood. `-Inf` outside the
component bounds (hard walls: keep priors well inside them).
"""
function log_posterior(bm::BayesModel, u::AbstractVector)
    um, lτ, ls = split_free(bm, u)
    θ = from_free(bm.model, um)
    if !in_bounds(bm.model, θ)
        return -Inf
    end

    log_prior_term = log_prior(bm.priors, θ, lτ, ls) + log_jacobian(bm.model, um)
    if !isfinite(log_prior_term)
        return -Inf
    end

    physical = to_physical(bm, u)
    return log_prior_term + loglike(bm, θ, physical.τ, physical.s)
end


"""
    log_posterior_and_gradient(bm, u)

Log posterior and its gradient in free coordinates, by forward-mode AD (ForwardDiff).
The gradient is zero where the density is `-Inf` (outside the component bounds).
"""
function log_posterior_and_gradient(bm::BayesModel, u::AbstractVector{<:Real})
    x = collect(float.(u))
    result = DiffResults.GradientResult(x)
    result = ForwardDiff.gradient!(result, v -> log_posterior(bm, v), x)

    lp = DiffResults.value(result)
    if !isfinite(lp)
        return (-Inf, zero(x))
    end
    return (lp, DiffResults.gradient(result))
end


function LogDensityProblems.logdensity(bm::BayesModel, u)
    return log_posterior(bm, u)
end

function LogDensityProblems.logdensity_and_gradient(bm::BayesModel, u)
    return log_posterior_and_gradient(bm, u)
end

function LogDensityProblems.dimension(bm::BayesModel)
    return ndim(bm)
end

function LogDensityProblems.capabilities(::Type{<:BayesModel})
    return LogDensityProblems.LogDensityOrder{1}()
end


# ---------------------------------------------------------------- MAP and weights
"""
    map_estimate(bm, u0; iterations=3000, tol=1e-6)

Posterior mode by Nelder–Mead from free coordinates `u0`; a good starting point for
a sampler. Returns `(u=…, logp=…, converged=…)`.
"""
function map_estimate(bm::BayesModel, u0::AbstractVector{<:Real}; iterations::Integer=3000,
                      tol::Real=1e-6)
    if length(u0) != ndim(bm)
        throw(ArgumentError("expected $(ndim(bm)) coordinates, got $(length(u0))"))
    end
    if !isfinite(log_posterior(bm, u0))
        throw(ArgumentError("the starting point has zero posterior density"))
    end

    options = Optim.Options(iterations=iterations, g_abstol=tol)
    negative_log_posterior(u) = -log_posterior(bm, u)
    result = Optim.optimize(negative_log_posterior, collect(float.(u0)), nelder_mead(), options)
    return (u=Optim.minimizer(result), logp=-Optim.minimum(result),
            converged=Optim.converged(result))
end


"""
    weight_posterior(bm, u)

Conditional posterior of the linear weights at free coordinates `u` (see `sample_weights`).
"""
function weight_posterior(bm::BayesModel, u::AbstractVector)
    physical = to_physical(bm, u)
    S = weighted_stats(bm, physical.θ, physical.s)
    return conditional_weights(S, column_scales(bm, physical.τ))
end
