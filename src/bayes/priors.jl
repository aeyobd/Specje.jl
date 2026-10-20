# Priors live on *physical* parameters θ; the sampler works in free coordinates u and
# the log-Jacobian of u -> θ is added in the posterior. Hyperparameter priors
# (log τ, log s) are defined directly on the sampled log scale.

"""
    Priors(θ, logτ, logs)

- `θ`    one univariate prior per model parameter (physical units)
- `logτ` one prior per component on log τ (used only when τ is fitted)
- `logs` one prior per data block on log s (used only when noise scales are fitted)
"""
struct Priors
    θ::Vector{UnivariateDistribution}
    logτ::Vector{UnivariateDistribution}
    logs::Vector{UnivariateDistribution}
end


# ---------------------------------------------------------------- default priors
# These are starting points, not recommendations: edit them for your sample.

"""
    default_priors(model_or_component)

Default priors on the physical parameters, one `UnivariateDistribution` per entry of θ:
truncated normal velocities and truncated log-normal dispersions within the component
bounds (see the docs for the values), and `LogNormal(0, 0.3)` on a joint-fit scale.
"""
function default_priors(::Component)
    return UnivariateDistribution[]
end

function default_priors(c::StellarComponent)
    velocity = truncated(Normal(0.0, c.vmax / 4), -c.vmax, c.vmax)
    dispersion = truncated(LogNormal(log(150.0), 0.75), c.σ_min, c.σ_max)
    return UnivariateDistribution[velocity, dispersion]
end

function default_priors(c::GasComponent)
    priors = UnivariateDistribution[]
    for _ in 1:ngroups(c)
        velocity = truncated(Normal(0.0, 150.0), -c.vmax, c.vmax)
        dispersion = truncated(LogNormal(log(100.0), 0.7), c.σ_min, c.σ_max)
        push!(priors, velocity, dispersion)
    end
    return priors
end

function default_priors(m::SpectralModel)
    priors = UnivariateDistribution[]
    for c in m.components
        append!(priors, default_priors(c))
    end
    return priors
end

function default_priors(jm::JointModel)
    priors = default_priors(jm.model)
    if jm.fit_scale
        scale = LogNormal(0.0, 0.3)
        priors = vcat(priors, UnivariateDistribution[scale])
    end
    return priors
end


"""
    default_tau(model, fd, θ0; factor=5)

Per-component prior scale for the weights: `factor ×` the largest NNLS weight of the
component at `θ0` (with a floor of 10% of the overall maximum, so components that NNLS
switches off still get a usable scale).
"""
function default_tau(model, fd::FitData, θ0::AbstractVector; factor::Real=5.0)
    if factor <= 0
        throw(ArgumentError("factor must be positive"))
    end

    w, _ = solve_weights(design_matrix(model, θ0), fd)
    wmax = maximum(w; init=0.0)
    floor = wmax > 0 ? 0.1 * wmax : 1.0

    τ = Float64[]
    for r in column_ranges(model)
        component_max = maximum(abs, w[r]; init=0.0)
        push!(τ, factor * max(component_max, floor))
    end
    return τ
end


"""
    build_priors(priors, model, τ, nblocks)

Complete `Priors` from what the user passed: a `Priors` object (used as is), `nothing`
(the defaults), or a vector of priors on θ alone (hyperparameter priors are then
`Normal(log τ, 1)` on log τ and `Normal(0, 0.3)` on log s).
"""
function build_priors(p::Priors, model, τ::AbstractVector, nblocks::Integer)
    return p
end

function build_priors(::Nothing, model, τ::AbstractVector, nblocks::Integer)
    return build_priors(default_priors(model), model, τ, nblocks)
end

function build_priors(θ_priors::AbstractVector, model, τ::AbstractVector, nblocks::Integer)
    logτ_priors = UnivariateDistribution[Normal(log(t), 1.0) for t in τ]
    logs_priors = UnivariateDistribution[Normal(0.0, 0.3) for _ in 1:nblocks]
    return Priors(Vector{UnivariateDistribution}(θ_priors), logτ_priors, logs_priors)
end


# ---------------------------------------------------------------- evaluation
"""
    sum_logpdf(dists, x)

Σ log pdf of each `x[i]` under `dists[i]` (0 for empty inputs).
"""
function sum_logpdf(dists::AbstractVector, x::AbstractVector)
    if length(dists) != length(x)
        throw(ArgumentError("need one distribution per value: got $(length(dists)) for $(length(x))"))
    end

    total = 0.0
    for (d, xi) in zip(dists, x)
        total += logpdf(d, xi)
    end
    return total
end


"""
    log_prior(pr, θ, logτ, logs)

Log prior; `logτ` and `logs` are empty when those hyperparameters are not sampled.
"""
function log_prior(pr::Priors, θ::AbstractVector, logτ::AbstractVector, logs::AbstractVector)
    θ_term = sum_logpdf(pr.θ, θ)
    logτ_term = isempty(logτ) ? 0.0 : sum_logpdf(pr.logτ, logτ)
    logs_term = isempty(logs) ? 0.0 : sum_logpdf(pr.logs, logs)
    return θ_term + logτ_term + logs_term
end
