# Affine-invariant ensemble sampler (Goodman & Weare 2010; the "stretch move" used by
# emcee), written against the LogDensityProblems interface. It is gradient-free; since
# BayesModel also provides gradients, NUTS samplers (AdvancedHMC, DynamicHMC) work too.

"""
    EnsembleChain

Output of `sample_ensemble`: `samples (ndim, nwalkers, nrecorded)`, log densities
`logp (nwalkers, nrecorded)`, and the numbers of accepted (`nacc`) and proposed
(`nprop`) moves.
"""
struct EnsembleChain{T<:AbstractFloat}
    samples::Array{T,3}
    logp::Matrix{T}
    nacc::Int
    nprop::Int
end


"""
    acceptance_fraction(chain)

Fraction of proposed moves that were accepted.
"""
function acceptance_fraction(c::EnsembleChain)
    return c.nacc / c.nprop
end


"""
    flatten_chain(chain; burn=0)

All recorded samples after `burn` steps as an `(ndim, nsamples)` matrix.
"""
function flatten_chain(c::EnsembleChain; burn::Integer=0)
    nrecorded = size(c.samples, 3)
    if !(0 <= burn < nrecorded)
        throw(ArgumentError("burn must lie in 0:$(nrecorded - 1)"))
    end

    kept = c.samples[:, :, burn+1:end]
    return reshape(kept, size(c.samples, 1), :)
end


# ---------------------------------------------------------------- one step
"""
    logp_columns(ℓ, Y; threaded=false)

Log densities of the columns of `Y` (optionally threaded; the density must be thread-safe).
"""
function logp_columns(ℓ, Y::AbstractMatrix; threaded::Bool=false)
    out = Vector{Float64}(undef, size(Y, 2))
    if threaded
        Threads.@threads for k in axes(Y, 2)
            out[k] = LogDensityProblems.logdensity(ℓ, Y[:, k])
        end
    else
        for k in axes(Y, 2)
            out[k] = LogDensityProblems.logdensity(ℓ, Y[:, k])
        end
    end
    return out
end


"""
    stretch_factor(rng, a)

Draw z from g(z) ∝ 1/√z on [1/a, a].
"""
function stretch_factor(rng, a::Real)
    return ((a - 1) * rand(rng) + 1)^2 / a
end


"""
    stretch_proposals(rng, X, idx, others, a)

Stretch-move proposals for the walkers `idx`, each towards a random walker from `others`.
Returns the proposals (columns) and their stretch factors.
"""
function stretch_proposals(rng, X::AbstractMatrix, idx, others, a::Real)
    Y = similar(X, size(X, 1), length(idx))
    z = Vector{Float64}(undef, length(idx))
    for (n, k) in enumerate(idx)
        j = rand(rng, others)
        z[n] = stretch_factor(rng, a)
        Y[:, n] = X[:, j] .+ z[n] .* (X[:, k] .- X[:, j])
    end
    return Y, z
end


"""
    accept_moves!(X, lp, idx, Y, lpy, z, rng)

Metropolis acceptance with the z^(d−1) factor; updates `X` and `lp` in place and
returns the number accepted.
"""
function accept_moves!(X, lp, idx, Y, lpy, z, rng)
    nd = size(X, 1)
    nacc = 0
    for (n, k) in enumerate(idx)
        log_ratio = (nd - 1) * log(z[n]) + lpy[n] - lp[k]
        if log(rand(rng)) < log_ratio
            X[:, k] = Y[:, n]
            lp[k] = lpy[n]
            nacc += 1
        end
    end
    return nacc
end


"""
    update_half!(X, lp, idx, others, ℓ, rng, a, threaded)

Propose and accept/reject moves for one half of the ensemble; returns the number accepted.
"""
function update_half!(X, lp, idx, others, ℓ, rng, a, threaded)
    Y, z = stretch_proposals(rng, X, idx, others, a)
    lpy = logp_columns(ℓ, Y; threaded=threaded)
    return accept_moves!(X, lp, idx, Y, lpy, z, rng)
end


# ---------------------------------------------------------------- driver
"""
    check_ensemble(ℓ, X0, a)

Throw an `ArgumentError` unless the walkers match the problem's dimension, there is an
even number of at least 4 of them, they do not all start at the same point, and the
stretch scale exceeds 1.
"""
function check_ensemble(ℓ, X0::AbstractMatrix, a::Real)
    nd, nw = size(X0)
    if nd != LogDensityProblems.dimension(ℓ)
        throw(ArgumentError("walkers have $nd coordinates, the problem has $(LogDensityProblems.dimension(ℓ))"))
    end
    if !iseven(nw) || nw < 4
        throw(ArgumentError("need an even number of walkers ≥ 4"))
    end
    if a <= 1
        throw(ArgumentError("stretch scale a must exceed 1"))
    end
    if all(k -> view(X0, :, k) == view(X0, :, 1), 2:nw)
        throw(ArgumentError("all walkers start at the same point: the stretch move can never leave it"))
    end
    return nothing
end


"""
    default_nwalkers(nd)

Default number of walkers for `nd` dimensions: `2 max(8, 2nd)`.
"""
function default_nwalkers(nd::Integer)
    return 2 * max(8, 2nd)
end


"""
    sample_ensemble(rng, ℓ, x0; nwalkers, nsteps=1000, a=2, ball_scale=0.05, thin=1, threaded=false)
    sample_ensemble(rng, ℓ, X0::AbstractMatrix; nsteps, a, thin, threaded)

Stretch-move ensemble sampling of the `LogDensityProblems` problem `ℓ`. With a vector
`x0` the walkers start in a Gaussian ball of width `ball_scale` (free-coordinate units,
≈ one pixel in v and a fractional change in σ); pass a matrix `(ndim, nwalkers)` to
choose them yourself. Use at least `2·ndim` walkers. `threaded=true` evaluates proposals on
`Threads.nthreads()` threads; only enable it after checking that your density is thread-safe.
"""
function sample_ensemble(rng, ℓ, x0::AbstractVector{<:Real};
                         nwalkers::Integer=default_nwalkers(length(x0)), ball_scale::Real=0.05, kw...)
    if ball_scale <= 0
        throw(ArgumentError("ball_scale must be positive"))
    end

    jitter = ball_scale .* randn(rng, length(x0), nwalkers)
    X0 = float.(x0) .+ jitter
    return sample_ensemble(rng, ℓ, X0; kw...)
end

function sample_ensemble(rng, ℓ, X0::AbstractMatrix{<:Real};
                         nsteps::Integer=1000, a::Real=2.0, thin::Integer=1, threaded::Bool=false)
    check_ensemble(ℓ, X0, a)
    if nsteps < 1
        throw(ArgumentError("nsteps must be ≥ 1"))
    end
    if !(1 <= thin <= nsteps)
        throw(ArgumentError("thin must lie in 1:nsteps"))
    end

    nd, nw = size(X0)
    X = Matrix{Float64}(X0)
    lp = logp_columns(ℓ, X; threaded=threaded)
    if !all(isfinite, lp)
        throw(ArgumentError("some initial walkers have zero density; shrink ball_scale or move x0"))
    end

    first_half = 1:nw÷2
    second_half = nw÷2+1:nw
    nrecorded = nsteps ÷ thin
    samples = Array{Float64,3}(undef, nd, nw, nrecorded)
    logps = Matrix{Float64}(undef, nw, nrecorded)
    nacc = 0

    for step in 1:nsteps
        nacc += update_half!(X, lp, first_half, second_half, ℓ, rng, a, threaded)
        nacc += update_half!(X, lp, second_half, first_half, ℓ, rng, a, threaded)
        if step % thin == 0
            samples[:, :, step ÷ thin] = X
            logps[:, step ÷ thin] = lp
        end
    end

    return EnsembleChain(samples, logps, nacc, nsteps * nw)
end


# ---------------------------------------------------------------- diagnostics
"""
    autocorr_function(x)

Normalised autocorrelation function of a series (FFT, zero-padded).
"""
function autocorr_function(x::AbstractVector{<:Real})
    if length(x) < 2
        throw(ArgumentError("need at least two samples"))
    end
    if all(==(first(x)), x)
        throw(ArgumentError("the series is constant (e.g. a walker that never moved): its autocorrelation is undefined"))
    end

    n = length(x)
    centred = x .- mean(x)
    padded = vcat(centred, zeros(n))
    spectrum = rfft(padded)
    power = complex.(abs2.(spectrum))
    acf = irfft(power, 2n)[1:n]
    return acf ./ acf[1]
end


"""
    auto_window(taus, c)

Sokal's automatic window: the first lag M with M ≥ c τ(M) (or the last lag).
"""
function auto_window(taus::AbstractVector, c::Real)
    k = findfirst(i -> (i - 1) >= c * taus[i], eachindex(taus))
    if isnothing(k)
        return length(taus)
    end
    return k
end


"""
    integrated_time(acf, c)

Integrated autocorrelation time from an autocorrelation function, with Sokal's window.
"""
function integrated_time(acf::AbstractVector, c::Real)
    taus = 2 .* cumsum(acf) .- 1
    return taus[auto_window(taus, c)]
end


"""
    autocorr_time_1d(x; c=5.0)

Integrated autocorrelation time of one series (in steps).
"""
function autocorr_time_1d(x::AbstractVector{<:Real}; c::Real=5.0)
    return integrated_time(autocorr_function(x), c)
end


"""
    autocorr_time(chain; c=5.0)

Integrated autocorrelation time per coordinate, from the walker-averaged ACF.
"""
function autocorr_time(chain::EnsembleChain; c::Real=5.0)
    nd, nw, _ = size(chain.samples)

    times = Float64[]
    for d in 1:nd
        mean_acf = autocorr_function(view(chain.samples, d, 1, :))
        for k in 2:nw
            mean_acf = mean_acf .+ autocorr_function(view(chain.samples, d, k, :))
        end
        mean_acf = mean_acf ./ nw
        push!(times, integrated_time(mean_acf, c))
    end
    return times
end
