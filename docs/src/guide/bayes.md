# Bayesian inference

## The posterior

```julia
bm = BayesModel(model, fitdata, θ0; priors=nothing, tau=nothing, tau_factor=5,
                fit_tau=false, fit_noise=false)
```

`model` is a `SpectralModel` or `JointModel`, and `fitdata` a [`FitData`](@ref). The
posterior is over the nonlinear parameters `θ`, with the linear weights integrated out
analytically (see [the marginal likelihood](../concepts.md#The-marginal-likelihood)):

```
log p(u | d) = log prior(θ) + log|∂θ/∂u| + log p(d | θ, s, τ)   (+ hyperparameter priors)
```

`θ0` is used only to set the default weight-prior scales.

### Options

- `fit_noise=true`: sample a noise-scale factor `s_k` per data block (spectrum, and
  photometry if present), so that `σ_eff = s_k · σ_obs`. Use it when the noise vector
  may be mis-scaled; it is cheap.
- `fit_tau=true`: sample the weight-prior scale `τ`, one per component, hierarchically.
  Otherwise `τ` is fixed.
- `tau`: fixed `τ` per component. The default ([`default_tau`](@ref)) is `tau_factor ×`
  the component's largest NNLS weight at `θ0`, with a floor of 10% of the overall
  largest weight.
- `priors`: a `Priors` object, a vector of priors on `θ` alone, or `nothing` for the
  defaults.

### Free coordinates

The sampler sees

```
u = [ u_model | log τ (one per component, if fit_tau) | log s (one per block, if fit_noise) ]
```

with `u_model` as in [Concepts](../concepts.md#Free-coordinates).

| Function | Purpose |
|---|---|
| [`initial_free`](@ref)`(bm, θ)` | `u` for physical `θ` (τ at its fixed value, s = 1) |
| [`to_physical`](@ref)`(bm, u)` | `(θ=…, τ=…, s=…)` |
| [`log_posterior`](@ref)`(bm, u)` | the log density; `-Inf` outside the bounds |
| [`log_posterior_and_gradient`](@ref)`(bm, u)` | value and ForwardDiff gradient |
| `LogDensityProblems.dimension(bm)` | length of `u` |

`BayesModel` implements the order-1 `LogDensityProblems` interface, so any sampler
built on it works directly.

## Priors

Priors are on the **physical** parameters, as `Distributions.jl` univariate
distributions. The Jacobian of the free-coordinate map is added automatically. The
defaults ([`default_priors`](@ref)) are:

| Parameter | Default prior |
|---|---|
| stellar `v` | `Normal(0, vmax/4)` truncated to `[−vmax, vmax]` |
| stellar `σ` | `LogNormal(log 150, 0.75)` truncated to `[σ_min, σ_max]` |
| gas `v` | `Normal(0, 150)` truncated to `[−vmax, vmax]` |
| gas `σ` | `LogNormal(log 100, 0.7)` truncated to `[σ_min, σ_max]` |
| joint-fit scale `s` | `LogNormal(0, 0.3)` |
| `log τ` | `Normal(log τ₀, 1)` |
| `log s` (noise) | `Normal(0, 0.3)` |

These are starting points, not recommendations; edit them for your sample. To replace
only the `θ` priors:

```julia
using Distributions
pθ = [truncated(Normal(0, 200), -1000, 1000),           # v★
      truncated(LogNormal(log(200), 0.5), 10, 600),     # σ★
      truncated(Normal(0, 100), -800, 800),             # v_gas
      truncated(LogNormal(log(80), 0.7), 5, 1000)]      # σ_gas
bm = BayesModel(model, fd, θ0; priors=pθ)
```

For full control, pass `Priors(θ_priors, logτ_priors, logs_priors)`.

Keep the priors' support inside the component bounds (`σ_min`, `σ_max`, `vmax`).
Outside the bounds the density is `-Inf`, and a sampler pressed against a hard wall
mixes badly (NUTS reports the steps as divergences).

## Posterior mode

```julia
m = map_estimate(bm, u0; iterations=3000, tol=1e-6)    # (u=…, logp=…, converged=…)
```

Nelder–Mead on `-log_posterior`. Good for starting a sampler. Start it from a χ² fit:
`u0 = initial_free(bm, fit.θ)`.

## Sampling with NUTS

Any `LogDensityProblems`-based NUTS implementation works. With AdvancedHMC:

```julia
using AdvancedHMC, AbstractMCMC, Random
samples = AbstractMCMC.sample(Xoshiro(1), AbstractMCMC.LogDensityModel(bm), NUTS(0.8), 2000;
                              n_adapts=1000, initial_params=m.u)
U = reduce(hcat, [t.z.θ for t in samples[1001:end]])                 # free coordinates
Θ = reduce(hcat, [to_physical(bm, u).θ for u in eachcol(U)])         # km/s
count(t.stat.numerical_error for t in samples[1001:end])             # divergences
```

With DynamicHMC:

```julia
using DynamicHMC
r = mcmc_with_warmup(Xoshiro(2), bm, 1000; initialization=(q=m.u,))
Θ = reduce(hcat, [to_physical(bm, u).θ for u in eachcol(r.posterior_matrix)])
```

(The exact result types depend on the sampler package version.)

NUTS needs the gradient, which costs about 2–3 density evaluations for a few
parameters. It usually needs far fewer evaluations per effective sample than the
ensemble sampler, especially as parameters are added (several gas groups, two stellar
components).

## Ensemble sampler

A gradient-free affine-invariant ensemble sampler (Goodman & Weare stretch move, as in
emcee) is built in:

```julia
chain = sample_ensemble(rng, bm, u0; nwalkers=2max(8, 2ndim), nsteps=1000,
                        a=2.0, ball_scale=0.05, thin=1, threaded=false)
```

- The walkers start in a Gaussian ball of width `ball_scale` (free-coordinate units:
  pixels in v, fractional changes in σ) around `u0`. Or pass an `(ndim, nwalkers)`
  matrix of starting points.
- `nwalkers` must be even and at least 4. Use at least `2·ndim`.
- `threaded=true` evaluates proposals on `Threads.nthreads()` threads (start Julia with
  `-t N`). Only turn it on after checking that the density is thread-safe for your
  model; the package doesn't test this.

Diagnostics and output:

```julia
acceptance_fraction(chain)          # ≈ 0.2–0.5 is healthy
autocorr_time(chain)                # integrated autocorrelation time per coordinate (steps)
X = flatten_chain(chain; burn=500)  # (ndim, nsamples)
chain.logp                          # (nwalkers, nrecorded)
```

Run for many autocorrelation times (≳ 50 τ), and discard a burn-in of a few τ.
[`autocorr_function`](@ref) and [`autocorr_time_1d`](@ref) work on any single series.

## The weights

Given `θ` (and `s`, `τ`), the weights have a Gaussian conditional posterior,
`N(μ, Λ⁻¹)` with `Λ = AᵀA + diag(1/τ²)`:

```julia
wp = weight_posterior(bm, u)                      # WeightPosterior: mean, prec, chol
W  = sample_weights(rng, wp, 200; positive=true, burn=50, init=nothing)   # (ncolumns, 200)
```

- `positive=true` (default): truncated to `w ≥ 0` and sampled by Gibbs sweeps.
  Successive draws are correlated, and weakly constrained directions mix slowly, so
  increase `burn` and thin if needed.
- `positive=false`: independent draws from the unconstrained Gaussian.

To propagate the kinematic uncertainty into population quantities, draw a few weight
vectors for each posterior sample of `u`:

```julia
ages = Float64[]
p = template_params(tlib)
for u in eachcol(X[:, 1:10:end])
    for w in eachcol(sample_weights(rng, weight_posterior(bm, u), 5))
        ws = first(split_weights(model, w))
        push!(ages, sum(ws .* p.log_age) / sum(ws))       # light-weighted log age
    end
end
```

## Lower-level pieces

[`block_stats_all`](@ref), [`combine`](@ref), [`marginal_loglike`](@ref) and
[`conditional_weights`](@ref) compute the sufficient statistics (`AᵀA`, `Aᵀb`, `bᵀb`
per data block), apply noise scales, and evaluate the marginal likelihood or the weight
conditional. Use them to build a custom posterior, e.g. with a different weight prior
covariance.
