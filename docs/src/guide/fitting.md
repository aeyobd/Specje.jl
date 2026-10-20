# χ² fitting

```julia
fit = fit_kinematics(model, spectrum, θ0; iterations=2000, tol=1e-6)
fit = fit_kinematics(model, fitdata, θ0; ...)
```

[`fit_kinematics`](@ref) minimises

```
χ²(θ) = min_{w ≥ 0} ‖ (A(θ) w − flux) / noise ‖²     over the good pixels
```

by **variable projection**. Nelder–Mead searches the nonlinear parameters, in free
coordinates. At every step the weights are solved exactly by non-negative least squares
([`nnls`](@ref), the Lawson–Hanson active-set method). Only `nparams(model)` dimensions
are searched, however many templates there are.

- `θ0`: starting point in the model's parameter layout (km/s). Every σ must exceed its
  component's `σ_min`.
- `iterations`: maximum Nelder–Mead iterations.
- `tol`: convergence tolerance on the spread of χ² values across the simplex.

The optimiser is **local**. For multi-component models (two stellar populations,
several gas groups) or noisy data, try several `θ0` and keep the lowest χ².

## The result

A `FitResult` has the fields:

| Field | Meaning |
|---|---|
| `θ` | best-fit parameters |
| `weights` | NNLS weights at `θ` (all `ncolumns`) |
| `chi2` | χ² at the best fit |
| `ndata` | number of good data points |
| `nactive` | number of non-zero weights |
| `converged` | Nelder–Mead convergence flag |
| `iterations` | iterations used |

and helpers:

```julia
reduced_chi2(fit)               # chi2 / ndata
model_flux(model, fit)          # best-fit model on all pixels (including masked ones)
split_weights(model, fit.weights)   # per component; polynomial pairs combined
```

`reduced_chi2` divides by the number of data points, not degrees of freedom. With
thousands of pixels and a handful of parameters, the difference is negligible.

## Lower-level pieces

| Function | Purpose |
|---|---|
| [`FitData`](@ref)`(spectrum)` | noise-weighted data on the good pixels |
| [`design_matrix`](@ref)`(model, θ)` | `A(θ)` on all pixels |
| [`solve_weights`](@ref)`(A, fd)` | NNLS weights and χ² for a given `A` |
| [`profile_chi2`](@ref)`(model, fd, θ)` | χ² with the weights profiled out; `Inf` outside bounds |

These are useful for χ² maps, for checking degeneracies, or for writing your own
optimiser. For example, a χ² scan in σ★ at fixed v★:

```julia
fd = FitData(data)
scan = [profile_chi2(model, fd, [fit.θ[1], σ, fit.θ[3:end]...]) for σ in 100:5:300]
```

## Errors on the parameters

`fit_kinematics` returns no uncertainties. Options:

- the posterior ([Bayesian inference](bayes.md)), which also accounts for the weights
  and can fit the noise scale;
- a Monte Carlo bootstrap: refit after adding noise drawn from `noise` to the best-fit
  model.

Both assume the noise vector is right. If `reduced_chi2(fit)` is far from 1, rescale
the noise, or use `fit_noise=true` in the posterior.

## Joint fits with photometry

`fit_kinematics` also accepts a [`JointModel`](@ref): see [Photometry](photometry.md).
