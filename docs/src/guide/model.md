# Building a model

A [`SpectralModel`](@ref) is a [`KinematicsGrid`](@ref), a tuple of components, and an
optional projection onto the data pixels:

```julia
model = SpectralModel(grid, star, gas, poly; projection=P)
```

Components are evaluated on the **model grid**: the log-uniform wavelengths passed to
[`kinematics_grid`](@ref) as the data grid. With `projection=P` (from
[`pixel_projection`](@ref)), the design matrix is `P · A_model` on the native data
pixels. Without it, the model grid *is* the data grid, i.e. the data were log-rebinned.
`npix(model)` is the number of data pixels, and `Specje.nmodel(model)` the number of
model-grid pixels.

- `θ` is the concatenation of the components' nonlinear parameters, in order
  ([`param_ranges`](@ref) gives each component's slice).
- The design matrix `A(θ)` ([`design_matrix`](@ref)) concatenates their columns
  ([`column_ranges`](@ref)), then applies the projection if there is one.
- `nparams(model)`, `ncolumns(model)`, `npix(model)` give the sizes.

## The kinematics grid

```julia
grid = kinematics_grid(template_wave, model_wave; margin_kms=2500.0)
```

This links the (wider) template grid to the model grid (the data grid, if you rebinned
the data). It checks that both are
log-uniform with the same velscale, and that the templates extend at least
`margin_kms` beyond the data on both sides. It then picks a padded FFT length and
records the crop and the sub-pixel offset between the two grids. Choose `margin_kms`
larger than `vmax + 5σ_max` of the stellar components (see
[Concepts](../concepts.md#The-stellar-convolution)).

## Stellar component

```julia
StellarComponent(grid, T; σ_min=0, σ_max=1000, vmax=2000, σ_templ=0, phot=nothing)
```

- `T`: the template matrix `(ntemplate_pixels, ntemplates)` on the grid's template
  wavelengths. Usually it comes from [`match_resolution`](@ref).
- `θ = (v, σ)` in km/s: one Gaussian LOSVD shared by all of the component's templates.
- `σ_min`, `σ_max`, `vmax`: bounds. `σ_min` is enforced by the free-coordinate map;
  `σ_max` and `vmax` are hard walls. Set `σ_min` below any plausible value, but above
  zero if the data can't resolve very small dispersions. Starting values must satisfy
  `σ > σ_min`.
- `σ_templ`: a constant velocity resolution (km/s) the templates already have. The
  kernel then applies only `√(σ² − σ_templ²)`. Needed when a resolution matrix in the
  projection applies the instrument; see
  [Resolution matrices in the forward model](resolution.md#Resolution-matrices-in-the-forward-model-(DESI)).
  It requires `σ_min ≥ σ_templ`.
- `phot`: [`BandTemplates`](@ref), needed only for joint fits with photometry.

The templates are Fourier-transformed once, at construction. Each evaluation then costs
one inverse FFT per template.

**Several stellar populations with different kinematics** (e.g. bulge and disc): use
two `StellarComponent`s with different template sets on the same grid. Each gets its own
`(v, σ)`.

## Gas component

```julia
GasComponent(model_wave, groups, lsf; σ_min=1, σ_max=1000, vmax=1000)
GasComponent(model_wave, groups; …)    # no instrumental broadening (resolution matrix in the projection)
```

Emission lines are pixel-integrated Gaussians in `ln λ` (velocity), evaluated on the
model grid (rest frame); `lsf` must be given on the same grid. They aren't convolved with the stellar LOSVD. The
building blocks are:

- [`Multiplet`](@ref)`(name, wave)` or `Multiplet(name, waves, ratios)`: lines with
  fixed flux ratios, sharing one amplitude (one design-matrix column). The amplitude is
  the integrated flux of the line whose ratio is 1.
- [`GasGroup`](@ref)`(name, multiplets...)`: multiplets sharing one `(v, σ)` (two
  nonlinear parameters).

```julia
groups = [
    GasGroup("balmer", Multiplet("Hbeta", 4862.683), Multiplet("Hgamma", 4341.692)),
    GasGroup("forbidden", Multiplet("[OIII]", [4960.295, 5008.240], [1.0, 2.98])),
]
gas = GasComponent(mwave, groups, lsf; σ_min=5.0, vmax=800.0)
```

- `θ = (v₁, σ₁, v₂, σ₂, …)`, one pair per group, km/s; `σ` is intrinsic (the LSF is
  applied internally, see [Instrumental resolution](resolution.md)).
- Lines outside the data range are dropped, and so are multiplets with no lines
  left. A group with no lines left is an error.
- [`default_gas_groups`](@ref)`()` gives Balmer, low-ionisation and high-ionisation
  groups with vacuum wavelengths. The line list and the [OIII] (2.98) and [NII] (3.05)
  ratios are starting points; check them against your preferred reference.
- [`line_names`](@ref) and [`group_names`](@ref) list what was kept, in column and
  parameter order.

A line partly off the grid edge is truncated, not an error: only the flux on the grid
enters the column.

## Polynomial component

```julia
PolynomialComponent(length(model_wave), degree)
```

An additive Legendre polynomial `P₀ … P_degree` over the model-grid pixels, for continuum
mismatch (flux calibration, sky residuals, template mismatch). It has no nonlinear
parameters. Because all weights are non-negative, it adds `2(degree + 1)` columns
`[P  −P]`; [`split_weights`](@ref) returns the signed coefficients `w₊ − w₋`.

Higher degrees absorb more continuum mismatch, but also more real population
information (the continuum shape carries age and dust information). Typical degrees are
4–10 for spectra a few thousand Å long.

## Writing a custom component

Any subtype of `Component` can join a `SpectralModel`. Required methods:

```julia
Specje.ncolumns(c::MyComponent)            # number of design-matrix columns
Specje.columns(c::MyComponent, grid, θ)    # (nmodel, ncolumns) matrix on the model grid
```

Optional methods (defaults in brackets):

```julia
Specje.nparams(c)                          # [0]
Specje.in_bounds(c, θ)                     # [true]
Specje.to_free(c, θ), Specje.from_free(c, u)   # [identity]
Specje.log_jacobian(c, u)                  # [0]  log|∂θ/∂u| of from_free
Specje.physical_weights(c, w)              # [w]
Specje.check_component(c, npix)            # [nothing]; validate at model construction
Specje.default_priors(c)                   # [none]; needed for BayesModel
Specje.check_photometry(c, phot), Specje.phot_columns(c, phot, θ)   # for JointModel
```

To keep gradients working, write `columns` (and the maps) for generic element types.
Allocate with `promote_type(eltype(…), eltype(θ))`, not `Float64`, so ForwardDiff dual
numbers can pass through.
