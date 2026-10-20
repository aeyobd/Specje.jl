# Specje.jl

Specje fits galaxy spectra with a library of simple stellar populations (SSPs). Given a
spectrum, and optionally broadband photometry, it measures:

- the **stellar line-of-sight velocity distribution** (Gaussian: velocity `v` and
  dispersion `σ`),
- **emission-line kinematics and fluxes** for groups of lines sharing `(v, σ)`,
- the **stellar-population weights**: how much light, and hence mass, each SSP of a
  given age and metallicity contributes,

either as a best fit (χ²) or as a posterior distribution.

## How it works

The model spectrum is linear in a set of non-negative **weights** `w` and nonlinear in a
few **kinematic parameters** `θ`:

```
F(θ) = A(θ) w
```

Each column of the design matrix `A(θ)` is one template: an SSP spectrum broadened and
shifted by the stellar LOSVD, an emission-line multiplet at the gas `(v, σ)`, or an
additive polynomial term. Because the problem is linear in `w`, Specje never searches
over the weights:

- **χ² fitting** ([`fit_kinematics`](@ref)) optimises over `θ` only, solving for the best
  non-negative `w` at every step (variable projection, as in pPXF).
- **Bayesian inference** ([`BayesModel`](@ref)) integrates `w` out analytically against a
  Gaussian prior, which leaves a low-dimensional posterior over `θ`. Gradients come from
  forward-mode AD, so NUTS/HMC samplers work directly. The weights can be drawn afterwards
  from their conditional posterior.

## Package layout

| Area | Main entry points |
|---|---|
| Spectra and pixels | [`Spectrum`](@ref), [`model_grid`](@ref), [`pixel_projection`](@ref), [`rebin_log`](@ref), [`crop`](@ref) |
| SSP libraries | [`SSPLibrary`](@ref), [`read_fsps_hdf5`](@ref), [`normalize_window`](@ref), [`templates`](@ref) |
| Resolution | [`GaussianLSF`](@ref), [`BandedLSF`](@ref), [`match_resolution`](@ref) |
| Kinematics | [`kinematics_grid`](@ref), [`convolve_templates`](@ref), [`gaussian_losvd_ft`](@ref) |
| Model | [`StellarComponent`](@ref), [`GasComponent`](@ref), [`PolynomialComponent`](@ref), [`SpectralModel`](@ref) |
| Fitting | [`fit_kinematics`](@ref), [`FitData`](@ref), [`nnls`](@ref) |
| Photometry | [`Filter`](@ref), [`Photometry`](@ref), [`band_templates`](@ref), [`JointModel`](@ref) |
| Inference | [`BayesModel`](@ref), [`map_estimate`](@ref), [`sample_ensemble`](@ref), [`sample_weights`](@ref) |

## Where to start

- New to the package: read the [Tutorial](tutorial.md), and run `examples/quickstart.jl`.
- Before fitting real data: read [Concepts and conventions](concepts.md). Wavelength
  frames, velocity units and the velscale requirement trip people up most.
- Something not working: see [Troubleshooting](troubleshooting.md).
