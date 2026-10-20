# Concepts and conventions

## Wavelengths and frames

- All wavelengths are in **Å**. FSPS (and so `export_fsps.py`) gives **vacuum**
  wavelengths, and the built-in emission-line list ([`default_gas_groups`](@ref)) is
  vacuum too. Convert air-wavelength data to vacuum first.
- Fits run in the **rest frame** of a systemic redshift `z` that you choose: pass
  `wave ./ (1 + z)` and FWHM values (Å) divided by `1 + z`. The fitted velocities are
  residuals relative to `z`.
- Velocity dispersions in km/s don't depend on the frame. That's why an LSF can be
  built from either observed or rest wavelengths, as long as the FWHM is converted
  consistently.
- [`Photometry`](@ref) takes **observer-frame** filters and the redshift `z`. It shifts
  the filters to the rest frame itself.

## Velocities and pixels

Specje uses the pPXF convention `v = c · Δln λ`. On a log-uniform grid with `velscale`
km/s per pixel ([`velscale`](@ref)), a velocity `v` is a shift of `v / velscale`
pixels. Every grid involved in the stellar kinematics must be log-uniform with the
**same** velscale. [`kinematics_grid`](@ref) checks this to a relative tolerance of 1e-6.

## Model grid and data pixels

The model is computed on a **model grid**, log-uniform in wavelength. The data can stay
on their **native pixels**, which may be linear, irregular, or log-uniform. A sparse
projection `P` ([`pixel_projection`](@ref)) averages the model onto them, conserving
flux:

```
A_data(θ) = P · A_model(θ)
```

This keeps the data's noise independent, as measured, so the diagonal likelihood is
exact. A resolution matrix on the native pixels (DESI) can be folded in too:
`A_data = R · P · A_model` ([`pixel_projection`](@ref) with `resolution=R`). Rebinning the *data* onto a log grid instead correlates neighbouring pixels,
and the errors then come out too small (see [Spectra and templates](guide/data.md#Native-pixels-or-log-rebinning)).
It's the same forward-modelling idea as DESI's spectroperfectionism: compare model and
data in the data's own pixel space, rather than transforming the data.

## The stellar convolution

Each template is convolved with a Gaussian LOSVD of velocity `v` and dispersion `σ`.
This is done in Fourier space with the analytic transform of the kernel
([`gaussian_losvd_ft`](@ref)):

```
K̂(f) = exp(−2π² σ² f²) · exp(−2πi f s),     s = v / velscale − frac,  σ in pixels
```

so the kernel is never sampled or truncated, and sub-pixel shifts are exact.
`frac` is the sub-pixel offset between the template and data grids.

The FFT makes the convolution **circular**: anything shifted past one end of the
template array reappears at the other. [`kinematics_grid`](@ref) prevents this from
reaching the data by requiring the templates to extend at least `margin_kms` (default
2500 km/s) beyond the data on both sides, and by zero-padding the transform.

!!! warning "Keep the parameter bounds inside the margin"
    `margin_kms` should exceed `vmax + 5 σ_max` of every stellar component. The defaults
    `StellarComponent(...; vmax=2000, σ_max=1000)` exceed the default 2500 km/s margin,
    so set `vmax` and `σ_max` to physically sensible values, or widen both the template
    wavelength range and `margin_kms`.

## The linear model and the weights

The model is `F(θ) = A(θ) w`. A [`SpectralModel`](@ref) builds `A` by concatenating the
columns of its components, in order:

| Component | Parameters `θ` | Columns |
|---|---|---|
| [`StellarComponent`](@ref) | `(v, σ)` km/s | one per template |
| [`GasComponent`](@ref) | `(v₁, σ₁, v₂, σ₂, …)`, one pair per [`GasGroup`](@ref) | one per [`Multiplet`](@ref) |
| [`PolynomialComponent`](@ref) | none | `2(degree + 1)`: Legendre `[P  −P]` |

In χ² fitting all weights are **non-negative** ([`nnls`](@ref)). A signed term, like
the additive polynomial, is therefore supplied as a `±` pair of columns.
[`split_weights`](@ref) recombines each pair into one signed coefficient.

### Template order

SSP templates are flattened **age fastest, metallicity slowest**: column `k`
corresponds to `(log_age[i], logzsol[j])` with `k = i + (j − 1)·nage`.
[`template_params`](@ref) lists the pairs in column order, and
[`reshape_weights`](@ref) turns a weight vector back into an `(nage, nmet)` matrix.

### What the weights mean

- **Stellar weights.** The fitted templates are `flux ./ norm`. After
  [`normalize_window`](@ref), `norm` is each SSP's mean flux in the window, so a weight
  is that SSP's contribution to the mean flux in the window, in data units.
  [`formed_mass`](@ref) `= w ./ norm` converts back to "mass formed", in units of
  (data flux unit) / (library flux unit per M☉). For SSP fluxes in L☉/Å per M☉ and data
  in erg/s/cm²/Å, multiply by `4π d_L² / L☉` (with consistent units) to get M☉.
  [`stellar_mass`](@ref) also applies the surviving-mass fraction.
- **Gas weights** are integrated line fluxes, in data flux density × Å, of the line
  whose ratio within its multiplet is 1.
- **Polynomial weights** are signed Legendre coefficients over the data pixels, with x
  mapped to `[−1, 1]`.

## Free coordinates

Optimisers and samplers work in unconstrained **free coordinates** `u`, not in `θ`:

| Parameter | Free coordinate |
|---|---|
| velocity `v` | `v / velscale` (pixels) |
| dispersion `σ` | `log(σ − σ_min)` |
| joint-fit calibration scale `s` | `log s` |
| weight scale `τ` (when fitted) | `log τ` |
| noise scale (when fitted) | `log s` |

[`to_physical`](@ref) and [`initial_free`](@ref) convert between the two. The log-σ
map keeps `σ > σ_min`. The upper bounds `σ ≤ σ_max` and `|v| ≤ vmax` are **hard walls**:
the χ² is `Inf` and the log posterior `-Inf` beyond them.

## The data vector

[`FitData`](@ref) holds the noise-weighted data on the good (native) pixels, `b = flux / noise`,
together with `1/noise` and the row indices. With photometry
(`FitData(spectrum, photometry)`) the photometric points are stacked under the spectral
pixels. Each source is a **block**, and noise-scale factors are fitted per block.

## The marginal likelihood

With the rows divided by `s_k σ_i` (`s_k` the noise scale of the row's block), the noise
is unit-variance. A zero-mean Gaussian prior `w_j ~ N(0, τ_j²)` on each weight then
gives the evidence in closed form. With `Λ = AᵀA + diag(1/τ²)`:

```
log p(d | θ, s, τ) = −½ [ n log 2π + log|Λ| + 2 Σ log τ_j + bᵀb − (Aᵀb)ᵀ Λ⁻¹ (Aᵀb) ]
                     − Σ_i log(s_k σ_i)
```

This needs one Cholesky factorisation of an `m × m` matrix, where `m` is the number of
columns, not the number of pixels ([`marginal_loglike`](@ref)).

!!! note "The prior ignores positivity"
    The analytic marginalisation uses a Gaussian prior on `w`, not a truncated one. So
    the posterior over `θ` is integrated over negative weights too. In practice the
    prior acts as ridge regularisation. Positivity comes back when you draw the weights
    themselves with `sample_weights(...; positive=true)`. The prior scales `τ` matter:
    with very large `τ`, the Occam factor `log|Λ| + 2 Σ log τ` dominates the evidence.
    Use the defaults ([`default_tau`](@ref)), choose `τ` deliberately, or fit it with
    `fit_tau=true`.

## Gradients

The log posterior can be differentiated with ForwardDiff (forward mode).
`LogDensityProblems.logdensity_and_gradient` is implemented, so a [`BayesModel`](@ref)
works with any gradient-based sampler. The FFT step transforms the value and each
partial derivative separately. One gradient costs about 2–3 evaluations of the density
for a model with a few parameters.

Reverse-mode AD (Zygote) is not supported: the gas and banded-LSF code fill arrays in
place. With a handful of nonlinear parameters, forward mode is the right choice anyway.
