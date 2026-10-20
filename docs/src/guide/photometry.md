# Photometry

A spectrum covers a limited wavelength range, and its flux calibration is often uncertain.
Broadband photometry adds the overall spectral energy distribution, which constrains
ages and metallicities that the spectrum alone leaves degenerate. Specje fits both
together, with one set of stellar-population weights.

## Filters

```julia
Filter(name, wave, trans)               # tabulated response, wave in Å
gaussian_filter(name, λ0, σ)            # for tests and toy setups
tophat_filter(name, a, b)
```

A [`Filter`](@ref) is a **photon-counting** response, with `norm = ∫ T λ dλ`
precomputed. The band-averaged flux density is

```
⟨F⟩ = ∫ F(λ) T(λ) λ dλ / ∫ T(λ) λ dλ
```

Transmission is zero outside the tabulated range: trim long zero tails before
constructing a filter. Useful functions: [`transmission`](@ref)`(f, λ)`,
[`pivot_wavelength`](@ref)`(f)`, [`to_rest`](@ref)`(f, z)`.

## Photometric data

```julia
Photometry(filters, flux, err; z=0, good=trues, error_floor=0)
```

- `filters`: **observer-frame** filters, one per measurement.
- `flux`, `err`: band-averaged **f_λ in the same units as the spectrum**. Convert
  magnitudes or Jansky with [`ab_to_flambda`](@ref), [`jy_to_flambda`](@ref),
  [`mag_err_to_flux_err`](@ref). Each takes a `unit` keyword: your spectrum's flux unit
  in erg/s/cm²/Å, e.g. `unit=1e-17`.
- `z`: the systemic redshift used for the spectrum. The filters are shifted to the rest
  frame internally.
- `error_floor`: a fractional systematic error added in quadrature,
  `hypot(err, error_floor · |flux|)`. Broadband zero points and SSP models are rarely
  better than a few per cent, so 0.02–0.05 is common.
- Points with non-finite flux, or `err ≤ 0`, are flagged bad.

## Band templates

Filters usually extend far beyond the spectral templates, so the band fluxes of each
SSP come from a **wide** copy of the library. It must have the same ages and
metallicities, and is exported e.g. with `--wmin 1000 --wmax 30000`:

```julia
wide  = read_fsps_hdf5("fsps_ssp_wide.h5")
bands = band_templates(wide, tlib, phot)      # tlib: the normalised spectral library
star  = StellarComponent(grid, T; phot=bands, σ_min=10.0, σ_max=600.0, vmax=1000.0)
```

[`band_templates`](@ref) applies `tlib`'s normalisation, so a weight means the same
thing in the spectral and photometric rows. Band fluxes don't depend on the stellar
LOSVD (σ ≈ 100 km/s is negligible against a filter's width), so they are computed once.

How each component contributes to the photometry:

| Component | Photometric columns |
|---|---|
| `StellarComponent` | the cached band fluxes (needs `phot=`) |
| `GasComponent` | each line integrated through the filters at its shifted wavelength |
| `PolynomialComponent` | zero (a spectral-calibration nuisance) |

## Joint model

```julia
jm = JointModel(model, phot; fit_scale=false, s_min=1e-3, s_max=1e3)
fit = fit_kinematics(jm, data, [θ0; 1.0])          # last entry: scale s (if fit_scale)
```

The stacked design matrix is `[s · A_spec(θ); A_phot(θ)]`. With `fit_scale=true`, an
extra positive parameter `s` (last in `θ`) absorbs aperture and calibration mismatch:
the data spectrum is modelled as `s ×` the photometric-scale model. Use it whenever the
spectrum and photometry don't come from the same aperture.

After fitting:

```julia
spectral_scale(jm, fit)       # s (1 when not fitted)
model_spectrum(jm, fit)       # model spectrum on the photometric scale (data ≈ s × this)
model_photometry(jm, fit)     # model band fluxes
```

For Bayesian inference, build the data with `FitData(data, phot)`. The spectrum and the
photometry are separate **blocks**, so `fit_noise=true` fits an independent noise scale
for each:

```julia
bm = BayesModel(jm, FitData(data, phot), fit.θ; fit_noise=true)
```

With thousands of spectral pixels and a handful of bands, the spectrum dominates the
likelihood. If the photometry should carry more weight, use realistic `error_floor`s
and fitted noise scales, not ad hoc weighting.
