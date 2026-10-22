# Instrumental resolution

Observed line widths combine the galaxy's velocity dispersion with the instrument's
line-spread function (LSF). Specje supports two LSF representations behind one
interface (`abstract type LSF`).

## Gaussian LSF

```julia
GaussianLSF(sigma_kms::Vector)        # σ in km/s per pixel
GaussianLSF(wave, fwhm)               # fwhm in Å, scalar or per pixel
GaussianLSF(spectrum)                 # from spectrum.lsf_fwhm
GaussianLSF(spectrum, model_wave)     # spectrum.lsf_fwhm interpolated onto another grid
```

With a projection, `P` applies the native pixels' own averaging, so the LSF should
describe the instrument's response *before* pixel integration. A FWHM measured from arc
lines on the native pixels already includes the pixel width, which is then counted
twice. That matters little when the pixels are much narrower than the LSF. Otherwise,
subtract the pixel's contribution (a variance of width²/12) in quadrature first.

Conversions: [`fwhm_to_sigma_kms`](@ref)`(fwhm, wave)` and
[`sigma_kms_to_fwhm`](@ref)`(σ, wave)`, with `σ = c · FWHM / (2√(2 ln 2) · λ)`.

## Banded resolution matrix

Some pipelines (DESI, for example) give a full resolution matrix `R` per spectrum:
`model_observed = R * model_true`. [`BandedLSF`](@ref) stores it by diagonals:

```julia
BandedLSF(bands, offsets)    # bands[d, i] = R[i, i + offsets[d]]
banded_from_dia(data, offsets)   # from scipy dia_matrix / DESI storage
```

Useful functions:

| Function | Purpose |
|---|---|
| [`apply_banded`](@ref)`(R, X)` | `R * X` for a vector or each matrix column |
| [`dense_matrix`](@ref)`(R)` | full `npix × npix` matrix (small problems, tests) |
| [`normalize_rows`](@ref)`(R)` | rescale rows to sum to 1 |
| [`banded_gaussian`](@ref)`(σ_pix; nsig=5)` | banded matrix of pixel-integrated Gaussians |
| [`to_banded`](@ref)`(lsf, velscale)` | banded version of a `GaussianLSF` |
| [`inst_sigma_kms`](@ref)`(lsf, velscale)` | effective σ per pixel (km/s) |

For a `BandedLSF`, `inst_sigma_kms` is the square root of each row's second central
moment. That is exact for Gaussian rows, and includes the 1/12 px² from pixel
integration.

## Resolution matrices in the forward model (DESI)

A resolution matrix can enter the model exactly, as part of the projection onto the
native pixels. Its full shape is used (asymmetry, wings, variation along the spectrum),
not just an effective Gaussian width:

```
A_data(θ) = R · P · A_model(θ)
```

Here `A_model` holds the model with **no instrumental broadening**, on the log model
grid; `P` samples it onto the native pixels; and `R` applies the instrument there.

```julia
R     = banded_from_dia(resolution_data, offsets)     # per spectrum, on its native pixels
mwave = model_grid(spec.wave; oversample=2)
P     = pixel_projection(mwave, spec.wave; resolution=R)          # R · P
```

`spec.wave` must be the pixel grid that `R` is defined on, in the rest frame. Dividing
by `1 + z` doesn't change `R`, which is defined in pixels.

!!! note "DESI storage"
    desispec stores each spectrum's resolution as `(ndiag, nwave)` diagonals with
    offsets `ndiag÷2, ndiag÷2 − 1, …, −ndiag÷2`, i.e. `offsets = collect(ndiag÷2:-1:-(ndiag÷2))`.
    Pass the array with the diagonal index first. Check the conversion once for your
    files: `dense_matrix(R)` should equal desispec's `Resolution(data).toarray()`.
    Each camera (b, r, z) has its own pixels and R. `Spectrum` needs strictly increasing
    wavelengths, so fit one camera or a camera-coadded spectrum, not overlapping arms
    stacked together.

### Sampling convention

`pixel_projection(…; sampling=…)` decides what `R` acts on:

- `:average` (default): the model averaged over each native pixel;
- `:point`: the model sampled at the pixel centres (linear interpolation).

Use whichever matches how your resolution matrix was defined. If R was derived for
point-sampled flux, `:average` adds an extra boxcar of one pixel width, with variance
`Δλ²/12`. For DESI's 0.8 Å pixels and ≈ 1.8 Å FWHM, that is ≈ 9% of the instrumental
variance, enough to lower σ★ by ≈ 4% at 50 km/s (much less at higher σ).

### Templates and gas

`R` supplies all the instrumental broadening, so the components must not add any:

- **Gas.** Build it without an LSF: `GasComponent(mwave, groups; …)`.
- **Stellar templates.** These can't be infinitely sharp: the SSP library has its own
  resolution, and `R` acts on top of it. Bring the library to a resolution that is
  **constant in velocity**, σ_templ, and declare it:

```julia
m    = match_resolution_auto(tlib, mwave)          # no LSF: R applies the instrument
star = StellarComponent(grid, m.templates; σ_templ=m.σ_templ, σ_max=600.0, vmax=1000.0)
gas  = GasComponent(mwave, groups; σ_min=5.0, vmax=800.0)
model = SpectralModel(grid, star, gas, PolynomialComponent(length(mwave), 4); projection=P)
```

Without an LSF, [`match_resolution_auto`](@ref) brings the library to its coarsest
resolution *over the data's wavelength range*, constant in velocity, and returns that
σ_templ (`σ_min` defaults to it). You can also do this by hand:
`σ_templ = maximum(library_sigma_kms(tlib))` and `match_resolution(tlib, σ_templ)`,
which uses the whole template range, margins included.

The LOSVD kernel then has width `√(σ² − σ_templ²)`, so the fitted σ is the galaxy's
full dispersion.

!!! warning "Declare σ_templ: the data can't tell you"
    If the templates' own width is left out, the fit is just as good, but the result is
    `σ_fit = √(σ_true² − σ_templ²)`. The broadening is exactly degenerate with the
    LOSVD, so χ² can't flag it. In the tests, a true σ = 90 km/s with σ_templ = 40 km/s
    gives 90.0 when declared, and 80.6 when not.

The cost is that dispersions below σ_templ can't be fitted (`σ_min ≥ σ_templ` is
enforced). That is a property of the library, not of the method: a library broader
than the galaxy's lines can't measure them. With MILES (2.5 Å FWHM), σ_templ ≈ 64 km/s
at 5000 Å, but ≈ 88 km/s if the fit extends to 3600 Å. Restricting the wavelength
range, or using a sharper library, lowers it.

With a resolution matrix the whole library width counts, because R is applied on top
of it. If that σ_templ is too large for your galaxies, the alternative is to drop R
from the projection and match to its effective Gaussian width instead:

```julia
edges     = Specje.native_pixel_edges(spec.wave)
pixel_kms = C_KMS .* diff(log.(edges))              # each native pixel's width in km/s
σ_R       = inst_sigma_kms(R, 1.0) .* pixel_kms     # R's effective σ, pixel by pixel
# R's first and last rows are truncated, so they look sharper than the instrument:
# give them the width of the nearest complete row
half = maximum(abs, R.offsets)
σ_R[1:half] .= σ_R[half + 1]
σ_R[end-half+1:end] .= σ_R[end-half]
m = match_resolution_auto(tlib, mwave, GaussianLSF(Specje.sigma_on_grid(σ_R, spec.wave, mwave)))
P = pixel_projection(mwave, spec.wave)              # no resolution= here
```

The conversion is per pixel because linear pixels, like DESI's, change width in km/s
along the spectrum. Without the edge fix, the truncated rows create a large spurious
deficit at both ends, and σ_c would be set by them.

This only costs the *deficit* `√(σ_lib² − σ_R²)` (next section), at the price of
treating R as a Gaussian.

## Matching templates to the data

Before fitting the stellar kinematics, smooth the templates to the data's resolution,
so the fitted `σ` is intrinsic:

```
σ_diff² = σ_data² − σ_templates²        (pPXF convention)
```

```julia
T = match_resolution(tlib, data.wave, lsf; strict=true)
```

- `tlib` must be log-uniform and have `lsf_fwhm`. The result is a template matrix on
  `tlib.wave`, ready for [`StellarComponent`](@ref).
- `data.wave` should be rest-frame. The data σ is interpolated onto the template
  wavelengths (clamped at the ends).
- A `BandedLSF` enters through its effective Gaussian σ (`inst_sigma_kms`).
- With `strict=true` (the default), an error is raised wherever the templates are
  *broader* than the data, i.e. where the data have better resolution than the library
  and can't be matched by smoothing. Equal resolutions (to 1 part in 10⁹) are not
  broader. With `strict=false` those pixels are left unsmoothed, and the fitted σ is
  biased low (see below). For these cases use `match_resolution_auto` instead.
- The smoothing adds exactly `σ_diff²` of variance: the kernel's pixel-integrated
  weights would add another 1/12 px², which is subtracted first.

## Libraries broader than the data

No current SSP library is sharper than every modern spectrograph everywhere. Where the
library is broader, the deficit

```
d(λ) = √(σ_lib² − σ_data²)
```

can't be removed by smoothing, but it can be folded into the LOSVD. Choose one constant
σ_c ≥ max d (constant because the LOSVD kernel has one width in velocity), bring the
templates to `√(σ_data² + σ_c²)` at every wavelength (never sharper than the library,
so always possible), and fit with a kernel of width `√(σ² − σ_c²)`:

```julia
m    = match_resolution_auto(tlib, mwave, lsf)     # (templates=…, σ_templ=σ_c)
star = StellarComponent(grid, m.templates; σ_templ=m.σ_templ, σ_max=600.0, vmax=1000.0)
```

The fitted σ is then the galaxy's intrinsic dispersion, exactly, for any σ ≥ σ_c, and
`σ_min` defaults to σ_c. [`minimum_sigma_templ`](@ref)`(tlib, mwave, lsf)` gives σ_c
alone; only library pixels inside the data's wavelength range count. Pass
`σ_templ=…` to use a larger value; a smaller one is refused. When the library is
sharper everywhere, σ_c = 0 and the result is the same as `match_resolution`.

For example, MILES (2.5 Å FWHM, σ_lib ≈ 33.5 km/s at 9500 Å) against data with a 0.8 Å
FWHM (≈ 10.7 km/s) at 9500 Å has a deficit of ≈ 32 km/s there. Ignoring it
(`strict=false`) gives `σ_fit ≈ √(σ_true² − d²)`, with d varying along the spectrum:

| true σ | fitted σ with `strict=false` |
|---|---|
| 50 km/s | ≈ 39 km/s (−23 %) |
| 100 km/s | ≈ 95 km/s (−5 %) |
| 200 km/s | ≈ 197 km/s (−1.3 %) |

In the tests, a library at 60 km/s against data at 30 km/s and a true σ of 90 km/s
gives 90.00 with `match_resolution_auto` and 73.5 with `strict=false`.

!!! note "σ_c is set by the worst wavelength"
    A single wavelength where the library is much broader than the data raises σ_c
    for the whole fit. [`resolution_deficit_kms`](@ref)`(tlib, mwave, lsf)` returns the
    deficit at each wavelength, so you can see which ones set it. Check it before
    fitting low-dispersion galaxies, and if needed restrict the fit to where the library
    is sharp enough. Spuriously sharp LSF values, such as the truncated edge rows of a
    resolution matrix, show up there as spikes.

A lower-level method works on any template matrix:

```julia
match_resolution(T, twave, tfwhm, dwave, dlsf, velscale; strict=true)
```

and [`sigma_diff_kms`](@ref)`(σ_data, σ_templ)` gives the width to apply.

## Emission lines and the LSF

[`GasComponent`](@ref) applies the LSF itself, so don't pre-broaden the gas:

- with a `GaussianLSF`, the instrumental σ at each line's wavelength is added in
  quadrature to the intrinsic σ;
- with a `BandedLSF`, the resolution matrix is applied exactly to the intrinsic
  line profiles.

The fitted gas `σ` is therefore intrinsic, like the stellar one.
