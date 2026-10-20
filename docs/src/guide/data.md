# Spectra and templates

## Spectra

```julia
Spectrum(wave, flux, noise; good=trues(length(wave)), lsf_fwhm=nothing)
```

- `wave`: pixel centres in Å, strictly increasing.
- `flux`, `noise`: flux density per Å and its 1σ error, in any consistent unit.
- `good`: pixels to use. Pixels with non-finite flux or noise, or `noise ≤ 0`, are
  always marked bad, whatever you pass.
- `lsf_fwhm`: optional instrumental FWHM per pixel, in Å. Needed for
  `GaussianLSF(spectrum)` and resolution matching.

Useful functions:

| Function | Purpose |
|---|---|
| `npix(s)` | number of pixels |
| [`velscale`](@ref)`(s)` | km/s per pixel of a log-uniform grid |
| [`is_loguniform`](@ref)`(s)` | check the grid is uniform in `ln λ` |
| [`crop`](@ref)`(s, λmin, λmax)` | keep `λmin ≤ wave ≤ λmax` |

### Masking

To mask sky lines, chip gaps, or emission lines you don't model, set `good` to `false`
for those pixels. Masked pixels are left out of χ², the likelihood, and the weight
solution, but the model is still computed there ([`model_flux`](@ref)). With a projection,
mask native pixels: the model grid has no mask.

### Native pixels or log rebinning

The stellar kinematics are computed on log-uniform pixels. There are two ways to
connect them to data on any other grid.

**Recommended: keep the native pixels and project the model.**

```julia
mwave = model_grid(spec.wave; oversample=2)
P     = pixel_projection(mwave, spec.wave)
model = SpectralModel(grid, components...; projection=P)   # components built on mwave
fit   = fit_kinematics(model, spec, θ0)                    # spec: native pixels
```

[`model_grid`](@ref) returns a log grid whose pixels cover every native pixel, at a
velscale `oversample ×` finer than the narrowest one. [`pixel_projection`](@ref)
returns the sparse matrix `P` that averages model pixels onto native pixels, weighted
by overlap and conserving flux: `P * f == rebin_flux(edges, log_pixel_edges(mwave), f)`, where
`edges` are the native pixel edges (geometric midpoints for log-uniform data such as SDSS,
arithmetic midpoints otherwise; pass `data_edges` to give them explicitly).
The design matrix becomes `P · A_model(θ)`. The data and their errors are left exactly
as measured, so the independent-pixel likelihood is correct. Projecting costs a fraction
of a millisecond per evaluation, and gradients pass through it.

Build every component on the model grid: templates matched with
`match_resolution(tlib, mwave, lsf)`, `GasComponent(mwave, …)`,
`PolynomialComponent(length(mwave), …)`, and the LSF with
[`GaussianLSF`](@ref)`(spec, mwave)`.

**Alternative: rebin the data.**

```julia
data = rebin_log(spec; velscale=median_velscale)
```

[`rebin_log`](@ref) resamples onto a log-uniform grid. Each new pixel is the
overlap-weighted mean of the old pixels it covers (the SpectRes algorithm), so flux
density is conserved. Errors are propagated assuming independent pixels, and
`lsf_fwhm` is interpolated. A new pixel is good only if it is fully covered by old
pixels and every old pixel it overlaps is good. Bad pixels get flux 0 and noise `Inf`.

Rebinning makes neighbouring pixels share old pixels, so their errors are correlated.
The likelihood ignores that, and the parameter errors come out too small. In the
quickstart simulation (0.5 Å native pixels, rebinned to their median velscale), the
actual scatter of σ★ over 30 noise draws was 5.76 km/s, while the fits quoted 4.57 km/s:
26% too small. Projecting the model gave 5.65 km/s scatter against 5.74 km/s quoted.
The estimates themselves were unbiased either way. Rebinning is still convenient for
plotting, for data that are already log-uniform, and for quick looks.

The lower-level functions [`rebin_flux`](@ref), [`rebin_noise`](@ref),
[`rebin_good`](@ref) take explicit pixel edges (see [`pixel_edges`](@ref),
[`log_pixel_edges`](@ref), [`loggrid`](@ref)). `rebin_flux` also accepts a matrix, one
spectrum per column.

## SSP libraries

```julia
SSPLibrary(wave, flux, log_age, logzsol, mass_frac; norm=ones, lsf_fwhm=nothing)
```

| Field | Shape | Meaning |
|---|---|---|
| `wave` | `nwave` | Å, vacuum, increasing |
| `flux` | `(nwave, nage, nmet)` | L☉/Å per M☉ *formed* (FSPS units) |
| `log_age` | `nage` | log₁₀(age / yr), increasing |
| `logzsol` | `nmet` | log₁₀(Z / Z☉), increasing |
| `mass_frac` | `(nage, nmet)` | surviving mass (stars + remnants) per M☉ formed |
| `norm` | `(nage, nmet)` | per-template scale; fitted templates are `flux ./ norm` |
| `lsf_fwhm` | `nwave` or `nothing` | library resolution, Gaussian FWHM in Å |

### From FSPS

`export_fsps.py` (in the repository root) writes an FSPS SSP grid to HDF5. It needs
`python-fsps` and `h5py`:

```bash
python export_fsps.py --out fsps_ssp.h5 [--imf-type 1] [--wmin 3525 --wmax 7500] [--lsf-fwhm 2.5]
```

- `--imf-type`: 0 Salpeter, 1 Chabrier (default), 2 Kroupa.
- `--wmin/--wmax`: wavelength range in Å. The default is the MILES range. Outside it,
  FSPS splices in a lower-resolution library, so a constant `lsf_fwhm` would be wrong.
- `--lsf-fwhm`: the library FWHM in Å (2.5 Å is the usual MILES value; check your build).

Read it with [`read_fsps_hdf5`](@ref). [`write_ssp_hdf5`](@ref) writes any
`SSPLibrary` in the same layout, for example to cache a log-rebinned library.

For photometry you need a second, **wide** export (e.g. `--wmin 1000 --wmax 30000`) on
the same age/metallicity grid; see [Photometry](photometry.md).

### Selecting and preparing templates

```julia
lib  = subset_library(lib; log_age=(8.0, 10.2), logzsol=(-1.0, 0.3))
tlib = rebin_log(lib; velscale=velscale(mwave), λmin=3600.0, λmax=7400.0)
tlib = normalize_window(tlib; window=(5300.0, 5700.0))
```

- [`subset_library`](@ref) keeps a range of ages and metallicities.
- [`crop_wave`](@ref) keeps a wavelength range.
- [`rebin_log`](@ref)`(lib; velscale, λmin, λmax)` crops and log-rebins every SSP. Use
  the model grid's velscale exactly (or the data's, if you rebinned the data). Choose `λmin`/`λmax` so the templates extend at least
  `margin_kms` beyond the data on both sides (≈ 40 Å at 5000 Å for the default 2500 km/s).
- [`normalize_window`](@ref) sets `norm` to each SSP's mean flux in `window`. This
  replaces any existing `norm`, so call it once, after rebinning.

### Template matrix and bookkeeping

| Function | Returns |
|---|---|
| [`templates`](@ref)`(lib)` | `(nwave, nage·nmet)` matrix of `flux ./ norm`, age fastest |
| [`template_params`](@ref)`(lib)` | `(log_age=…, logzsol=…)` per column |
| [`reshape_weights`](@ref)`(lib, w)` | weights as `(nage, nmet)` |
| [`formed_mass`](@ref)`(lib, W)` | `W ./ norm` |
| [`stellar_mass`](@ref)`(lib, W)` | `formed_mass .* mass_frac` |
| [`age_gyr`](@ref)`(lib)` | ages in Gyr |

For example, the light-weighted mean age and metallicity of a fit:

```julia
W = reshape_weights(tlib, first(split_weights(model, fit.weights)))
p = template_params(tlib)
w = vec(W) ./ sum(W)
mean_log_age = sum(w .* p.log_age)
mean_logzsol = sum(w .* p.logzsol)
```
