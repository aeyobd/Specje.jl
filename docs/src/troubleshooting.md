# Troubleshooting

## Errors at set-up

**`template velscale … ≠ data velscale …`**
The templates and the model grid must have the same velscale. Rebin the library with
`rebin_log(lib; velscale=velscale(mwave), …)`, using the model grid's velscale (or the
rebinned data's).

**`σ_min = … must be ≥ σ_templ = …`**
The templates are already broadened by `σ_templ`, so smaller dispersions can't be
fitted. Raise `σ_min`, or use a sharper library or narrower wavelength range.

**`resolution matrix is N×N, data have M pixels`**
The resolution matrix must be defined on exactly the spectrum's native pixels. Check
you haven't cropped or masked the wavelengths: mask with `good` instead.

**`data pixel N is not covered by the model grid`**
Build the model grid with `model_grid(spec.wave)`, which covers every native pixel.

**`polynomial has N pixels, model has M`** / **`gas component has N pixels…`**
With a projection, components are built on the model grid: use `length(mwave)` and
`mwave`, not the native pixel count or wavelengths.

**`template and data wavelengths must both be log-uniform`**
Rebin with [`rebin_log`](@ref). Check with [`is_loguniform`](@ref).

**`templates must extend ≥ N pixels beyond the data on both sides`**
Widen `λmin`/`λmax` when rebinning the library, crop the data, or lower `margin_kms` in
[`kinematics_grid`](@ref) (but keep it above `vmax + 5σ_max`).

**`templates are broader than the data in N pixels`** (from `match_resolution`)
The data have better resolution than the library in those pixels. Use
`match_resolution_auto`, which folds the deficit into the LOSVD and fits exactly for
galaxies with σ above `minimum_sigma_templ`. `strict=false` also runs, but biases σ
low (see [Libraries broader than the data](guide/resolution.md#Libraries-broader-than-the-data)).

**`initial σ must exceed σ_min`**
Starting values must satisfy `σ > σ_min` for every stellar component and gas group.

**`group '…' has no lines in […] Å`**
A [`GasGroup`](@ref) has no lines inside the data range. Drop it, or check that the data
wavelengths are rest-frame.

**`rebinned library has non-finite flux`**
The requested `λmin`/`λmax` aren't fully covered by the library.

## The fit doesn't move, or stops at once

`fit_kinematics` and `map_estimate` use Nelder–Mead with a fixed initial step of 0.5 in
every free coordinate: half a pixel in velocity, and a factor of about e^0.5 in
`σ − σ_min`. If the very first steps land outside the bounds, the run can stall. Check
`fit.converged` and `fit.iterations`. If `iterations` is 0 or very small:

- start closer to the answer, away from `σ_min`, `σ_max` and `±vmax`;
- check the bounds are wide enough for the data;
- check `profile_chi2(model, FitData(data), θ0)` is finite.

## The fit converges to the wrong solution

- **Local minimum.** Try several `θ0`, especially for `v` (aliasing between absorption
  lines) and for multi-component models.
- **Wrong frame.** Velocities near `c·z` mean the data weren't divided by `1 + z`.
- **Resolution not matched.** σ★ is biased high if the templates weren't smoothed to
  the data's LSF, and low if they were smoothed too much (e.g. a wrong library
  `lsf_fwhm`).
- **Resolution matrix in the projection, σ★ too low.** The templates' own resolution
  must be declared with `σ_templ` (after `match_resolution(tlib, σ_templ)`). Otherwise
  σ★ comes out as `√(σ² − σ_templ²)`, with no sign of it in χ². Also check the
  `sampling` convention.
- **Unmasked emission or artefacts.** Mask sky lines, bad columns, and any emission
  lines you aren't modelling.
- **σ below the resolution.** Dispersions well below the instrumental σ (or about one
  pixel) are poorly constrained, and the posterior piles up at `σ_min`.

## Errors look too small

- **Rebinned data.** Rebinning correlates neighbouring pixels, but the likelihood
  treats them as independent. In tests the errors came out about 25% too small. Fit
  the native pixels with a projection instead (`model_grid`, `pixel_projection`).
- **Mis-scaled noise.** If `reduced_chi2(fit)` is far from 1, fit `fit_noise=true` in
  the posterior, or rescale `noise`.
- **Template mismatch.** No SSP library matches a real galaxy exactly. Systematics from
  the library choice, polynomial degree and masking are often larger than the
  statistical errors. Compare fits that vary them.

## The posterior sampler struggles

- Many NUTS divergences, or ensemble walkers stuck: the posterior is probably pressed
  against a hard wall (`σ_min`, `σ_max`, `vmax`). Check the posterior isn't piling up
  at a bound, and keep the priors inside the bounds.
- A low ensemble acceptance fraction (< 0.1): start from `map_estimate`, reduce
  `ball_scale`, or use more walkers.
- Very long autocorrelation times: the posterior is strongly correlated (e.g. v★ and
  v_gas when lines overlap absorption features). NUTS copes with this much better than
  the ensemble sampler.
- Gas parameters wander over their prior: there are no detectable emission lines.
  Remove the gas component, or report upper limits on the fluxes.

## Weights

- **Many zero weights.** That's normal for NNLS: the solution is sparse. The weights of
  similar SSPs are degenerate, so look at derived quantities (mean age, metallicity,
  mass) and their posterior, not at individual weights.
- **Negative "weights".** Only the polynomial's combined coefficients can be negative.
  If you see negative SSP weights in `weight_posterior(…).mean`, that's the
  unconstrained Gaussian mean; draw with `sample_weights(…; positive=true)`.
