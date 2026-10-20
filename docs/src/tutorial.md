# Tutorial

This walks through a complete analysis: prepare the data and templates, build a model,
fit it, then sample the posterior. The code follows `examples/quickstart.jl`, which you
can run as is. It simulates its own SSP library and spectrum, so it needs no external
data. To use real data, replace the library and spectrum in sections 1–2; the rest
stays the same.

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/quickstart.jl
```

## 1. Load an SSP library

With real data the library usually comes from FSPS via the exporter script that ships
with the repository:

```bash
python export_fsps.py --out fsps_ssp.h5 --imf-type 1 --wmin 3525 --wmax 7500 --lsf-fwhm 2.5
```

```julia
using Specje
lib = read_fsps_hdf5("fsps_ssp.h5")
nage(lib), nmet(lib), nwave(lib)
```

An [`SSPLibrary`](@ref) holds a flux cube `(nwave, nage, nmet)`, the age and metallicity
grids, the surviving mass fraction of each SSP, and optionally the library's resolution
(`lsf_fwhm`, in Å). The quickstart builds a toy library with the same structure.

## 2. Load the spectrum, in the rest frame

Specje works in the **rest frame**. Divide the observed wavelengths by `1 + z`, and
give FWHM values in Å divided by the same factor. The fitted velocity is then relative
to `z`.

```julia
rest = Spectrum(obs_wave ./ (1 + z), flux, noise;
                good=mask, lsf_fwhm=obs_fwhm ./ (1 + z))
```

`good` marks the pixels that enter the fit. Pixels with non-finite flux or noise, or
`noise ≤ 0`, are always excluded.

## 3. Choose a model grid

A velocity shift is a constant offset in `ln λ`, so the model is computed on log-uniform
pixels. The data **stay on their native pixels**. Their noise is independent there,
and resampling them would correlate it. Instead, the model is evaluated on a finer log
grid and averaged onto the native pixels by a projection matrix:

```julia
mwave = model_grid(rest.wave; oversample=2)    # log grid covering every native pixel
P     = pixel_projection(mwave, rest.wave)     # sparse (native pixels × model pixels)
vs    = velscale(mwave)                        # km/s per model pixel
```

[`model_grid`](@ref) picks a velscale `oversample ×` finer than the narrowest native
pixel. [`pixel_projection`](@ref) is the flux-conserving pixel average: each native
pixel is the overlap-weighted mean of the model pixels it covers, the same operation
[`rebin_log`](@ref) applies to data. See [Spectra and templates](guide/data.md#Native-pixels-or-log-rebinning)
for the alternative of rebinning the data, and why it understates the errors.

## 4. Prepare the templates

The templates must be on a log grid with **exactly the same velscale** as the model
grid, and must extend beyond it on both sides (by `margin_kms`, 2500 km/s by default):

```julia
tlib = rebin_log(lib; velscale=vs, λmin=3800.0, λmax=5600.0)
tlib = normalize_window(tlib; window=(4500.0, 4600.0))
```

[`normalize_window`](@ref) scales each SSP to unit mean flux in the window. The fitted
weights are then light fractions there, and the posterior's weight priors are well
scaled.

Next, bring the templates to the data's instrumental resolution. Then the fitted `σ`
is the galaxy's intrinsic dispersion:

```julia
lsf  = GaussianLSF(rest, mwave)              # the data's LSF, interpolated onto the model grid
T    = match_resolution(tlib, mwave, lsf)    # templates convolved by √(σ_data² − σ_lib²)
grid = kinematics_grid(tlib.wave, mwave)     # FFT geometry linking the two grids
```

## 5. Build the model

A [`SpectralModel`](@ref) concatenates components. Their nonlinear parameters are laid
out in order. Every component lives on the model grid, and `projection` maps the result
onto the native pixels:

```julia
star = StellarComponent(grid, T; σ_min=10.0, σ_max=600.0, vmax=1000.0)
gas  = GasComponent(mwave,
                    [GasGroup("balmer", Multiplet("Hbeta", 4862.683),
                                        Multiplet("Hgamma", 4341.692))],
                    lsf; σ_min=5.0, vmax=800.0)
poly = PolynomialComponent(length(mwave), 2)

model = SpectralModel(grid, star, gas, poly; projection=P)   # θ = (v★, σ★, v_gas, σ_gas)
nparams(model), ncolumns(model)                              # (4, 16)
npix(model)                                                  # native pixels
```

The 16 columns are 8 SSPs, 2 emission multiplets, and 6 polynomial columns (degree 2 as
`±` pairs, since every weight is non-negative).

## 6. Fit by χ²

```julia
fit = fit_kinematics(model, rest, [0.0, 150.0, 0.0, 100.0])
fit.θ, reduced_chi2(fit), fit.converged
```

`θ0` is the starting point. The optimiser is local, so start near the answer, or try
several starting points for complex models. The weights split per component:

```julia
w_star, w_gas, w_poly = split_weights(model, fit.weights)
W = reshape_weights(tlib, w_star)          # (nage, nmet) light weights
M = stellar_mass(tlib, W)                  # surviving mass per SSP, in flux units
```

`w_gas` holds the integrated line fluxes (data flux density × Å). `w_poly` holds the
signed polynomial coefficients.

## 7. Sample the posterior

[`BayesModel`](@ref) defines the posterior over `θ`, with the weights integrated out.
`fit_noise=true` also fits a noise-scale factor for each data block:

```julia
fd = FitData(rest)
bm = BayesModel(model, fd, fit.θ; fit_noise=true)
u_map = map_estimate(bm, initial_free(bm, fit.θ)).u
```

The sampler works in **free coordinates** `u`: pixel velocities, `log(σ − σ_min)`, and
log hyperparameters. [`to_physical`](@ref) converts back.

With the built-in ensemble sampler:

```julia
using Random, Statistics
chain = sample_ensemble(Xoshiro(1), bm, u_map; nsteps=2000)
acceptance_fraction(chain), autocorr_time(chain)
U = flatten_chain(chain; burn=500)
Θ = reduce(hcat, [to_physical(bm, u).θ for u in eachcol(U)])
mean(Θ; dims=2), std(Θ; dims=2)
```

Or with NUTS (see [Bayesian inference](guide/bayes.md)):

```julia
using AdvancedHMC, AbstractMCMC
samples = AbstractMCMC.sample(Xoshiro(1), AbstractMCMC.LogDensityModel(bm), NUTS(0.8), 2000;
                              n_adapts=1000, initial_params=u_map)
```

Draw the weights given a posterior sample of `θ`, constrained to `w ≥ 0`:

```julia
wp = weight_posterior(bm, U[:, end])
draws = sample_weights(Xoshiro(2), wp, 200; positive=true)   # (ncolumns, 200)
```

## What the quickstart prints

On the simulated data (truth `v = 80`, `σ = 180` km/s), one run gives

```
data: 2400 native pixels; model grid: 5444 pixels at 14.15 km/s
best fit θ = [73.7, 167.1, -41.5, 26.5],  χ²/N = 0.999
posterior mean θ = [74.9, 166.9, -27.0, 160.9], sd = [5.2, 5.4, 163.9, 154.3]
```

The simulated spectrum has no emission lines, so the gas parameters only reflect their
priors. σ★ is about 2.4 standard deviations from the truth: this particular noise draw
is unlucky. Fitting a stellar + polynomial model to 30 noise draws of the same
simulation gives σ★ = 179.9 on average, with a scatter of 5.65 km/s; the Δχ² = 1 error
quoted by the fits is 5.74 km/s. Without noise the full model returns σ★ = 180.4.
