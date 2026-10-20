# Specje.jl

Full-spectrum fitting of galaxy spectra with simple stellar population (SSP) templates,
in Julia. Specje measures stellar and gas kinematics and stellar-population weights,
optionally adds broadband photometry, and gives the posterior over everything with the
linear weights integrated out analytically.

- **Forward model** `F(θ) = A(θ) w`: stellar templates convolved with a Gaussian LOSVD
  (analytic Fourier transform, no kernel sampling), Gaussian emission-line multiplets,
  and an additive Legendre polynomial.
- **χ² fitting** by variable projection: Nelder–Mead over the few nonlinear parameters,
  non-negative least squares for the weights at each step (the pPXF approach).
- **Bayesian inference**: weights marginalised in closed form, priors on the physical
  parameters, optional fitted noise scales and hierarchical weight scales. Gradients
  come from ForwardDiff, so any `LogDensityProblems` sampler works, NUTS included. A
  gradient-free ensemble sampler is built in.
- **Native pixels**: the model is computed on a fine log grid and projected onto the
  data's own pixels, so the noise stays independent and the errors stay honest. No
  rebinning of the data is needed.
- **Data handling**: flux-conserving log rebinning, resolution matching (Gaussian or
  banded resolution matrices), FSPS library import, joint fits with photometry.

## Installation

Specje isn't registered. Add it from a local checkout:

```julia
using Pkg
Pkg.develop(path="/path/to/Specje.jl")
```

## Quick start

```julia
using Specje

lib  = read_fsps_hdf5("fsps_ssp.h5")              # see export_fsps.py
spec = Spectrum(wave ./ (1 + z), flux, noise; lsf_fwhm=fwhm ./ (1 + z))

mwave = model_grid(spec.wave)                      # log model grid covering the pixels
tlib  = normalize_window(rebin_log(lib; velscale=velscale(mwave), λmin=3600, λmax=7400))
lsf   = GaussianLSF(spec, mwave)
grid  = kinematics_grid(tlib.wave, mwave)
star  = StellarComponent(grid, match_resolution(tlib, mwave, lsf); σ_max=600, vmax=1000)
model = SpectralModel(grid, star, PolynomialComponent(length(mwave), 4);
                      projection=pixel_projection(mwave, spec.wave))

fit = fit_kinematics(model, spec, [0.0, 150.0])    # θ = (v, σ) in km/s; native pixels
fit.θ, reduced_chi2(fit)
```

A complete, runnable example with simulated data (no FSPS needed) is in
[`examples/quickstart.jl`](examples/quickstart.jl):

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/quickstart.jl
```

## Documentation

The full documentation is in [`docs/src`](docs/src). It covers a tutorial, the
conventions and maths, a guide to each part of the package, troubleshooting, and the API.
To build the HTML site:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```
