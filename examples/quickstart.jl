# Specje quickstart: simulate a galaxy spectrum from a toy SSP library, then recover its
# kinematics and stellar population with a χ² fit and a Bayesian posterior.
#
#     julia --project=examples examples/quickstart.jl
#
# Replace section 1 with `lib = read_fsps_hdf5("fsps_ssp.h5")` and section 2 with your
# own spectrum to run it on real data; sections 3 onwards are unchanged.

using Specje, Random, Statistics, LinearAlgebra, LogDensityProblems

rng = Xoshiro(42)

# ---------------------------------------------------------------- 1. SSP library
# Toy stand-in for an FSPS export: 4 ages × 2 metallicities on a *linear* 0.5 Å grid
# (FSPS libraries are not log-uniform), with absorption lines and coloured continua.
wave = collect(3600.0:0.5:5800.0)
log_age, logzsol = [8.5, 9.0, 9.5, 10.0], [-0.5, 0.0]
line_centres = [3934.8, 3969.6, 4102.9, 4305.0, 4341.7, 4383.5, 4862.7, 5176.7, 5270.0, 5328.0]
flux = Array{Float64}(undef, length(wave), length(log_age), length(logzsol))
for ia in eachindex(log_age), iz in eachindex(logzsol)
    f = (wave ./ 4500.0) .^ (-2.5 + 0.5ia + 0.3iz)
    for λ in line_centres
        f .*= 1 .- (0.1 + 0.05ia + 0.1iz) .* exp.(-(wave .- λ) .^ 2 ./ (2 * 1.2^2))
    end
    flux[:, ia, iz] = f
end
mass_frac = [0.75 - 0.05ia for ia in eachindex(log_age), iz in eachindex(logzsol)]
lib = SSPLibrary(wave, flux, log_age, logzsol, mass_frac; lsf_fwhm=fill(1.0, length(wave)))

# ---------------------------------------------------------------- 2. an "observed" spectrum
# Truth: v = 80 km/s, σ = 180 km/s, two populations, observed at z = 0.02 with a
# 2.5 Å (rest-frame) instrumental FWHM, 2% noise, on a linear observed-frame grid.
z_sys = 0.02
θ_true = [80.0, 180.0]
obs_wave = collect(range(4100.0, 5300.0; length=2400)) .* (1 + z_sys)
obs_fwhm = fill(2.5 * (1 + z_sys), length(obs_wave))
observed = let
    sim = normalize_window(rebin_log(lib; velscale=20.0); window=(4500.0, 4600.0))
    Tsim = match_resolution(sim, sim.wave, GaussianLSF(sim.wave, 2.5))
    inner = sim.wave[300:end-300]
    g = kinematics_grid(sim.wave, inner)
    w_true = zeros(ntemplates(sim)); w_true[2] = 1.0; w_true[7] = 0.6
    model_flux = convolve_templates(g, transform_templates(g, Tsim), θ_true...) * w_true
    rest_wave = obs_wave ./ (1 + z_sys)
    f = [Specje.interp_linear(λ, inner, model_flux) for λ in rest_wave]
    σn = 0.02 * mean(f)
    Spectrum(obs_wave, f .+ σn .* randn(rng, length(f)), fill(σn, length(f)); lsf_fwhm=obs_fwhm)
end

# ---------------------------------------------------------------- 3. prepare the data
# Work in the rest frame: divide wavelengths (and FWHM in Å) by 1 + z. The data stay on
# their native pixels, so their noise stays independent; the model is evaluated on a
# finer log grid and averaged onto those pixels by a projection matrix.
rest = Spectrum(observed.wave ./ (1 + z_sys), observed.flux, observed.noise;
                good=observed.good, lsf_fwhm=observed.lsf_fwhm ./ (1 + z_sys))
mwave = model_grid(rest.wave; oversample=2)               # log grid covering the pixels
P = pixel_projection(mwave, rest.wave)                     # (native pixels, model pixels)
vs = velscale(mwave)
println("data: $(npix(rest)) native pixels; model grid: $(length(mwave)) pixels at ",
        "$(round(vs; digits=2)) km/s")

# ---------------------------------------------------------------- 4. prepare the templates
# Same velscale as the model grid; wider in wavelength than it (kinematics_grid margin).
tlib = rebin_log(lib; velscale=vs, λmin=3800.0, λmax=5600.0)
tlib = normalize_window(tlib; window=(4500.0, 4600.0))  # weights ≈ light fractions there
lsf = GaussianLSF(rest, mwave)                           # data LSF on the model grid
T = match_resolution(tlib, mwave, lsf)                   # templates at the data resolution
grid = kinematics_grid(tlib.wave, mwave)                 # FFT geometry (margin 2500 km/s)

# ---------------------------------------------------------------- 5. the model
# Components live on the model grid; `projection` maps them to the native pixels.
star = StellarComponent(grid, T; σ_min=10.0, σ_max=600.0, vmax=1000.0)
gas = GasComponent(mwave,
                   [GasGroup("balmer", Multiplet("Hbeta", 4862.683), Multiplet("Hgamma", 4341.692))],
                   lsf; σ_min=5.0, vmax=800.0)
poly = PolynomialComponent(length(mwave), 2)             # additive Legendre, degree 2
model = SpectralModel(grid, star, gas, poly; projection=P)   # θ = (v★, σ★, v_gas, σ_gas)
println("parameters: $(nparams(model)), design-matrix columns: $(ncolumns(model))")

# ---------------------------------------------------------------- 6. χ² fit
fit = fit_kinematics(model, rest, [0.0, 150.0, 0.0, 100.0])
println("best fit θ = ", round.(fit.θ; digits=1), ",  χ²/N = ", round(reduced_chi2(fit); digits=3))
w_star, w_gas, w_poly = split_weights(model, fit.weights)
W = reshape_weights(tlib, w_star)                        # (nage, nmet)
println("light weights (age × [Z/H]):"); display(round.(W; digits=3))
println("surviving stellar mass: ", round(sum(stellar_mass(tlib, W)); digits=3))

# ---------------------------------------------------------------- 7. posterior
fd = FitData(rest)
bm = BayesModel(model, fd, fit.θ; fit_noise=true)        # weights marginalised analytically
u_map = map_estimate(bm, initial_free(bm, fit.θ)).u
println("MAP θ = ", round.(to_physical(bm, u_map).θ; digits=1))

chain = sample_ensemble(Xoshiro(1), bm, u_map; nsteps=600)
println("acceptance ", round(acceptance_fraction(chain); digits=2),
        ", autocorrelation times ", round.(autocorr_time(chain); digits=1))
U = flatten_chain(chain; burn=300)
Θ = reduce(hcat, [to_physical(bm, u).θ for u in eachcol(U)])
println("posterior mean θ = ", round.(vec(mean(Θ; dims=2)); digits=1),
        ", sd = ", round.(vec(std(Θ; dims=2)); digits=1))

# Weights given one posterior draw, truncated to w ≥ 0.
wp = weight_posterior(bm, U[:, end])
draws = sample_weights(Xoshiro(2), wp, 200; positive=true)
println("weight draws: ", size(draws))

# Gradient for NUTS / HMC (see docs/src/guide/bayes.md for an AdvancedHMC example).
lp, grad = LogDensityProblems.logdensity_and_gradient(bm, u_map)
println("log posterior at MAP ", round(lp; digits=2), ", |∇| = ", round(norm(grad); sigdigits=3))
