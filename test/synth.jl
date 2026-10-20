using Random

# Absorption-line templates: continuum with random Gaussian lines and a slope.
function line_templates(rng, n, ntemp; nlines=60, σ=2.0)
    x = 1:n
    T = ones(n, ntemp)
    for k in 1:ntemp
        for _ in 1:nlines
            c, d, s = rand(rng) * n, 0.1 + 0.3rand(rng), σ * (0.7 + 0.6rand(rng))
            @. T[:, k] *= 1 - d * exp(-(x - c)^2 / (2 * s^2))
        end
        T[:, k] .*= 1 .+ (rand(rng) - 0.5) .* (x ./ n .- 0.5)
    end
    return T
end

"Template grid (1200 px) containing a data grid (400 px) with ≥ 50 px margins."
function synthetic_setup(; seed=1, ntemp=6, vs=50.0, nt=1200, crop=301:700)
    rng = Xoshiro(seed)
    twave = 4000.0 .* exp.((0:nt-1) .* vs ./ C_KMS)
    dwave = twave[crop]
    grid = kinematics_grid(twave, dwave)
    return (; rng, twave, dwave, grid, T=line_templates(rng, nt, ntemp), vs)
end

"""
Wide-wavelength SSP library (3000–9000 Å, log grid) with coloured line templates, a
narrow copy for the spectral fit, a 400-pixel data grid, and four Gaussian filters.
"""
function joint_setup(; seed=11, vs=50.0, nw=6587, ntemp=6)
    rng = Xoshiro(seed)
    wave = 3000.0 .* exp.((0:nw-1) .* vs ./ C_KMS)
    T = line_templates(rng, nw, ntemp; nlines=330)
    β = range(-2.0, 2.0; length=ntemp)
    for k in 1:ntemp
        T[:, k] .*= (wave ./ 5000.0) .^ β[k]
    end
    wide = SSPLibrary(wave, reshape(T, nw, 3, 2), [8.0, 9.0, 10.0], [-0.5, 0.0], fill(0.6, 3, 2))
    narrow = normalize_window(crop_wave(wide, 4000.0, 4900.0); window=(4400.0, 4600.0))
    dwave = narrow.wave[301:700]
    grid = kinematics_grid(narrow.wave, dwave)
    centres = (4500.0, 5700.0, 6800.0, 7500.0)
    filters = [gaussian_filter(n, λ, 200.0; n=801) for (n, λ) in zip(("B", "V", "R", "I"), centres)]
    phot0 = Photometry(filters, ones(4), ones(4))
    return (; rng, wide, narrow, dwave, grid, vs, filters, phot0,
            bands=band_templates(wide, narrow, phot0))
end

"Stellar-only joint model (with photometry) on the `joint_setup` data, plus true weights."
function joint_fixture(; fit_scale=true)
    j = joint_setup()
    ws = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    star = StellarComponent(j.grid, templates(j.narrow); σ_min=25.0, vmax=1000.0, phot=j.bands)
    stars = SpectralModel(j.grid, star)
    truth = j.bands.M * ws
    phot = Photometry(j.filters, truth, 0.02 .* truth)
    return (; j, ws, star, stars, phot, jm=JointModel(stars, phot; fit_scale=fit_scale))
end
