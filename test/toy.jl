# Small synthetic libraries shared by the template and kinematics tests.
function toy_library(; nw=50, na=4, nz=3, fwhm=2.5)
    wave = collect(range(4000.0, 5000.0; length=nw))
    flux = [(1.0 + a) * (1.0 + 0.1z) * (1.0 + 1e-3w) for w in 1:nw, a in 1:na, z in 1:nz]
    log_age = collect(range(8.0, 10.0; length=na))
    logzsol = collect(range(-1.0, 0.0; length=nz))
    mass_frac = [0.5 + 0.1a - 0.05z for a in 1:na, z in 1:nz]
    return SSPLibrary(wave, flux, log_age, logzsol, mass_frac; lsf_fwhm=fill(fwhm, nw))
end

# Wavelength-independent SSPs: rebinning must leave them unchanged.
function flat_library(; nw=500, na=3, nz=2)
    wave = collect(range(4000.0, 5000.0; length=nw))
    flux = [float(a * z) for _ in 1:nw, a in 1:na, z in 1:nz]
    return SSPLibrary(wave, flux, collect(8.0:(7.0 + na)), collect(-1.0:(-2.0 + nz)),
                      fill(0.6, na, nz))
end

# Log-uniform library with a template resolution, for resolution matching.
function log_library(; nw=600, vs=50.0, na=2, nz=2, fwhm=2.5)
    wave = 4000.0 .* exp.((0:nw-1) .* vs ./ C_KMS)
    flux = [1.0 + 0.01a * z + 0.001w for w in 1:nw, a in 1:na, z in 1:nz]
    return SSPLibrary(wave, flux, collect(8.0:(7.0 + na)), collect(-1.0:(-2.0 + nz)),
                      fill(0.6, na, nz); lsf_fwhm=fill(fwhm, nw))
end

# Population variance of a profile sampled at 0, 1, 2, ...
function profile_variance(y)
    x = 0:length(y)-1
    w = y ./ sum(y)
    μ = sum(x .* w)
    return sum((x .- μ) .^ 2 .* w)
end
