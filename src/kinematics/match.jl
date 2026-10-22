# Degrade templates to the data's resolution before fitting kinematics, so the
# fitted σ is the intrinsic one:  σ_diff² = σ_data² − σ_templ²  (pPXF convention).

"""
    sigma_diff_kms(σ_data, σ_templ; strict=true)

Gaussian width [km/s] needed to bring templates to the data's resolution. Either
argument may be a scalar or a vector (broadcast). With `strict` an error is raised
where the data are *sharper* than the templates; otherwise those pixels are left
unsmoothed.
"""
function sigma_diff_kms(σ_data, σ_templ; strict::Bool=true)
    if any(<(0), σ_data) || any(<(0), σ_templ)
        throw(ArgumentError("resolutions must be ≥ 0"))
    end

    d2 = @. σ_data^2 - σ_templ^2
    # equal resolutions must not trip the check through round-off (e.g. FWHM ↔ σ conversions)
    tolerance = @. 1e-9 * max(σ_data^2, σ_templ^2)
    nsharper = count(d2 .< -tolerance)
    if strict && nsharper > 0
        throw(ArgumentError("templates are broader than the data in $nsharper pixels; " *
                            "use match_resolution_auto to fit with the deficit folded into the LOSVD"))
    end

    return @. sqrt(max(d2, 0))
end


"""
    sigma_on_grid(σ, from_wave, to_wave)

Linearly interpolate `σ(from_wave)` onto `to_wave` (clamped at the ends).
"""
function sigma_on_grid(σ::AbstractVector{<:Real}, from_wave::AbstractVector{<:Real},
                       to_wave::AbstractVector{<:Real})
    if length(σ) != length(from_wave)
        throw(ArgumentError("σ has $(length(σ)) entries, its wavelength grid $(length(from_wave))"))
    end

    return [interp_linear(w, from_wave, σ) for w in to_wave]
end


"""
    check_log_template_grid(wave)

Throw an `ArgumentError` unless the template wavelengths are log-uniform; return their velscale.
"""
function check_log_template_grid(wave::AbstractVector{<:Real})
    if !is_loguniform(wave)
        throw(ArgumentError("template wavelengths must be log-uniform"))
    end
    return velscale(wave)
end


"""
    match_resolution(T, twave, tfwhm, dwave, dlsf, velscale; strict=true)

Smooth template columns `T` (on `twave`, Gaussian FWHM `tfwhm` [Å]) down to the
data resolution `dlsf::LSF` given on `dwave`. Pass `dwave` in the rest frame
(`wave / (1 + z_sys)`); σ in km/s is unaffected by that rescaling. A `BandedLSF`
enters through its effective σ (see `inst_sigma_kms`).
"""
function match_resolution(T::AbstractMatrix{<:Real}, twave::AbstractVector{<:Real}, tfwhm,
                          dwave::AbstractVector{<:Real}, dlsf::LSF, velscale::Real;
                          strict::Bool=true)
    if size(T, 1) != length(twave)
        throw(DimensionMismatch("templates have $(size(T, 1)) pixels, twave has $(length(twave))"))
    end
    if npix(dlsf) != length(dwave)
        throw(DimensionMismatch("data LSF has $(npix(dlsf)) pixels, dwave has $(length(dwave))"))
    end
    if velscale <= 0
        throw(ArgumentError("velscale must be positive"))
    end

    σ_templ = fwhm_to_sigma_kms(tfwhm, twave)
    σ_data = sigma_on_grid(inst_sigma_kms(dlsf, velscale), dwave, twave)
    σ_diff = sigma_diff_kms(σ_data, σ_templ; strict=strict)
    return apply_banded(matching_kernel(σ_diff ./ velscale), T)
end


"""
    matching_kernel(σ_pix)

Banded smoothing matrix that adds a Gaussian of width `σ_pix[i]` [pixels] to sampled
templates. `banded_gaussian` uses pixel-integrated weights, whose discrete variance is
σ² + 1/12 px², so 1/12 px² is taken off the requested variance first (widths below
√(1/12) px, where that is not possible, are left unsmoothed).
"""
function matching_kernel(σ_pix::AbstractVector{<:Real})
    if any(<(0), σ_pix)
        throw(ArgumentError("σ must be ≥ 0"))
    end

    corrected = sqrt.(max.(σ_pix .^ 2 .- 1 / 12, 0))
    return banded_gaussian(corrected)
end


"""
    match_resolution(lib::SSPLibrary, dwave, dlsf; strict=true)

Normalised, resolution-matched template matrix for a log-uniform library with `lsf_fwhm`.
"""
function match_resolution(lib::SSPLibrary, dwave::AbstractVector{<:Real}, dlsf::LSF;
                          strict::Bool=true)
    if isnothing(lib.lsf_fwhm)
        throw(ArgumentError("library has no lsf_fwhm"))
    end

    vs = check_log_template_grid(lib.wave)
    return match_resolution(templates(lib), lib.wave, lib.lsf_fwhm, dwave, dlsf, vs;
                            strict=strict)
end


# ---------------------------------------------------------------- constant velocity resolution
"""
    library_sigma_kms(lib)

Library resolution σ [km/s] per pixel, from `lsf_fwhm`.
"""
function library_sigma_kms(lib::SSPLibrary)
    if isnothing(lib.lsf_fwhm)
        throw(ArgumentError("library has no lsf_fwhm"))
    end

    return fwhm_to_sigma_kms(lib.lsf_fwhm, lib.wave)
end


"""
    match_resolution(lib::SSPLibrary, σ_kms::Real; strict=true)

Normalised template matrix broadened to a *constant* velocity resolution `σ_kms` at every
wavelength (the library's own resolution is subtracted in quadrature). Pair it with
`StellarComponent(...; σ_templ=σ_kms)` when a resolution matrix in the model's
projection applies the instrument. Choose `σ_kms ≥ maximum(library_sigma_kms(lib))`.
"""
function match_resolution(lib::SSPLibrary, σ_kms::Real; strict::Bool=true)
    if σ_kms <= 0
        throw(ArgumentError("σ_kms must be positive"))
    end

    constant = GaussianLSF(fill(float(σ_kms), nwave(lib)))
    return match_resolution(lib, lib.wave, constant; strict=strict)
end


# ---------------------------------------------------------------- templates broader than the data
# When the library is broader than the data somewhere, the deficit
#     d(λ) = √(σ_lib² − σ_data²)
# cannot be removed by smoothing. It can be folded into the LOSVD instead: bring the
# templates to √(σ_data² + σ_c²) with one constant σ_c ≥ max d, and convolve with a
# kernel of width √(σ² − σ_c²) (`StellarComponent(...; σ_templ=σ_c)`). The fit is then
# exact for any galaxy with σ ≥ σ_c.

"""
    data_sigma_on_library(lib, dwave, dlsf)

The data's instrumental σ [km/s] on the library's wavelengths (zero without an LSF),
and the mask of library pixels inside the data's wavelength range.
"""
function data_sigma_on_library(lib::SSPLibrary, dwave::AbstractVector{<:Real}, dlsf)
    vs = check_log_template_grid(lib.wave)
    if isnothing(dlsf)
        σ_data = zeros(nwave(lib))
    else
        if npix(dlsf) != length(dwave)
            throw(DimensionMismatch("data LSF has $(npix(dlsf)) pixels, dwave has $(length(dwave))"))
        end
        σ_data = sigma_on_grid(inst_sigma_kms(dlsf, vs), dwave, lib.wave)
    end

    inside = first(dwave) .<= lib.wave .<= last(dwave)
    if !any(inside)
        throw(ArgumentError("the library does not overlap the data wavelengths"))
    end
    return σ_data, inside
end


"""
    resolution_deficit_kms(lib, dwave, dlsf=nothing) -> (wave=…, deficit=…)

The deficit `√(max(σ_lib² − σ_data², 0))` [km/s] at each library wavelength inside the
data's range: where, and by how much, the library is broader than the data. Its
maximum is `minimum_sigma_templ`; look here to see which wavelengths set it.
"""
function resolution_deficit_kms(lib::SSPLibrary, dwave::AbstractVector{<:Real}, dlsf=nothing)
    σ_lib = library_sigma_kms(lib)
    σ_data, inside = data_sigma_on_library(lib, dwave, dlsf)

    deficit_squared = σ_lib[inside] .^ 2 .- σ_data[inside] .^ 2
    deficit = sqrt.(max.(deficit_squared, 0.0))
    return (wave=lib.wave[inside], deficit=deficit)
end


"""
    minimum_sigma_templ(lib, dwave, dlsf=nothing)

Smallest constant σ_c [km/s] for which `lib` can be matched to data with LSF `dlsf`
(on `dwave`): the largest deficit `√(σ_lib² − σ_data²)` over the library pixels inside
the data's wavelength range, or 0 if the library is sharper everywhere. Without `dlsf`
the instrument is taken to be applied later (a resolution matrix in the projection),
so the deficit is the library's full resolution.
"""
function minimum_sigma_templ(lib::SSPLibrary, dwave::AbstractVector{<:Real}, dlsf=nothing)
    deficit = resolution_deficit_kms(lib, dwave, dlsf).deficit
    return maximum(deficit)
end


"""
    match_resolution_auto(lib, dwave, dlsf=nothing; σ_templ=minimum_sigma_templ(lib, dwave, dlsf))
        -> (templates=T, σ_templ=σ_c)

Resolution-match a library that may be *broader* than the data. The templates are
brought to `√(σ_data² + σ_c²)` at every wavelength, where σ_c = `σ_templ` (by default
the smallest value that works, `minimum_sigma_templ`). Use the result as

    m = match_resolution_auto(tlib, mwave, lsf)
    star = StellarComponent(grid, m.templates; σ_templ=m.σ_templ)

and the fitted σ is the galaxy's intrinsic dispersion, for any σ ≥ σ_c. If the library
is sharper than the data everywhere, σ_c = 0 and this equals `match_resolution`.

Without `dlsf` no instrument is matched (use this when a resolution matrix in the
model's projection applies it): the templates are brought to the constant σ_c.
"""
function match_resolution_auto(lib::SSPLibrary, dwave::AbstractVector{<:Real}, dlsf=nothing;
                               σ_templ::Union{Nothing,Real}=nothing)
    σ_minimum = minimum_sigma_templ(lib, dwave, dlsf)
    if isnothing(σ_templ)
        σ_c = σ_minimum
    else
        σ_c = float(σ_templ)
        if σ_c < σ_minimum * (1 - 1e-9)
            throw(ArgumentError("σ_templ = $σ_c km/s is below the library's deficit of $(round(σ_minimum; sigdigits=4)) km/s"))
        end
    end

    vs = check_log_template_grid(lib.wave)
    σ_lib = library_sigma_kms(lib)
    σ_data, _ = data_sigma_on_library(lib, dwave, dlsf)
    σ_target = sqrt.(σ_data .^ 2 .+ σ_c^2)
    # Inside the data range σ_target ≥ σ_lib by construction; outside it (the template
    # margin) a shortfall is left unsmoothed, which only affects the outermost pixels.
    σ_diff = sigma_diff_kms(σ_target, σ_lib; strict=false)

    T = apply_banded(matching_kernel(σ_diff ./ vs), templates(lib))
    return (templates=T, σ_templ=σ_c)
end
