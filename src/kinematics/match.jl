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
    nsharper = count(<(0), d2)
    if strict && nsharper > 0
        throw(ArgumentError("templates are broader than the data in $nsharper pixels"))
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
    smoothing = banded_gaussian(σ_diff ./ velscale)
    return apply_banded(smoothing, T)
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
