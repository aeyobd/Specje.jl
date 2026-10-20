# SSP library: flux cube (nwave, nage, nmet) plus the grid it lives on.
# Flat template ordering is always "age fastest, metallicity slowest".

# ---------------------------------------------------------------- validation
"""
    check_library_shapes(wave, flux, log_age, logzsol, mass_frac, norm)

Throw an `ArgumentError` unless `flux` is `(nwave, nage, nmet)` and `mass_frac` and
`norm` are both `(nage, nmet)`.
"""
function check_library_shapes(wave, flux, log_age, logzsol, mass_frac, norm)
    expected_cube = (length(wave), length(log_age), length(logzsol))
    if size(flux) != expected_cube
        throw(ArgumentError("flux must have size (nwave, nage, nmet)"))
    end

    expected_grid = (length(log_age), length(logzsol))
    if size(mass_frac) != expected_grid || size(norm) != expected_grid
        throw(ArgumentError("mass_frac and norm must have size (nage, nmet)"))
    end

    if !all(isfinite, flux)
        throw(ArgumentError("library flux must be finite (found $(count(!isfinite, flux)) NaN/Inf values)"))
    end
    return nothing
end


"""
    check_increasing(x, name)

Throw an `ArgumentError` unless `x` is strictly increasing.
"""
function check_increasing(x::AbstractVector{<:Real}, name::AbstractString)
    if !all(>(0), diff(x))
        throw(ArgumentError("$name must be strictly increasing"))
    end
    return nothing
end


"""
    check_norm(norm)

Throw an `ArgumentError` unless every template normalisation is positive.
"""
function check_norm(norm::AbstractMatrix{<:Real})
    if !all(>(0), norm)
        throw(ArgumentError("norm must be positive"))
    end
    return nothing
end


"""
    check_lsf_length(lsf, n)

Throw an `ArgumentError` unless the optional per-pixel `lsf` has `n` entries.
"""
function check_lsf_length(::Nothing, n::Integer)
    return nothing
end

function check_lsf_length(lsf::AbstractVector, n::Integer)
    if length(lsf) != n
        throw(ArgumentError("lsf_fwhm length ≠ wave length"))
    end
    return nothing
end


# ---------------------------------------------------------------- type
"""
    SSPLibrary(wave, flux, log_age, logzsol, mass_frac; norm=ones, lsf_fwhm=nothing)

Simple stellar populations on an (age, metallicity) grid.

- `wave`      pixel centres [Å] (vacuum, as FSPS provides)
- `flux`      (nwave, nage, nmet), L☉/Å per M☉ of mass *formed*
- `log_age`   log10(age/yr), ascending
- `logzsol`   log10(Z/Z☉), ascending
- `mass_frac` (nage, nmet), surviving stellar mass (incl. remnants) per mass formed
- `norm`      (nage, nmet), per-template scale; fitted templates are `flux ./ norm`
- `lsf_fwhm`  optional per-pixel template resolution, FWHM [Å] (Gaussian)
"""
struct SSPLibrary{T<:AbstractFloat}
    wave::Vector{T}
    flux::Array{T,3}
    log_age::Vector{T}
    logzsol::Vector{T}
    mass_frac::Matrix{T}
    norm::Matrix{T}
    lsf_fwhm::Union{Nothing,Vector{T}}

    function SSPLibrary{T}(wave, flux, log_age, logzsol, mass_frac, norm,
                           lsf_fwhm) where {T<:AbstractFloat}
        check_library_shapes(wave, flux, log_age, logzsol, mass_frac, norm)
        check_lsf_length(lsf_fwhm, length(wave))
        check_wave(wave)
        check_increasing(log_age, "log_age")
        check_increasing(logzsol, "logzsol")
        check_norm(norm)
        return new{T}(wave, flux, log_age, logzsol, mass_frac, norm, lsf_fwhm)
    end
end


function SSPLibrary(wave::AbstractVector{<:Real}, flux::AbstractArray{<:Real,3},
                    log_age::AbstractVector{<:Real}, logzsol::AbstractVector{<:Real},
                    mass_frac::AbstractMatrix{<:Real};
                    norm::AbstractMatrix{<:Real}=ones(size(mass_frac)), lsf_fwhm=nothing)
    T = float(promote_type(eltype(wave), eltype(flux), eltype(log_age),
                           eltype(logzsol), eltype(mass_frac), eltype(norm)))
    return SSPLibrary{T}(Vector{T}(wave), Array{T,3}(flux), Vector{T}(log_age),
                         Vector{T}(logzsol), Matrix{T}(mass_frac), Matrix{T}(norm),
                         as_vec(T, lsf_fwhm))
end


# ---------------------------------------------------------------- accessors
"""
    nwave(lib)

Return the number of wavelength pixels of the library.
"""
function nwave(lib::SSPLibrary)
    return length(lib.wave)
end


"""
    nage(lib)

Return the number of ages in the library.
"""
function nage(lib::SSPLibrary)
    return length(lib.log_age)
end


"""
    nmet(lib)

Return the number of metallicities in the library.
"""
function nmet(lib::SSPLibrary)
    return length(lib.logzsol)
end


"""
    ntemplates(lib)

Return the number of SSP templates, `nage × nmet`.
"""
function ntemplates(lib::SSPLibrary)
    return nage(lib) * nmet(lib)
end


"""
    age_gyr(lib)

Return the library's ages in Gyr.
"""
function age_gyr(lib::SSPLibrary)
    return 10 .^ (lib.log_age .- 9)
end


# ---------------------------------------------------------------- selection
"""
    in_range(x, r)

`true` if `first(r) ≤ x ≤ last(r)`.
"""
function in_range(x::Real, r)
    return first(r) <= x <= last(r)
end


"""
    subset_library(lib; log_age=(-Inf, Inf), logzsol=(-Inf, Inf))

Keep the SSPs whose `log_age` and `logzsol` lie inside the given `(lo, hi)` ranges.
"""
function subset_library(lib::SSPLibrary; log_age=(-Inf, Inf), logzsol=(-Inf, Inf))
    if first(log_age) > last(log_age) || first(logzsol) > last(logzsol)
        throw(ArgumentError("ranges must be given as (lo, hi) with lo ≤ hi"))
    end

    ia = findall(a -> in_range(a, log_age), lib.log_age)
    iz = findall(z -> in_range(z, logzsol), lib.logzsol)
    if isempty(ia) || isempty(iz)
        throw(ArgumentError("selection is empty"))
    end

    return SSPLibrary(lib.wave, lib.flux[:, ia, iz], lib.log_age[ia], lib.logzsol[iz],
                      lib.mass_frac[ia, iz]; norm=lib.norm[ia, iz], lsf_fwhm=lib.lsf_fwhm)
end


"""
    wave_indices(wave, λmin, λmax)

Range of indices with `λmin ≤ wave ≤ λmax` (possibly empty).
"""
function wave_indices(wave::AbstractVector{<:Real}, λmin::Real, λmax::Real)
    first_pixel = searchsortedfirst(wave, λmin)
    last_pixel = searchsortedlast(wave, λmax)
    return first_pixel:last_pixel
end


"""
    crop_wave(lib, λmin, λmax)

Keep the pixels with `λmin ≤ wave ≤ λmax`.
"""
function crop_wave(lib::SSPLibrary, λmin::Real, λmax::Real)
    if λmin > λmax
        throw(ArgumentError("λmin = $λmin must not exceed λmax = $λmax"))
    end

    idx = wave_indices(lib.wave, λmin, λmax)
    if isempty(idx)
        throw(ArgumentError("no pixels in [$λmin, $λmax]"))
    end

    return SSPLibrary(lib.wave[idx], lib.flux[idx, :, :], lib.log_age, lib.logzsol,
                      lib.mass_frac; norm=lib.norm, lsf_fwhm=slice(lib.lsf_fwhm, idx))
end


# ---------------------------------------------------------------- normalisation
"""
    window_mean(lib, window)

Mean flux of every SSP over the wavelength window `(lo, hi)`, as an `(nage, nmet)` matrix.
"""
function window_mean(lib::SSPLibrary, window)
    idx = wave_indices(lib.wave, first(window), last(window))
    if isempty(idx)
        throw(ArgumentError("normalisation window outside wavelength range"))
    end

    in_window = view(lib.flux, idx, :, :)
    means = mean(in_window; dims=1)
    return dropdims(means; dims=1)
end


"""
    normalize_window(lib; window=(5300, 5700))

Set `norm` to each SSP's mean flux in `window`, so fitted templates are ≈1 there
and weights become O(1) light fractions. Replaces any existing `norm`.
"""
function normalize_window(lib::SSPLibrary; window=(5300.0, 5700.0))
    if first(window) >= last(window)
        throw(ArgumentError("window must be (lo, hi) with lo < hi"))
    end

    norm = window_mean(lib, window)
    return SSPLibrary(lib.wave, lib.flux, lib.log_age, lib.logzsol, lib.mass_frac;
                      norm=norm, lsf_fwhm=lib.lsf_fwhm)
end


# ---------------------------------------------------------------- flat templates
"""
    templates(lib)

Return the template matrix `(nwave, nage*nmet)` of normalised SSPs (`flux ./ norm`),
age index fastest.
"""
function templates(lib::SSPLibrary)
    flat_flux = reshape(lib.flux, nwave(lib), :)
    flat_norm = vec(lib.norm)
    return flat_flux ./ flat_norm'
end


"""
    template_params(lib)

Return `(log_age=…, logzsol=…)`, one entry per column of `templates(lib)`.
"""
function template_params(lib::SSPLibrary)
    log_age = repeat(lib.log_age, nmet(lib))
    logzsol = repeat(lib.logzsol; inner=nage(lib))
    return (log_age=log_age, logzsol=logzsol)
end


"""
    reshape_weights(lib, w)

Reshape a flat weight vector (one per template) to `(nage, nmet)`.
"""
function reshape_weights(lib::SSPLibrary, w::AbstractVector)
    if length(w) != ntemplates(lib)
        throw(ArgumentError("expected $(ntemplates(lib)) weights, got $(length(w))"))
    end

    return reshape(w, nage(lib), nmet(lib))
end


# ---------------------------------------------------------------- masses
"""
    check_weights(lib, w)

Throw an `ArgumentError` unless the weight matrix `w` is `(nage, nmet)`.
"""
function check_weights(lib::SSPLibrary, w::AbstractMatrix)
    if size(w) != size(lib.norm)
        throw(ArgumentError("weights must have size (nage, nmet)"))
    end
    return nothing
end


"""
    formed_mass(lib, w)

Mass formed per SSP for weights `w` (`(nage, nmet)`) on normalised templates, in units
of the flux scale.
"""
function formed_mass(lib::SSPLibrary, w::AbstractMatrix)
    check_weights(lib, w)
    return w ./ lib.norm
end


"""
    stellar_mass(lib, w)

Surviving stellar mass per SSP: formed mass × `mass_frac`.
"""
function stellar_mass(lib::SSPLibrary, w::AbstractMatrix)
    return formed_mass(lib, w) .* lib.mass_frac
end
