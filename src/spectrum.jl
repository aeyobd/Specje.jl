"Speed of light [km/s]."
const C_KMS = 299_792.458

# ---------------------------------------------------------------- validation
"""
    check_lengths(n, flux, noise, good, lsf)

Throw an `ArgumentError` unless `flux`, `noise`, `good` and (if given) `lsf` all have
length `n`, the number of wavelength pixels.
"""
function check_lengths(n::Integer, flux, noise, good, lsf)
    if length(flux) != n
        throw(ArgumentError("flux length ≠ wave length"))
    end
    if length(noise) != n
        throw(ArgumentError("noise length ≠ wave length"))
    end
    if length(good) != n
        throw(ArgumentError("mask length ≠ wave length"))
    end
    if !isnothing(lsf) && length(lsf) != n
        throw(ArgumentError("lsf_fwhm length ≠ wave length"))
    end
    return nothing
end


"""
    check_wave(wave)

Throw an `ArgumentError` unless `wave` is strictly increasing.
"""
function check_wave(wave::AbstractVector{<:Real})
    if !all(>(0), diff(wave))
        throw(ArgumentError("wave must be strictly increasing"))
    end
    return nothing
end


"""
    check_noise(noise, good)

Throw an `ArgumentError` unless the noise is positive on every good pixel.
"""
function check_noise(noise::AbstractVector{<:Real}, good::AbstractVector{Bool})
    if !all(>(0), view(noise, good))
        throw(ArgumentError("noise must be > 0 on good pixels"))
    end
    return nothing
end


# ---------------------------------------------------------------- type
"""
    Spectrum(wave, flux, noise; good=trues(length(wave)), lsf_fwhm=nothing)

A 1D spectrum. `wave` are pixel centres [Å], `flux`/`noise` are flux densities
per Å, `good[i]` is `true` if pixel `i` should enter the likelihood and
`lsf_fwhm` (optional) is the instrumental FWHM [Å] per pixel.
Pixels with non-finite flux/noise or `noise ≤ 0` are always flagged bad.
"""
struct Spectrum{T<:AbstractFloat}
    wave::Vector{T}
    flux::Vector{T}
    noise::Vector{T}
    good::BitVector
    lsf_fwhm::Union{Nothing,Vector{T}}

    function Spectrum{T}(wave, flux, noise, good, lsf_fwhm) where {T<:AbstractFloat}
        check_lengths(length(wave), flux, noise, good, lsf_fwhm)
        check_wave(wave)
        check_noise(noise, good)
        return new{T}(wave, flux, noise, good, lsf_fwhm)
    end
end


"""
    default_good(flux, noise)

Return the mask of pixels whose flux and noise are finite and whose noise is positive.
"""
function default_good(flux::AbstractVector{<:Real}, noise::AbstractVector{<:Real})
    finite_flux = isfinite.(flux)
    finite_noise = isfinite.(noise)
    positive_noise = noise .> 0
    return BitVector(finite_flux .& finite_noise .& positive_noise)
end


"""
    as_vec(T, x)

Convert `x` to a `Vector{T}`, passing `nothing` through unchanged.
"""
function as_vec(T::Type, x)
    return Vector{T}(x)
end

function as_vec(T::Type, ::Nothing)
    return nothing
end


function Spectrum(wave::AbstractVector{<:Real}, flux::AbstractVector{<:Real},
                  noise::AbstractVector{<:Real};
                  good=trues(length(wave)), lsf_fwhm=nothing)
    T = float(promote_type(eltype(wave), eltype(flux), eltype(noise)))
    check_lengths(length(wave), flux, noise, good, lsf_fwhm)

    ok = BitVector(good) .& default_good(flux, noise)
    return Spectrum{T}(as_vec(T, wave), as_vec(T, flux), as_vec(T, noise), ok,
                       as_vec(T, lsf_fwhm))
end


# ---------------------------------------------------------------- accessors
"""
    npix(s::Spectrum)

Return the number of pixels of the spectrum.
"""
function npix(s::Spectrum)
    return length(s.wave)
end


"""
    velscale(wave)
    velscale(s::Spectrum)

Return the mean velocity step per pixel [km/s] of a log-uniform wavelength grid.
"""
function velscale(wave::AbstractVector{<:Real})
    if length(wave) < 2
        throw(ArgumentError("velscale needs at least two wavelengths"))
    end
    if first(wave) <= 0
        throw(ArgumentError("wavelengths must be positive"))
    end

    total_dlnλ = log(last(wave) / first(wave))
    nsteps = length(wave) - 1
    return C_KMS * total_dlnλ / nsteps
end

function velscale(s::Spectrum)
    return velscale(s.wave)
end


"""
    is_loguniform(wave; rtol=1e-6)
    is_loguniform(s::Spectrum; rtol=1e-6)

Return `true` if `wave` is uniformly spaced in ln λ (to relative tolerance `rtol`).
"""
function is_loguniform(wave::AbstractVector{<:Real}; rtol::Real=1e-6)
    if length(wave) < 2
        throw(ArgumentError("is_loguniform needs at least two wavelengths"))
    end

    steps = diff(log.(wave))
    mean_step = mean(steps)
    return all(step -> isapprox(step, mean_step; rtol=rtol), steps)
end

function is_loguniform(s::Spectrum; kw...)
    return is_loguniform(s.wave; kw...)
end


# ---------------------------------------------------------------- slicing
"""
    slice(x, idx)

Return `x[idx]`, passing `nothing` through unchanged (for optional per-pixel fields).
"""
function slice(::Nothing, idx)
    return nothing
end

function slice(x::AbstractVector, idx)
    return x[idx]
end


"""
    crop(s::Spectrum, λmin, λmax)

Restrict a spectrum to the pixels with `λmin ≤ wave ≤ λmax`.
"""
function crop(s::Spectrum{T}, λmin::Real, λmax::Real) where {T}
    if λmin > λmax
        throw(ArgumentError("λmin = $λmin must not exceed λmax = $λmax"))
    end

    first_pixel = searchsortedfirst(s.wave, λmin)
    last_pixel = searchsortedlast(s.wave, λmax)
    idx = first_pixel:last_pixel
    if isempty(idx)
        throw(ArgumentError("no pixels in [$λmin, $λmax]"))
    end

    return Spectrum{T}(s.wave[idx], s.flux[idx], s.noise[idx], s.good[idx],
                       slice(s.lsf_fwhm, idx))
end
