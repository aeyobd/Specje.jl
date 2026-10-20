# Photon-counting bandpasses. Band-averaged flux density:
#   ⟨F⟩ = ∫ F(λ) T(λ) λ dλ / ∫ T(λ) λ dλ        (F per Å, same units as the spectrum)

"AB zero point conversion constant: c in Å/s."
const C_ANGSTROM_S = 2.99792458e18


"""
    trapz(x, y)

Trapezoid-rule integral of `y(x)`.
"""
function trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    if length(x) != length(y)
        throw(ArgumentError("x and y must have equal length"))
    end

    total = zero(float(promote_type(eltype(x), eltype(y))))
    for i in 1:length(x)-1
        width = x[i+1] - x[i]
        mean_height = (y[i] + y[i+1]) / 2
        total += width * mean_height
    end
    return total
end


"""
    Filter(name, wave, trans)

Bandpass response `trans ≥ 0` on ascending `wave` [Å]. `norm = ∫ T λ dλ` is
precomputed. Transmission is zero outside the tabulated range, so trim long
zero tails before constructing.
"""
struct Filter{T<:AbstractFloat}
    name::String
    wave::Vector{T}
    trans::Vector{T}
    norm::T
end


function Filter(name::AbstractString, wave::AbstractVector{<:Real}, trans::AbstractVector{<:Real})
    if length(wave) != length(trans)
        throw(ArgumentError("wave and trans must have equal length"))
    end
    if length(wave) < 2
        throw(ArgumentError("a filter needs at least two points"))
    end
    check_wave(wave)
    if !all(>=(0), trans)
        throw(ArgumentError("transmission must be ≥ 0"))
    end

    T = float(promote_type(eltype(wave), eltype(trans)))
    w = Vector{T}(wave)
    t = Vector{T}(trans)
    norm = trapz(w, t .* w)
    if norm <= 0
        throw(ArgumentError("filter '$name' has no throughput"))
    end
    return Filter{T}(String(name), w, t, norm)
end


"""
    transmission(f, λ)

Linearly interpolated transmission at `λ` (zero outside the tabulated range).
"""
function transmission(f::Filter, λ::Real)
    if λ <= first(f.wave) || λ >= last(f.wave)
        return zero(float(λ))
    end
    return interp_linear(λ, f.wave, f.trans)
end


"""
    pivot_wavelength(f)

Pivot wavelength [Å]: λ_p² = ∫Tλ dλ / ∫T/λ dλ.
"""
function pivot_wavelength(f::Filter)
    integral_over_λ = trapz(f.wave, f.trans ./ f.wave)
    return sqrt(f.norm / integral_over_λ)
end


"""
    to_rest(f, z)

The filter as seen from the rest frame of a source at redshift `z`: wavelengths are
divided by 1 + z, transmission is unchanged. Band averages of a spectrum labelled
with rest wavelengths (as in this package) then match the observed ones.
"""
function to_rest(f::Filter, z::Real)
    if z <= -1
        throw(ArgumentError("redshift must exceed -1"))
    end

    return Filter(f.name, f.wave ./ (1 + z), f.trans)
end


"""
    gaussian_filter(name, λ0, σ; n=401, nσ=5)

Gaussian-response filter centred on `λ0` with width `σ` [Å], tabulated at `n` points
over ±`nσ` σ. Useful for tests and toy setups.
"""
function gaussian_filter(name::AbstractString, λ0::Real, σ::Real; n::Integer=401, nσ::Real=5)
    if σ <= 0
        throw(ArgumentError("σ must be positive"))
    end
    if n < 2
        throw(ArgumentError("need at least two points"))
    end
    if λ0 - nσ * σ <= 0
        throw(ArgumentError("the filter must lie at positive wavelengths"))
    end

    wave = collect(range(λ0 - nσ * σ, λ0 + nσ * σ; length=n))
    trans = @. exp(-(wave - λ0)^2 / (2σ^2))
    return Filter(name, wave, trans)
end


"""
    tophat_filter(name, a, b; n=1001)

Top-hat filter with unit transmission on `[a, b]` Å.
"""
function tophat_filter(name::AbstractString, a::Real, b::Real; n::Integer=1001)
    if !(0 < a < b)
        throw(ArgumentError("need 0 < a < b"))
    end

    wave = collect(range(a, b; length=n))
    return Filter(name, wave, ones(n))
end


# ---------------------------------------------------------------- unit conversions
# `unit` is the flux-density unit of your spectrum in erg/s/cm²/Å (e.g. 1e-17).
# Conversions use the observer-frame filter's pivot wavelength (exact for ⟨f_ν⟩ ↔ ⟨f_λ⟩).

"""
    check_unit(unit)

Throw an `ArgumentError` unless the flux-density unit is positive.
"""
function check_unit(unit::Real)
    if unit <= 0
        throw(ArgumentError("unit must be positive"))
    end
    return nothing
end


"""
    ab_to_flambda(mag, f; unit=1.0)

AB magnitude → band-averaged f_λ in units of `unit` erg/s/cm²/Å.
"""
function ab_to_flambda(mag::Real, f::Filter; unit::Real=1.0)
    check_unit(unit)

    fν = 10^(-0.4 * (mag + 48.6))                 # erg/s/cm²/Hz
    λp = pivot_wavelength(f)
    return fν * C_ANGSTROM_S / λp^2 / unit
end


"""
    flambda_to_ab(F, f; unit=1.0)

Band-averaged f_λ (in `unit`) → AB magnitude.
"""
function flambda_to_ab(F::Real, f::Filter; unit::Real=1.0)
    check_unit(unit)
    if F <= 0
        throw(ArgumentError("flux must be positive to have a magnitude"))
    end

    λp = pivot_wavelength(f)
    fν = F * unit * λp^2 / C_ANGSTROM_S
    return -2.5 * log10(fν) - 48.6
end


"""
    jy_to_flambda(jy, f; unit=1.0)

Flux density in Jy → band-averaged f_λ in units of `unit` erg/s/cm²/Å.
"""
function jy_to_flambda(jy::Real, f::Filter; unit::Real=1.0)
    check_unit(unit)

    fν = jy * 1e-23                               # erg/s/cm²/Hz
    λp = pivot_wavelength(f)
    return fν * C_ANGSTROM_S / λp^2 / unit
end


"""
    mag_err_to_flux_err(F, dmag)

1σ flux error from a magnitude error `dmag` at flux `F`.
"""
function mag_err_to_flux_err(F::Real, dmag::Real)
    if dmag < 0
        throw(ArgumentError("magnitude error must be ≥ 0"))
    end

    return F * dmag * log(10) / 2.5
end
