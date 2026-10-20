# Band fluxes of stellar templates. Broadband fluxes do not depend on the LOSVD
# (σ ~ 10² km/s is negligible against filter widths), so they are computed once from a
# wide-wavelength copy of the SSP library and cached as a constant matrix.

"""
    trapz_weights(x)

Trapezoid-rule quadrature weights for abscissae `x`, so `sum(trapz_weights(x) .* y)`
integrates `y(x)`.
"""
function trapz_weights(x::AbstractVector{<:Real})
    n = length(x)
    if n < 2
        throw(ArgumentError("need at least two abscissae"))
    end

    w = similar(x, float(eltype(x)))
    w[1] = (x[2] - x[1]) / 2
    w[n] = (x[n] - x[n-1]) / 2
    for i in 2:n-1
        w[i] = (x[i+1] - x[i-1]) / 2
    end
    return w
end


"""
    check_filter_coverage(filters, wave)

Throw an `ArgumentError` unless every filter lies inside the wavelength range `wave`.
"""
function check_filter_coverage(filters::AbstractVector{<:Filter}, wave::AbstractVector{<:Real})
    for f in filters
        inside = first(f.wave) >= first(wave) && last(f.wave) <= last(wave)
        if !inside
            throw(ArgumentError("filter '$(f.name)' extends beyond the library's wavelength range"))
        end
    end
    return nothing
end


"""
    bandpass_weights(filters, wave)

Matrix `W (nfilt × nwave)` such that `W * F` is the band-averaged flux density of a
spectrum `F` sampled on `wave`. Filters must lie inside `wave`.
"""
function bandpass_weights(filters::AbstractVector{<:Filter}, wave::AbstractVector{<:Real})
    check_wave(wave)
    check_filter_coverage(filters, wave)

    dw = trapz_weights(wave)
    W = zeros(float(eltype(wave)), length(filters), length(wave))
    for (j, f) in enumerate(filters)
        for (i, λ) in enumerate(wave)
            W[j, i] = transmission(f, λ) * λ * dw[i] / f.norm
        end
    end
    return W
end


"""
    BandTemplates(names, M)

Band fluxes `M (nfilt × ntemplates)` of every template, tagged with the filter names
they belong to. Build with `band_templates`.
"""
struct BandTemplates{T<:AbstractFloat}
    names::Vector{String}
    M::Matrix{T}

    function BandTemplates(names::Vector{String}, M::Matrix{T}) where {T<:AbstractFloat}
        if length(names) != size(M, 1)
            throw(ArgumentError("need one filter name per row of M"))
        end
        return new{T}(names, M)
    end
end


"""
    convert_bands(T, bands)

Convert band templates (or `nothing`) to element type `T`.
"""
function convert_bands(T::Type, ::Nothing)
    return nothing
end

function convert_bands(T::Type, b::BandTemplates)
    return BandTemplates(b.names, Matrix{T}(b.M))
end


"""
    check_band_columns(bands, n)

Throw an `ArgumentError` unless the band templates (if any) have `n` columns, one per template.
"""
function check_band_columns(::Nothing, n::Integer)
    return nothing
end

function check_band_columns(b::BandTemplates, n::Integer)
    if size(b.M, 2) != n
        throw(ArgumentError("band templates have $(size(b.M, 2)) columns, expected $n"))
    end
    return nothing
end


"""
    with_norm(lib, norm)

Copy of the library with the template normalisation replaced by `norm`.
"""
function with_norm(lib::SSPLibrary, norm::AbstractMatrix{<:Real})
    return SSPLibrary(lib.wave, lib.flux, lib.log_age, lib.logzsol, lib.mass_frac;
                      norm=norm, lsf_fwhm=lib.lsf_fwhm)
end


"""
    check_same_grid(a, b)

Throw an `ArgumentError` unless two libraries share the same (age, metallicity) grid.
"""
function check_same_grid(a::SSPLibrary, b::SSPLibrary)
    same_shape = nage(a) == nage(b) && nmet(a) == nmet(b)
    if !same_shape || !(a.log_age ≈ b.log_age) || !(a.logzsol ≈ b.logzsol)
        throw(ArgumentError("wide and narrow libraries must share the same (age, metallicity) grid"))
    end
    return nothing
end


"""
    band_templates(wide, narrow, phot) -> BandTemplates

Band fluxes of the templates in `narrow` (the library used for the spectral fit),
computed from `wide`, the same SSPs over a wider wavelength range (e.g. exported with
`--wmin 1000 --wmax 30000`). The two must share the (age, metallicity) grid; `narrow`'s
`norm` is applied so weights mean the same thing in the spectral and photometric rows.
"""
function band_templates(wide::SSPLibrary, narrow::SSPLibrary, phot::Photometry)
    check_same_grid(wide, narrow)

    W = bandpass_weights(phot.rest, wide.wave)
    wide_templates = templates(with_norm(wide, narrow.norm))
    return BandTemplates(band_names(phot), W * wide_templates)
end
