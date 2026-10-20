# Log-λ resampling of a whole library, reusing the Spectrum rebinning primitives.

"""
    check_finite(flux)

Throw an `ArgumentError` if the rebinned library flux contains non-finite values,
which happens when the requested range is not covered by the library.
"""
function check_finite(flux::AbstractArray)
    if !all(isfinite, flux)
        throw(ArgumentError("rebinned library has non-finite flux; check wavelength coverage"))
    end
    return nothing
end


"""
    rebin_cube(new_edges, old_edges, flux)

Rebin every SSP of a flux cube `(nwave, nage, nmet)` onto `new_edges`.
"""
function rebin_cube(ne::AbstractVector{<:Real}, e::AbstractVector{<:Real},
                    flux::AbstractArray{<:Real,3})
    nw, na, nz = size(flux)
    flat = reshape(flux, nw, na * nz)
    rebinned = rebin_flux(ne, e, flat)
    return reshape(rebinned, :, na, nz)
end


"""
    rebin_log(lib::SSPLibrary; velscale, λmin=first(wave), λmax=last(wave))

Crop to `[λmin, λmax]` and resample every SSP onto a log-uniform grid with
`velscale` km/s per pixel (flux-conserving). Use the same velscale as the model grid
(or the rebinned data).
"""
function rebin_log(lib::SSPLibrary; velscale::Real,
                   λmin::Real=first(lib.wave), λmax::Real=last(lib.wave))
    if velscale <= 0
        throw(ArgumentError("velscale must be positive"))
    end

    sub = crop_wave(lib, λmin, λmax)
    e = pixel_edges(sub.wave)
    grid = loggrid_within(e, velscale)
    flux = rebin_cube(log_pixel_edges(grid), e, sub.flux)
    check_finite(flux)

    lsf = rebin_lsf(sub.lsf_fwhm, grid, sub.wave)
    return SSPLibrary(grid, flux, sub.log_age, sub.logzsol, sub.mass_frac;
                      norm=sub.norm, lsf_fwhm=lsf)
end
