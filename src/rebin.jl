# Flux-conserving rebinning (the SpectRes algorithm): each new pixel is the
# overlap-weighted mean of the old pixels it covers.
# See https://arxiv.org/abs/1705.05165 for a description of the algorithm

# ---------------------------------------------------------------- overlap helpers
"""
    overlap(lo, hi, e, j)

Width of the intersection of [lo, hi] with old bin `j` (edges `e`).
"""
function overlap(lo, hi, e, j)
    return max(zero(lo), min(hi, e[j+1]) - max(lo, e[j]))
end


"""
    bin_range(lo, hi, edges)

Return the range of old bins touching [lo, hi] or `nothing` if lo or hi extend outside `edges`.
"""
function bin_range(lo, hi, edges)
    if (lo < first(edges) || hi > last(edges))
        return nothing
    end

    n = nbins(edges)
    j1 = clamp(searchsortedlast(edges, lo), 1, n)
    j2 = clamp(searchsortedfirst(edges, hi) - 1, 1, n)
    return j1:j2
end


"""
    nbins(edges)

Return the number of bins described by a vector of bin edges.
"""
function nbins(edges::AbstractVector{<:Real})
    return length(edges) - 1
end


"""
    check_edges(edges, name)

Throw an `ArgumentError` unless `edges` has at least two entries and is strictly increasing.
"""
function check_edges(edges::AbstractVector{<:Real}, name::AbstractString)
    if length(edges) < 2
        throw(ArgumentError("$name must contain at least two edges"))
    end
    if !all(>(0), diff(edges))
        throw(ArgumentError("$name must be strictly increasing"))
    end
    return nothing
end


# ---------------------------------------------------------------- single bins
"""
    rebin_one(lo, hi, e, f)

Overlap-weighted mean of the old flux densities `f` (on edges `e`) over the new bin
[lo, hi]; `NaN` if the new bin is not fully covered.
"""
function rebin_one(lo, hi, e, f)
    r = bin_range(lo, hi, e)
    if isnothing(r)
        return oftype(float(lo), NaN)
    end

    weighted_sum = sum(j -> overlap(lo, hi, e, j) * f[j], r)
    return weighted_sum / (hi - lo)
end


"""
    rebin_noise_one(lo, hi, e, σ)

1σ error of `rebin_one` for independent old pixels with errors `σ`; `NaN` if the new
bin is not fully covered.
"""
function rebin_noise_one(lo, hi, e, σ)
    r = bin_range(lo, hi, e)
    if isnothing(r)
        return oftype(float(lo), NaN)
    end

    variance_sum = sum(j -> (overlap(lo, hi, e, j) * σ[j])^2, r)
    return sqrt(variance_sum) / (hi - lo)
end


"""
    rebin_good_one(lo, hi, e, good)

`true` if the new bin [lo, hi] is fully covered and every old pixel it overlaps is good.
"""
function rebin_good_one(lo, hi, e, good)
    r = bin_range(lo, hi, e)
    if isnothing(r)
        return false
    end

    return all(j -> good[j] || overlap(lo, hi, e, j) == 0, r)
end


# ---------------------------------------------------------------- whole arrays
"""
    rebin_flux(new_edges, old_edges, flux)

Flux-density rebinning. `flux` may be a vector or a matrix (one spectrum per
column, e.g. SSP templates). Bins not fully covered by `old_edges` are `NaN`.
"""
function rebin_flux(ne::AbstractVector{<:Real}, e::AbstractVector{<:Real},
                    f::AbstractVector{<:Real})
    check_edges(ne, "new edges")
    check_edges(e, "old edges")
    if length(f) != nbins(e)
        throw(DimensionMismatch("flux has $(length(f)) pixels, old edges describe $(nbins(e))"))
    end

    return [rebin_one(ne[i], ne[i+1], e, f) for i in 1:nbins(ne)]
end

function rebin_flux(ne::AbstractVector{<:Real}, e::AbstractVector{<:Real},
                    F::AbstractMatrix{<:Real})
    columns = [rebin_flux(ne, e, column) for column in eachcol(F)]
    return reduce(hcat, columns)
end


"""
    rebin_noise(new_edges, old_edges, σ)

Propagate 1σ errors through `rebin_flux`, assuming independent pixels.
"""
function rebin_noise(ne::AbstractVector{<:Real}, e::AbstractVector{<:Real},
                     σ::AbstractVector{<:Real})
    check_edges(ne, "new edges")
    check_edges(e, "old edges")
    if length(σ) != nbins(e)
        throw(DimensionMismatch("noise has $(length(σ)) pixels, old edges describe $(nbins(e))"))
    end

    return [rebin_noise_one(ne[i], ne[i+1], e, σ) for i in 1:nbins(ne)]
end


"""
    rebin_good(new_edges, old_edges, good)

Rebin a pixel mask: a new pixel is good only if it is fully covered and every old
pixel it overlaps is good.
"""
function rebin_good(ne::AbstractVector{<:Real}, e::AbstractVector{<:Real},
                    good::AbstractVector{Bool})
    check_edges(ne, "new edges")
    check_edges(e, "old edges")
    if length(good) != nbins(e)
        throw(DimensionMismatch("mask has $(length(good)) pixels, old edges describe $(nbins(e))"))
    end

    mask = [rebin_good_one(ne[i], ne[i+1], e, good) for i in 1:nbins(ne)]
    return BitVector(mask)
end


# ---------------------------------------------------------------- LSF
"""
    interp_linear(x, xs, ys)

Linear interpolation of `ys(xs)` at `x`, clamped to the end values outside `xs`.
"""
function interp_linear(x, xs, ys)
    i = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    t = clamp((x - xs[i]) / (xs[i+1] - xs[i]), 0, 1)
    return ys[i] + t * (ys[i+1] - ys[i])
end


"""
    rebin_lsf(lsf, xnew, xold)

Interpolate a per-pixel LSF (or `nothing`) from wavelengths `xold` onto `xnew`.
"""
function rebin_lsf(::Nothing, xnew, xold)
    return nothing
end

function rebin_lsf(lsf::AbstractVector{<:Real}, xnew, xold)
    return [interp_linear(x, xold, lsf) for x in xnew]
end


# ---------------------------------------------------------------- log grid
"""
    median_velscale(wave)

Median velocity step [km/s] between neighbouring pixels of an ascending wavelength grid.
"""
function median_velscale(wave::AbstractVector{<:Real})
    if length(wave) < 2
        throw(ArgumentError("need at least two wavelengths"))
    end

    steps = diff(log.(wave))
    return C_KMS * median(steps)
end


"""
    loggrid_within(e, velscale; pad=1e-12)

Log grid whose outer pixel edges lie inside the old edges `e`. The tiny pad
guards against round-off pushing an edge outside the covered range.
"""
function loggrid_within(e::AbstractVector{<:Real}, velscale::Real; pad::Real=1e-12)
    half_step = exp(velscale / C_KMS / 2)
    λmin = first(e) * half_step * (1 + pad)
    λmax = last(e) / half_step * (1 - pad)

    grid = λmax >= λmin ? loggrid(λmin, λmax, velscale) : Float64[]
    if length(grid) < 2
        range_kms = C_KMS * log(last(e) / first(e))
        throw(ArgumentError("velscale = $velscale km/s leaves fewer than two pixels in a $(round(range_kms; sigdigits=4)) km/s range"))
    end
    return grid
end


# ---------------------------------------------------------------- forward-model projection
# Instead of rebinning the data (which correlates the noise), evaluate the model on a
# fine log grid and average it onto the native pixels: A_native = P · A_model.

"""
    native_pixel_edges(wave)

Pixel edges of a data grid: geometric midpoints if `wave` is log-uniform (as for SDSS),
arithmetic midpoints otherwise (linear or irregular grids).
"""
function native_pixel_edges(wave::AbstractVector{<:Real})
    if length(wave) >= 3 && is_loguniform(wave)
        return log_pixel_edges(wave)
    else
        return pixel_edges(wave)
    end
end

"""
    model_grid(wave; oversample=2)

Log-uniform model wavelengths whose pixels fully cover the native pixels of `wave`
(any ascending grid), with a velscale `oversample ×` finer than the narrowest native
pixel. Use it as the data grid for templates, gas and polynomial, and project with
`pixel_projection`.
"""
function model_grid(wave::AbstractVector{<:Real}; oversample::Real=2)
    if oversample < 1
        throw(ArgumentError("oversample must be ≥ 1"))
    end

    if first(wave) <= 0
        throw(ArgumentError("the native pixels must lie at positive wavelengths"))
    end
    e = native_pixel_edges(wave)
    if first(e) <= 0
        throw(ArgumentError("the native pixels must lie at positive wavelengths"))
    end

    narrowest_dlnλ = minimum(diff(log.(e)))
    dlnλ = narrowest_dlnλ / oversample
    n = ceil(Int, log(last(e) / first(e)) / dlnλ) + 1
    return first(e) .* exp.(dlnλ .* (0:n-1))
end


"""
    pixel_projection(model_wave, data_wave; data_edges=native_pixel_edges(data_wave),
                     sampling=:average, resolution=nothing)

Sparse `(ndata, nmodel)` matrix `P` mapping a model on the log grid `model_wave` onto
the data pixels.

- `sampling=:average` (default): each data pixel is the overlap-weighted mean of the
  model pixels it covers, `P * f == rebin_flux(data_edges, log_pixel_edges(model_wave), f)`.
- `sampling=:point`: the model is linearly interpolated at the data pixel centres. Use
  this when the data pixels are point samples (as for a resolution matrix defined on
  sampled rather than pixel-integrated flux).
- `resolution`: a `BandedLSF` on the data pixels (e.g. a DESI resolution matrix, via
  `banded_from_dia`). It is applied after sampling, giving `R · P`. Components then
  need no instrumental broadening of their own (see `GasComponent`, `σ_templ`).

By default the data pixel edges are geometric midpoints for a log-uniform `data_wave`
and arithmetic midpoints otherwise (`native_pixel_edges`); pass `data_edges` to override.
`data_edges` is not used with `sampling=:point`.

Rows of the sampling part sum to one. Every data pixel must lie inside the model grid.
"""
function pixel_projection(model_wave::AbstractVector{<:Real}, data_wave::AbstractVector{<:Real};
                          data_edges=nothing, sampling::Symbol=:average, resolution=nothing)
    if length(model_wave) < 2
        throw(ArgumentError("the model grid needs at least two pixels"))
    end
    if isempty(data_wave)
        throw(ArgumentError("need at least one data pixel"))
    end

    if sampling === :average
        if isnothing(data_edges)
            data_edges = native_pixel_edges(data_wave)
        end
        if length(data_edges) != length(data_wave) + 1
            throw(DimensionMismatch("need $(length(data_wave) + 1) data edges, got $(length(data_edges))"))
        end
        P = average_projection(model_wave, data_edges)
    elseif sampling === :point
        P = point_projection(model_wave, data_wave)
    else
        throw(ArgumentError("sampling must be :average or :point, got $(repr(sampling))"))
    end

    if isnothing(resolution)
        return P
    end

    if npix(resolution) != size(P, 1)
        throw(DimensionMismatch("resolution matrix is $(npix(resolution))×$(npix(resolution)), data have $(size(P, 1)) pixels"))
    end
    return sparse_matrix(resolution) * P
end


"""
    average_projection(model_wave, data_edges)

Sparse matrix averaging model pixels (log grid `model_wave`) over each data pixel
`[data_edges[i], data_edges[i+1]]`, weighted by overlap.
"""
function average_projection(model_wave, data_edges)
    e = log_pixel_edges(model_wave)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for i in 1:nbins(data_edges)
        lo = data_edges[i]
        hi = data_edges[i+1]
        r = bin_range(lo, hi, e)
        if isnothing(r)
            throw(ArgumentError("data pixel $i is not covered by the model grid"))
        end

        for j in r
            o = overlap(lo, hi, e, j)
            if o <= 0
                continue
            end
            push!(rows, i)
            push!(cols, j)
            push!(vals, o / (hi - lo))
        end
    end

    return sparse(rows, cols, vals, nbins(data_edges), length(model_wave))
end


"""
    point_projection(model_wave, data_wave)

Sparse matrix linearly interpolating the model (on `model_wave`) at each data pixel centre.
"""
function point_projection(model_wave, data_wave)
    n = length(model_wave)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (i, x) in enumerate(data_wave)
        if !(first(model_wave) <= x <= last(model_wave))
            throw(ArgumentError("data pixel $i is not covered by the model grid"))
        end

        j = clamp(searchsortedlast(model_wave, x), 1, n - 1)
        t = (x - model_wave[j]) / (model_wave[j+1] - model_wave[j])
        push!(rows, i, i)
        push!(cols, j, j + 1)
        push!(vals, 1 - t, t)
    end

    P = sparse(rows, cols, vals, length(data_wave), n)
    return dropzeros!(P)
end


# ---------------------------------------------------------------- spectra
"""
    zero_where_bad(x, good)

Copy of `x` with the entries where `good` is `false` set to zero.
"""
function zero_where_bad(x::AbstractVector, good::AbstractVector{Bool})
    return ifelse.(good, x, zero(eltype(x)))
end


"""
    inf_where_bad(x, good)

Copy of `x` with the entries where `good` is `false` set to `Inf`.
"""
function inf_where_bad(x::AbstractVector, good::AbstractVector{Bool})
    return ifelse.(good, x, oftype(first(x), Inf))
end


"""
    rebin_log(s::Spectrum; velscale=median_velscale(s.wave))

Resample onto a log-uniform grid (`velscale` km/s per pixel), conserving flux.
Bad/uncovered pixels get flux 0, noise Inf, `good=false`.
"""
function rebin_log(s::Spectrum; velscale::Real=median_velscale(s.wave))
    if velscale <= 0
        throw(ArgumentError("velscale must be positive"))
    end

    e = pixel_edges(s.wave)
    grid = loggrid_within(e, velscale)
    ne = log_pixel_edges(grid)

    flux = rebin_flux(ne, e, zero_where_bad(s.flux, s.good))
    noise = rebin_noise(ne, e, zero_where_bad(s.noise, s.good))
    covered = rebin_good(ne, e, s.good)
    good = covered .& isfinite.(flux) .& isfinite.(noise)
    lsf = rebin_lsf(s.lsf_fwhm, grid, s.wave)

    return Spectrum(grid, zero_where_bad(flux, good), inf_where_bad(noise, good);
                    good=good, lsf_fwhm=lsf)
end
