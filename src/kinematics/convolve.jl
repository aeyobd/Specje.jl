# FFT convolution of templates with a Gaussian LOSVD on a shared log-λ grid.
# Velocity convention: v = c·Δln λ (pPXF's), so a shift is v / velscale pixels.
# Pass rest-frame data wavelengths (wave / (1 + z_sys)); v is then the residual.

"""
    KinematicsGrid

Geometry linking the (wider) template grid to the data (model) grid; build it with
`kinematics_grid`.

- `nfft`     padded transform length
- `ntempl`   number of template-grid pixels
- `crop`     data pixels within the template grid
- `frac`     leftover sub-pixel offset of the first data pixel (|frac| ≤ ½)
- `velscale` km/s per pixel, `freqs` the `rfft` frequencies
"""
struct KinematicsGrid{T<:AbstractFloat}
    nfft::Int
    ntempl::Int
    crop::UnitRange{Int}
    frac::T
    velscale::T
    freqs::Vector{T}
end


"""
    check_same_velscale(twave, dwave)

Throw an `ArgumentError` unless both grids are log-uniform with the same velscale
(to 1e-6); return the template velscale.
"""
function check_same_velscale(twave::AbstractVector{<:Real}, dwave::AbstractVector{<:Real})
    if !is_loguniform(twave) || !is_loguniform(dwave)
        throw(ArgumentError("template and data wavelengths must both be log-uniform"))
    end

    vt = velscale(twave)
    vd = velscale(dwave)
    if !isapprox(vt, vd; rtol=1e-6)
        throw(ArgumentError("template velscale $vt ≠ data velscale $vd"))
    end
    return vt
end


"""
    offset_pixels(twave, dwave, vs)

Position of the first data pixel in template-pixel units (0-based, fractional).
"""
function offset_pixels(twave::AbstractVector{<:Real}, dwave::AbstractVector{<:Real}, vs::Real)
    return C_KMS * log(first(dwave) / first(twave)) / vs
end


"""
    check_coverage(crop, ntempl, margin)

Throw an `ArgumentError` unless the template grid (`ntempl` pixels) extends at least
`margin` pixels beyond the data pixels `crop` on both sides.
"""
function check_coverage(crop::UnitRange{Int}, ntempl::Integer, margin::Integer)
    blue_margin = first(crop) - 1
    red_margin = ntempl - last(crop)
    if blue_margin < margin || red_margin < margin
        throw(ArgumentError("templates must extend ≥ $margin pixels beyond the data on both sides"))
    end
    return nothing
end


"""
    kinematics_grid(twave, dwave; margin_kms=2500)

Build the FFT geometry. Templates must extend at least `margin_kms` beyond the
data on both sides, and `margin_kms` should exceed `|v|max + 5σmax`; this is what
keeps wrap-around and edge effects out of the cropped result.
"""
function kinematics_grid(twave::AbstractVector{<:Real}, dwave::AbstractVector{<:Real};
                         margin_kms::Real=2500.0)
    if margin_kms < 0
        throw(ArgumentError("margin_kms must be ≥ 0"))
    end

    T = float(promote_type(eltype(twave), eltype(dwave)))
    vs = check_same_velscale(twave, dwave)

    offset = offset_pixels(twave, dwave, vs)
    i0 = round(Int, offset)
    crop = (i0 + 1):(i0 + length(dwave))
    margin = ceil(Int, margin_kms / vs)
    check_coverage(crop, length(twave), margin)

    nfft = nextprod((2, 3, 5), length(twave) + margin)
    freqs = T.(rfftfreq(nfft))
    return KinematicsGrid{T}(nfft, length(twave), crop, T(offset - i0), T(vs), freqs)
end


"""
    shift_pixels(kg, v)

Pixel displacement of the model for velocity `v` [km/s] on grid `kg`.
"""
function shift_pixels(kg::KinematicsGrid, v::Real)
    return v / kg.velscale - kg.frac
end


"""
    transform_templates(kg, T)

Zero-pad template columns (or a single template vector) to `kg.nfft` and Fourier
transform them. Done once per component.
"""
function transform_templates(kg::KinematicsGrid, T::AbstractMatrix{<:Real})
    n = size(T, 1)
    if n != kg.ntempl
        throw(ArgumentError("templates have $n pixels, the grid's template wavelengths have $(kg.ntempl)"))
    end

    padded = zeros(eltype(T), kg.nfft, size(T, 2))
    padded[1:n, :] .= T
    return rfft(padded, 1)
end

function transform_templates(kg::KinematicsGrid, t::AbstractVector{<:Real})
    as_column = reshape(t, :, 1)
    return transform_templates(kg, as_column)
end


"""
    convolve_templates(kg, Tft, v, σ; σ_templ=0)

Templates (transformed by `transform_templates`) broadened by a Gaussian LOSVD with
velocity `v` and dispersion `σ` [km/s], returned on the data pixels as `(npix, ntemp)`.
With `σ_templ > 0` the templates are taken to be already broadened by a Gaussian of
`σ_templ` km/s, and only the remainder `√(σ² − σ_templ²)` is applied (needs σ ≥ σ_templ).
"""
function convolve_templates(kg::KinematicsGrid, Tft::AbstractMatrix, v::Real, σ::Real;
                            σ_templ::Real=0)
    if size(Tft, 1) != length(kg.freqs)
        throw(DimensionMismatch("transformed templates have $(size(Tft, 1)) frequencies, the grid has $(length(kg.freqs))"))
    end

    variance_pixels = (σ^2 - σ_templ^2) / kg.velscale^2
    variance_pixels = max(variance_pixels, zero(variance_pixels))
    kernel = gaussian_losvd_ft_var(kg.freqs, shift_pixels(kg, v), variance_pixels)

    convolved = irfft_cols(Tft .* kernel, kg.nfft)
    return convolved[kg.crop, :]
end


"""
    irfft_cols(X, n)

Column-wise inverse real FFT of length `n`. Also works for ForwardDiff dual numbers,
which FFTW cannot transform: since the transform is linear, the value and each
partial derivative are transformed separately and reassembled.
"""
function irfft_cols(X::AbstractMatrix, n::Integer)
    return irfft(X, n, 1)
end

function irfft_cols(X::AbstractMatrix{Complex{ForwardDiff.Dual{Tg,V,N}}}, n::Integer) where {Tg,V,N}
    function transform_part(extract)
        part = map(z -> complex(extract(real(z)), extract(imag(z))), X)
        return irfft(part, n, 1)
    end

    values = transform_part(ForwardDiff.value)
    partials = ntuple(k -> transform_part(x -> ForwardDiff.partials(x, k)), N)
    return map((value, parts...) -> ForwardDiff.Dual{Tg}(value, parts...), values, partials...)
end


"""
    convolve_blocks(kg, blocks, kinematics)

One kinematic component per template block: `kinematics[i] = (v, σ)`. The
convolved blocks are concatenated column-wise (a two-component fit uses two blocks).
"""
function convolve_blocks(kg::KinematicsGrid, blocks, kinematics)
    if length(blocks) != length(kinematics)
        throw(ArgumentError("need one (v, σ) per template block: got $(length(kinematics)) for $(length(blocks))"))
    end

    convolved = [convolve_templates(kg, block, k...) for (block, k) in zip(blocks, kinematics)]
    return reduce(hcat, convolved)
end
