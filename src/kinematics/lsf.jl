# Instrumental resolution (LSF). Two representations share one interface:
#   GaussianLSF  per-pixel Gaussian σ [km/s]
#   BandedLSF    resolution matrix R stored by diagonals (e.g. from a pipeline)
# All velocity quantities assume a log-uniform wavelength grid.

const FWHM_PER_SIGMA = 2 * sqrt(2 * log(2))


"""
    fwhm_to_sigma_kms(fwhm, wave)

Gaussian velocity dispersion [km/s] for a FWHM [Å] at wavelength `wave` [Å]. Either
argument may be a scalar or a vector (broadcast).
"""
function fwhm_to_sigma_kms(fwhm, wave)
    if any(<(0), fwhm)
        throw(ArgumentError("FWHM must be ≥ 0"))
    end
    if any(<=(0), wave)
        throw(ArgumentError("wavelengths must be positive"))
    end

    return @. C_KMS * fwhm / (FWHM_PER_SIGMA * wave)
end


"""
    sigma_kms_to_fwhm(σ, wave)

FWHM [Å] at wavelength `wave` [Å] of a Gaussian of dispersion `σ` [km/s]; the inverse
of `fwhm_to_sigma_kms`.
"""
function sigma_kms_to_fwhm(σ, wave)
    if any(<(0), σ)
        throw(ArgumentError("σ must be ≥ 0"))
    end
    if any(<=(0), wave)
        throw(ArgumentError("wavelengths must be positive"))
    end

    return @. σ * FWHM_PER_SIGMA * wave / C_KMS
end


abstract type LSF end


# ---------------------------------------------------------------- Gaussian
"""
    GaussianLSF(sigma_kms)
    GaussianLSF(wave, fwhm)
    GaussianLSF(spectrum)
    GaussianLSF(spectrum, wave)

Per-pixel Gaussian instrumental dispersion σ [km/s]. From `(wave, fwhm)`, `fwhm` [Å]
may be a scalar or one value per pixel. From a spectrum, its `lsf_fwhm` is used,
optionally interpolated onto another grid `wave` (e.g. a `model_grid`).
"""
struct GaussianLSF{T<:AbstractFloat} <: LSF
    sigma_kms::Vector{T}

    function GaussianLSF{T}(sigma_kms) where {T<:AbstractFloat}
        if !all(>=(0), sigma_kms)
            throw(ArgumentError("σ must be ≥ 0"))
        end
        return new{T}(sigma_kms)
    end
end


function GaussianLSF(σ::AbstractVector{<:Real})
    T = float(eltype(σ))
    return GaussianLSF{T}(Vector{T}(σ))
end

function GaussianLSF(wave::AbstractVector{<:Real}, fwhm)
    if fwhm isa AbstractVector && length(fwhm) != length(wave)
        throw(ArgumentError("fwhm has $(length(fwhm)) entries, wave has $(length(wave))"))
    end

    σ = fwhm_to_sigma_kms(fwhm, wave)
    return GaussianLSF(σ)
end

function GaussianLSF(s::Spectrum)
    if isnothing(s.lsf_fwhm)
        throw(ArgumentError("spectrum has no lsf_fwhm"))
    end

    return GaussianLSF(s.wave, s.lsf_fwhm)
end

function GaussianLSF(s::Spectrum, wave::AbstractVector{<:Real})
    if isnothing(s.lsf_fwhm)
        throw(ArgumentError("spectrum has no lsf_fwhm"))
    end

    fwhm = [interp_linear(λ, s.wave, s.lsf_fwhm) for λ in wave]
    return GaussianLSF(wave, fwhm)
end


"""
    npix(lsf)

Return the number of pixels the LSF is defined on.
"""
function npix(l::GaussianLSF)
    return length(l.sigma_kms)
end


# ---------------------------------------------------------------- banded matrix
"""
    BandedLSF(bands, offsets)

Resolution matrix `R` in diagonal storage: `bands[d, i] = R[i, i + offsets[d]]`,
so `(R x)[i] = Σ_d bands[d, i] * x[i + offsets[d]]`. Entries pointing outside the
spectrum are ignored. Use `banded_from_dia` for scipy/DESI-style `(data, offsets)`.
"""
struct BandedLSF{T<:AbstractFloat} <: LSF
    bands::Matrix{T}
    offsets::Vector{Int}

    function BandedLSF{T}(bands, offsets) where {T<:AbstractFloat}
        if size(bands, 1) != length(offsets)
            throw(ArgumentError("bands needs one row per offset"))
        end
        if !allunique(offsets)
            throw(ArgumentError("offsets must be unique"))
        end
        return new{T}(bands, offsets)
    end
end


function BandedLSF(bands::AbstractMatrix{<:Real}, offsets::AbstractVector{<:Integer})
    T = float(eltype(bands))
    return BandedLSF{T}(Matrix{T}(bands), Vector{Int}(offsets))
end


function npix(R::BandedLSF)
    return size(R.bands, 2)
end


"""
    in_band(i, off, n)

`true` if column `i + off` of an `n × n` banded matrix exists.
"""
function in_band(i::Integer, off::Integer, n::Integer)
    return 1 <= i + off <= n
end


"""
    banded_from_dia(data, offsets)

Convert scipy `dia_matrix`-style storage (`data[d, c] = A[c - offsets[d], c]`,
as used by DESI resolution matrices) to a `BandedLSF`.
"""
function banded_from_dia(data::AbstractMatrix{<:Real}, offsets::AbstractVector{<:Integer})
    nd, n = size(data)
    if length(offsets) != nd
        throw(ArgumentError("need one offset per diagonal (row of data): got $(length(offsets)) for $nd"))
    end

    bands = zeros(float(eltype(data)), nd, n)
    for (d, off) in enumerate(offsets)
        for i in 1:n
            if in_band(i, off, n)
                bands[d, i] = data[d, i + off]
            end
        end
    end
    return BandedLSF(bands, offsets)
end


# ---------------------------------------------------------------- banded algebra
"""
    row_moment(R, p)

Return `Σ_d bands[d, i] * offsets[d]^p` over the in-range entries of each row `i`
(the `p`-th moment of each row).
"""
function row_moment(R::BandedLSF, p::Integer)
    n = npix(R)
    m = zeros(eltype(R.bands), n)
    for (d, off) in enumerate(R.offsets)
        for i in 1:n
            if in_band(i, off, n)
                m[i] += R.bands[d, i] * off^p
            end
        end
    end
    return m
end


"""
    normalize_rows(R)

Rescale each row of `R` to sum to one (over in-range entries).
"""
function normalize_rows(R::BandedLSF)
    row_sums = row_moment(R, 0)
    if !all(>(0), row_sums)
        throw(ArgumentError("a row of the resolution matrix has no support"))
    end

    return BandedLSF(R.bands ./ row_sums', R.offsets)
end


"""
    dense_matrix(R)

Return `R` as a dense `npix × npix` matrix (for tests and small problems).
"""
function dense_matrix(R::BandedLSF{T}) where {T}
    n = npix(R)
    M = zeros(T, n, n)
    for (d, off) in enumerate(R.offsets)
        for i in 1:n
            if in_band(i, off, n)
                M[i, i + off] = R.bands[d, i]
            end
        end
    end
    return M
end


"""
    sparse_matrix(R)

Return `R` as a sparse `npix × npix` matrix (in-range entries only).
"""
function sparse_matrix(R::BandedLSF)
    n = npix(R)
    rows = Int[]
    cols = Int[]
    vals = eltype(R.bands)[]
    for (d, off) in enumerate(R.offsets)
        for i in 1:n
            if !in_band(i, off, n)
                continue
            end
            push!(rows, i)
            push!(cols, i + off)
            push!(vals, R.bands[d, i])
        end
    end
    return sparse(rows, cols, vals, n, n)
end


"""
    add_band!(out, bands, d, off, X)

Add the contribution of diagonal `d` (offset `off`) of a banded matrix times `X` to `out`.
"""
function add_band!(out, bands, d, off, X)
    n = size(X, 1)
    rows = max(1, 1 - off):min(n, n - off)
    for j in axes(X, 2)
        for i in rows
            @inbounds out[i, j] += bands[d, i] * X[i + off, j]
        end
    end
    return out
end


"""
    apply_banded(R, X)

Apply `R` to a vector, or to every column of a matrix.
"""
function apply_banded(R::BandedLSF{T}, X::AbstractMatrix) where {T}
    if size(X, 1) != npix(R)
        throw(DimensionMismatch("matrix has $(size(X, 1)) rows, R is $(npix(R))×$(npix(R))"))
    end

    out = zeros(promote_type(T, eltype(X)), size(X))
    for (d, off) in enumerate(R.offsets)
        add_band!(out, R.bands, d, off, X)
    end
    return out
end

function apply_banded(R::BandedLSF, x::AbstractVector)
    as_column = reshape(x, :, 1)
    return vec(apply_banded(R, as_column))
end


# ---------------------------------------------------------------- Gaussian → banded
"""
    gauss_cdf(x)

Standard normal cumulative distribution function.
"""
function gauss_cdf(x)
    return (1 + erf(x / sqrt(2))) / 2
end


"""
    pixel_gauss_weight(k, σ)

Weight of offset `k` pixels for a pixel-integrated Gaussian of width `σ` pixels.
"""
function pixel_gauss_weight(k, σ)
    upper = gauss_cdf((k + 0.5) / σ)
    lower = gauss_cdf((k - 0.5) / σ)
    return upper - lower
end


"""
    banded_gaussian(σ_pix; nsig=5)

Banded smoothing matrix with a pixel-integrated Gaussian of width `σ_pix[i]`
(pixels, may vary per pixel and may be 0) in row `i`; rows sum to one.
"""
function banded_gaussian(σ_pix::AbstractVector{<:Real}; nsig::Real=5)
    if any(<(0), σ_pix)
        throw(ArgumentError("σ must be ≥ 0"))
    end
    if nsig <= 0
        throw(ArgumentError("nsig must be positive"))
    end

    half_width = ceil(Int, nsig * maximum(σ_pix))
    offsets = collect(-half_width:half_width)
    bands = [pixel_gauss_weight(k, σ) for k in offsets, σ in σ_pix]
    return normalize_rows(BandedLSF(bands, offsets))
end


"""
    to_banded(lsf, velscale)

Banded equivalent of an LSF on a grid with `velscale` km/s per pixel (a `BandedLSF` is
returned unchanged).
"""
function to_banded(l::GaussianLSF, velscale::Real)
    if velscale <= 0
        throw(ArgumentError("velscale must be positive"))
    end

    return banded_gaussian(l.sigma_kms ./ velscale)
end

function to_banded(R::BandedLSF, velscale::Real)
    return R
end


# ---------------------------------------------------------------- effective σ
"""
    inst_sigma_kms(lsf, velscale)

Per-pixel instrumental σ [km/s]. For a `BandedLSF` this is the square root of each
row's second central moment (exact for Gaussian rows, plus the 1/12 px² boxcar
term from pixel integration), so quadrature subtraction treats it as Gaussian.
"""
function inst_sigma_kms(l::GaussianLSF, velscale::Real)
    return l.sigma_kms
end

function inst_sigma_kms(R::BandedLSF, velscale::Real)
    if velscale <= 0
        throw(ArgumentError("velscale must be positive"))
    end

    m0 = row_moment(R, 0)
    m1 = row_moment(R, 1)
    m2 = row_moment(R, 2)
    mean_offset = m1 ./ m0
    variance = m2 ./ m0 .- mean_offset .^ 2
    return velscale .* sqrt.(max.(variance, 0))
end
