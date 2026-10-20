# Sufficient statistics of the weighted linear problem  b = A w + ε.
# Rows are divided by the observed errors (FitData); each data block (spectrum,
# photometry) may carry its own noise-scale factor s_k, entering only through 1/s_k².

"""
    BlockStats(AtA, Atb, btb, n, logσ)

Statistics of one block of noise-weighted rows: AᵀA, Aᵀb, bᵀb, the row count, and
Σ log σ over the rows. The element type is promoted, so AD numbers (from θ or s) and
plain floats (from the data) can mix.
"""
struct BlockStats{T<:Real}
    AtA::Matrix{T}
    Atb::Vector{T}
    btb::T
    n::Int
    logσ::T
end


function BlockStats(AtA::AbstractMatrix, Atb::AbstractVector, btb::Real, n::Integer, logσ::Real)
    T = promote_type(eltype(AtA), eltype(Atb), typeof(btb), typeof(logσ))
    return BlockStats{T}(AtA, Atb, btb, n, logσ)
end


"""
    block_stats(Aw, b, rows, logσ)

`BlockStats` of the noise-weighted design matrix `Aw` and data `b` over `rows`.
"""
function block_stats(Aw::AbstractMatrix, b::AbstractVector, rows::UnitRange{Int}, logσ::Real)
    Ab = view(Aw, rows, :)
    bb = view(b, rows)
    AtA = Ab' * Ab
    Atb = Ab' * bb
    btb = dot(bb, bb)
    return BlockStats(AtA, Atb, btb, length(rows), logσ)
end


"""
    weighted_design(A, fd)

The good rows of the design matrix `A`, each divided by its observed error.
"""
function weighted_design(A::AbstractMatrix, fd::FitData)
    return A[fd.idx, :] .* fd.inv_noise
end


"""
    block_logσ(fd, rows)

Σ log σ over the data rows `rows` (the noise normalisation of the likelihood).
"""
function block_logσ(fd::FitData, rows::UnitRange{Int})
    return -sum(log, view(fd.inv_noise, rows))
end


"""
    block_stats_all(A, fd)

`BlockStats` for each data block of `fd` at design matrix `A`.
"""
function block_stats_all(A::AbstractMatrix, fd::FitData)
    if size(A, 1) < maximum(fd.idx; init=0)
        throw(DimensionMismatch("design matrix has $(size(A, 1)) rows, the data use row $(maximum(fd.idx))"))
    end

    Aw = weighted_design(A, fd)
    return [block_stats(Aw, fd.b, rows, block_logσ(fd, rows)) for rows in fd.blocks]
end


"""
    CombinedStats(AtA, Atb, btb, n, logdetrows)

Statistics after applying per-block noise scales: rows effectively divided by s_k σ_i.
`logdetrows` is Σ_i log(s_k σ_i). Build with `combine`.
"""
struct CombinedStats{T<:Real}
    AtA::Matrix{T}
    Atb::Vector{T}
    btb::T
    n::Int
    logdetrows::T         # Σ_i log(s_k σ_i)
end


function CombinedStats(AtA::AbstractMatrix, Atb::AbstractVector, btb::Real, n::Integer,
                       logdetrows::Real)
    T = promote_type(eltype(AtA), eltype(Atb), typeof(btb), typeof(logdetrows))
    return CombinedStats{T}(AtA, Atb, btb, n, logdetrows)
end


"""
    combine(blocks, s)

Combine block statistics with noise-scale factors `s` (one per block): each block's
rows are divided by `s[k]`, so its AᵀA, Aᵀb and bᵀb are scaled by `1 / s[k]²`.
"""
function combine(blocks::AbstractVector{<:BlockStats}, s::AbstractVector)
    if length(s) != length(blocks)
        throw(ArgumentError("need one noise scale per block ($(length(blocks)))"))
    end
    if isempty(blocks)
        throw(ArgumentError("need at least one data block"))
    end

    weights = 1 ./ s .^ 2

    AtA = weights[1] .* blocks[1].AtA
    Atb = weights[1] .* blocks[1].Atb
    btb = weights[1] * blocks[1].btb
    n = blocks[1].n
    logdetrows = blocks[1].logσ + blocks[1].n * log(s[1])
    for k in 2:length(blocks)
        AtA = AtA .+ weights[k] .* blocks[k].AtA
        Atb = Atb .+ weights[k] .* blocks[k].Atb
        btb = btb + weights[k] * blocks[k].btb
        n = n + blocks[k].n
        logdetrows = logdetrows + blocks[k].logσ + blocks[k].n * log(s[k])
    end

    return CombinedStats(AtA, Atb, btb, n, logdetrows)
end
