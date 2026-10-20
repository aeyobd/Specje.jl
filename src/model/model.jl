"""
    cumranges(counts)

Consecutive index ranges for blocks of the given sizes (empty blocks give empty ranges),
e.g. `cumranges([2, 0, 3]) == [1:2, 3:2, 3:5]`.
"""
function cumranges(counts)
    stops = cumsum(collect(counts))
    return [(stop - count + 1):stop for (stop, count) in zip(stops, counts)]
end


"""
    SpectralModel(grid, components...; projection=nothing)

The forward model `F(θ) = A(θ) w`, with `A` the concatenation of every component's
columns. θ is the concatenation of the components' nonlinear parameters.

Components are evaluated on the grid's data pixels (`grid.crop`), the *model grid*. With
`projection` (e.g. `pixel_projection(model_wave, native_wave)`, size `(ndata, nmodel)`)
the columns are mapped onto the native data pixels, `A = P · A_model`, so the data can be
fitted without rebinning them.
"""
struct SpectralModel{G<:KinematicsGrid,C<:Tuple,P<:Union{Nothing,AbstractMatrix}}
    grid::G
    components::C
    projection::P

    function SpectralModel(grid::G, components::C, projection::P=nothing) where
            {G<:KinematicsGrid,C<:Tuple,P<:Union{Nothing,AbstractMatrix}}
        if isempty(components)
            throw(ArgumentError("a model needs at least one component"))
        end

        nmodel = length(grid.crop)
        for c in components
            check_component(c, nmodel)
        end

        if !isnothing(projection) && size(projection, 2) != nmodel
            throw(DimensionMismatch("projection has $(size(projection, 2)) columns, the model grid has $nmodel pixels"))
        end
        return new{G,C,P}(grid, components, projection)
    end
end


function SpectralModel(grid::KinematicsGrid, components::Component...; projection=nothing)
    return SpectralModel(grid, components, projection)
end


# ---------------------------------------------------------------- bookkeeping
"""
    nmodel(m::SpectralModel)

Number of model-grid pixels (where the components are evaluated).
"""
function nmodel(m::SpectralModel)
    return length(m.grid.crop)
end


"""
    npix(m::SpectralModel)

Number of data pixels: the projection's rows, or the model grid without a projection.
"""
function npix(m::SpectralModel)
    if isnothing(m.projection)
        return nmodel(m)
    else
        return size(m.projection, 1)
    end
end


function nparams(m::SpectralModel)
    return sum(nparams, m.components)
end


function ncolumns(m::SpectralModel)
    return sum(ncolumns, m.components)
end


"""
    param_ranges(m)

Index range of each component's parameters within θ.
"""
function param_ranges(m::SpectralModel)
    counts = map(nparams, m.components)
    return cumranges(counts)
end


"""
    column_ranges(m)

Index range of each component's columns within the design matrix (and the weights).
"""
function column_ranges(m::SpectralModel)
    counts = map(ncolumns, m.components)
    return cumranges(counts)
end


"""
    check_theta(model, θ)

Throw an `ArgumentError` unless `θ` has `nparams(model)` entries.
"""
function check_theta(m, θ::AbstractVector)
    if length(θ) != nparams(m)
        throw(ArgumentError("expected $(nparams(m)) parameters, got $(length(θ))"))
    end
    return nothing
end


# ---------------------------------------------------------------- design matrix
"""
    design_matrix(model, θ)

Design matrix `A(θ)`, `(npix, ncolumns)` on the data pixels; the model flux is `A * w`.
"""
function design_matrix(m::SpectralModel, θ::AbstractVector)
    check_theta(m, θ)

    blocks = [columns(c, m.grid, view(θ, r)) for (c, r) in zip(m.components, param_ranges(m))]
    A_model = reduce(hcat, blocks)
    return project(m.projection, A_model)
end


"""
    project(P, A)

Map model-grid columns `A` onto the data pixels (`P * A`; unchanged without a projection).
"""
function project(::Nothing, A::AbstractMatrix)
    return A
end

function project(P::AbstractMatrix, A::AbstractMatrix)
    return P * A
end


"""
    photometry_matrix(model, phot, θ)

Band fluxes `(nfilt, ncolumns)` of every design-matrix column at `θ`.
"""
function photometry_matrix(m::SpectralModel, phot, θ::AbstractVector)
    check_theta(m, θ)

    blocks = [phot_columns(c, phot, view(θ, r)) for (c, r) in zip(m.components, param_ranges(m))]
    return reduce(hcat, blocks)
end


"""
    split_weights(model, w)

Per-component physical weights (polynomial ± pairs are combined).
"""
function split_weights(m::SpectralModel, w::AbstractVector)
    if length(w) != ncolumns(m)
        throw(ArgumentError("expected $(ncolumns(m)) weights, got $(length(w))"))
    end

    return [physical_weights(c, w[r]) for (c, r) in zip(m.components, column_ranges(m))]
end


# ---------------------------------------------------------------- parameter maps
"""
    blockwise(f, m, x)

Apply `f(component, x_component)` to each component's slice of `x` and concatenate the
results (used for the free-coordinate maps).
"""
function blockwise(f, m::SpectralModel, x::AbstractVector)
    parts = [f(c, x[r]) for (c, r) in zip(m.components, param_ranges(m))]
    return reduce(vcat, parts; init=Float64[])
end


function to_free(m::SpectralModel, θ::AbstractVector)
    check_theta(m, θ)
    return blockwise(to_free, m, θ)
end


function from_free(m::SpectralModel, u::AbstractVector)
    check_theta(m, u)
    return blockwise(from_free, m, u)
end


function in_bounds(m::SpectralModel, θ::AbstractVector)
    check_theta(m, θ)

    for (c, r) in zip(m.components, param_ranges(m))
        if !in_bounds(c, θ[r])
            return false
        end
    end
    return true
end


# ---------------------------------------------------------------- Bayesian plumbing
"""
    model_components(model)

The model's components, in parameter and column order.
"""
function model_components(m::SpectralModel)
    return m.components
end


"""
    log_jacobian(model, u)

log |∂θ/∂u| of the map from free (optimiser/sampler) coordinates to parameters.
"""
function log_jacobian(m::SpectralModel, u::AbstractVector)
    check_theta(m, u)

    total = 0.0
    for (c, r) in zip(m.components, param_ranges(m))
        total += log_jacobian(c, view(u, r))
    end
    return total
end
