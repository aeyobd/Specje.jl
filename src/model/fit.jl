# Variable projection: for each θ the optimal non-negative weights come from NNLS,
# so only the few nonlinear parameters are searched (as in pPXF).

"""
    FitData(spectrum)
    FitData(spectrum, photometry)
    FitData(b, inv_noise, idx[, blocks])

Noise-weighted data on the good rows (spectral pixels, then any photometric points):
`b = value / error`, `inv_noise = 1 / error`, `idx` the good rows of the stacked design
matrix, and `blocks` the rows of `b` belonging to each data block (spectrum, photometry).
"""
struct FitData{T<:AbstractFloat}
    b::Vector{T}          # value / error
    inv_noise::Vector{T}
    idx::Vector{Int}      # good rows of the stacked design matrix
    blocks::Vector{UnitRange{Int}}   # rows of `b` per data block (spectrum, photometry)

    function FitData(b::Vector{T}, inv_noise::Vector{T}, idx::Vector{Int},
                     blocks::Vector{UnitRange{Int}}) where {T<:AbstractFloat}
        if length(inv_noise) != length(b) || length(idx) != length(b)
            throw(ArgumentError("b, inv_noise and idx must have equal length"))
        end
        if sum(length, blocks; init=0) != length(b)
            throw(ArgumentError("the data blocks must cover every row of b exactly once"))
        end
        return new{T}(b, inv_noise, idx, blocks)
    end
end


function FitData(b::Vector{<:AbstractFloat}, inv_noise::Vector{<:AbstractFloat}, idx::Vector{Int})
    return FitData(b, inv_noise, idx, [1:length(b)])
end

function FitData(s::Spectrum)
    idx = findall(s.good)
    if isempty(idx)
        throw(ArgumentError("no good pixels"))
    end

    inv_noise = 1 ./ s.noise[idx]
    b = s.flux[idx] .* inv_noise
    return FitData(b, inv_noise, idx)
end


"""
    check_compatible(model, spectrum)

Throw an `ArgumentError` unless the model predicts as many pixels as the spectrum has.
"""
function check_compatible(m::SpectralModel, s::Spectrum)
    if npix(m) != npix(s)
        throw(ArgumentError("model has $(npix(m)) pixels, spectrum has $(npix(s))"))
    end
    return nothing
end


"""
    solve_weights(A, fd) -> (w, χ²)

NNLS weights and χ² for design matrix `A` against the data `fd`.
"""
function solve_weights(A::AbstractMatrix, fd::FitData)
    if size(A, 1) < maximum(fd.idx; init=0)
        throw(DimensionMismatch("design matrix has $(size(A, 1)) rows, the data use row $(maximum(fd.idx))"))
    end

    Aw = A[fd.idx, :] .* fd.inv_noise
    w = nnls(Aw, fd.b)
    residual = fd.b .- Aw * w
    chi2 = sum(abs2, residual)
    return w, chi2
end


"""
    profile_chi2(model, fd, θ)

χ² at `θ` with weights profiled out; `Inf` outside the parameter bounds.
"""
function profile_chi2(m, fd::FitData, θ::AbstractVector)
    if !in_bounds(m, θ)
        return Inf
    end

    A = design_matrix(m, θ)
    _, chi2 = solve_weights(A, fd)
    return chi2
end


# ---------------------------------------------------------------- result
"""
    FitResult

Result of `fit_kinematics`: best-fit `θ`, NNLS `weights`, `chi2`, the number of good
data points `ndata`, the number of non-zero weights `nactive`, and the optimiser's
`converged` flag and `iterations`.
"""
struct FitResult{T<:AbstractFloat}
    θ::Vector{T}
    weights::Vector{T}
    chi2::T
    ndata::Int
    nactive::Int          # templates/columns with non-zero weight
    converged::Bool
    iterations::Int
end


"""
    model_flux(model, result)

Best-fit model flux on all spectral pixels (including masked ones).
"""
function model_flux(m::SpectralModel, r::FitResult)
    A = design_matrix(m, r.θ)
    return A * r.weights
end


"""
    reduced_chi2(result)

χ² per good data point.
"""
function reduced_chi2(r::FitResult)
    return r.chi2 / r.ndata
end


# ---------------------------------------------------------------- driver
"""
    fit_kinematics(model, spectrum, θ0; iterations=2000, tol=1e-6)
    fit_kinematics(model, fitdata, θ0; ...)

Nelder–Mead over the nonlinear parameters, NNLS for the weights at every step.
`θ0` is the starting point in the model's parameter layout. This is a local
optimiser: try several `θ0` if the problem is multimodal. `model` may also be a
`JointModel` (spectrum + photometry).
"""
function fit_kinematics(m::SpectralModel, s::Spectrum, θ0::AbstractVector; kw...)
    check_compatible(m, s)
    return fit_kinematics(m, FitData(s), θ0; kw...)
end


"""
    nelder_mead()

Nelder–Mead with fixed additive initial steps of 0.5 in the free coordinates. Optim's
default simplex scales each coordinate by 1.5, which on log(σ - σ_min) can jump past
σ_max; the objective is then Inf, the simplex spread NaN, and Optim stops at iteration 0.
"""
function nelder_mead()
    simplex = Optim.AffineSimplexer(a=0.5, b=0.0)
    return Optim.NelderMead(initial_simplex=simplex)
end


function fit_kinematics(m, fd::FitData, θ0::AbstractVector; iterations::Integer=2000, tol::Real=1e-6)
    check_theta(m, θ0)
    if iterations < 1
        throw(ArgumentError("iterations must be ≥ 1"))
    end
    if tol <= 0
        throw(ArgumentError("tol must be positive"))
    end

    # No nonlinear parameters: the weights are solved exactly, there is nothing to search.
    if nparams(m) == 0
        θ = Float64[]
        w, chi2 = solve_weights(design_matrix(m, θ), fd)
        return FitResult(θ, w, chi2, length(fd.b), count(>(0), w), true, 0)
    end

    objective(u) = profile_chi2(m, fd, from_free(m, u))
    options = Optim.Options(iterations=iterations, g_abstol=tol)
    result = Optim.optimize(objective, to_free(m, θ0), nelder_mead(), options)

    θ = from_free(m, Optim.minimizer(result))
    w, chi2 = solve_weights(design_matrix(m, θ), fd)
    nactive = count(>(0), w)
    return FitResult(θ, w, chi2, length(fd.b), nactive,
                     Optim.converged(result), Optim.iterations(result))
end
