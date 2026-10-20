# A Component contributes (i) nonlinear parameters θ_c and (ii) columns of the design
# matrix. Every column is constrained ≥ 0 by the solver; signed terms are therefore
# supplied as ± pairs (see PolynomialComponent). GasComponent (gas.jl) implements the
# same interface.
#
# Required:  ncolumns(c), columns(c, grid, θ)
# Optional:  nparams(c)=0, in_bounds(c, θ)=true, to_free/from_free (optimiser
#            coordinates), physical_weights(c, w), check_component(c, npix)

abstract type Component end


"""
    nparams(component)

Return the number of nonlinear parameters of a component (0 unless it defines its own).
"""
function nparams(::Component)
    return 0
end


"""
    in_bounds(component, θ)

`true` if the parameters `θ` lie within the component's bounds (always, by default).
"""
function in_bounds(::Component, θ::AbstractVector)
    return true
end


"""
    to_free(component, θ)

Map physical parameters to unconstrained optimiser/sampler coordinates (identity by default).
"""
function to_free(::Component, θ::AbstractVector)
    return collect(float.(θ))
end


"""
    from_free(component, u)

Map free coordinates back to physical parameters; the inverse of `to_free`.
"""
function from_free(::Component, u::AbstractVector)
    return collect(float.(u))
end


"""
    physical_weights(component, w)

Turn a component's (non-negative) solver weights into physical ones (unchanged by default).
"""
function physical_weights(::Component, w::AbstractVector)
    return w
end


"""
    check_component(component, npix)

Throw an `ArgumentError` if the component does not fit a model grid of `npix` pixels
(nothing to check by default).
"""
function check_component(::Component, npix::Integer)
    return nothing
end


"""
    check_nparams(c, θ)

Throw an `ArgumentError` unless `θ` has `nparams(c)` entries.
"""
function check_nparams(c::Component, θ::AbstractVector)
    if length(θ) != nparams(c)
        throw(ArgumentError("$(nameof(typeof(c))) has $(nparams(c)) parameters, got $(length(θ))"))
    end
    return nothing
end


# ---------------------------------------------------------------- stellar
"""
    StellarComponent(grid, templates; σ_min=0, σ_max=1000, vmax=2000, σ_templ=0, phot=nothing)

A block of templates sharing one Gaussian LOSVD, θ = (v, σ) in km/s. `templates` must
be on the template grid that `grid` was built for. `phot` (a `BandTemplates` from
`band_templates`) is needed only when fitting photometry. The optimiser sees
`(v / velscale, log(σ - σ_min))`, which keeps σ in `[σ_min, ∞)`; `σ_max` and `vmax`
reject wild steps (keep `vmax + 5σ_max` below the grid's `margin_kms`).

`σ_templ` [km/s] declares that the templates are already broadened by a Gaussian of that
(constant) width, so the kernel applied is `√(σ² − σ_templ²)` and the fitted σ stays the
full LOSVD width. Use it when the instrument is applied after the convolution (a
resolution-matrix projection) and the templates have been brought to a constant
velocity resolution; it requires `σ_min ≥ σ_templ`.
"""
struct StellarComponent{T<:AbstractFloat} <: Component
    ft::Matrix{Complex{T}}
    vscale::T
    σ_min::T
    σ_max::T
    vmax::T
    σ_templ::T
    phot::Union{Nothing,BandTemplates{T}}
end


function StellarComponent(grid::KinematicsGrid{T}, templates::AbstractMatrix{<:Real};
                          σ_min::Real=0.0, σ_max::Real=1000.0, vmax::Real=2000.0,
                          σ_templ::Real=0.0, phot=nothing) where {T}
    if σ_templ < 0
        throw(ArgumentError("σ_templ must be ≥ 0"))
    end
    if σ_min < σ_templ
        throw(ArgumentError("σ_min = $σ_min must be ≥ σ_templ = $σ_templ: the templates are already that broad"))
    end
    if σ_min < 0 || σ_max <= σ_min
        throw(ArgumentError("need 0 ≤ σ_min < σ_max, got σ_min = $σ_min, σ_max = $σ_max"))
    end
    if vmax <= 0
        throw(ArgumentError("vmax must be positive"))
    end

    ft = transform_templates(grid, Matrix{T}(templates))
    bands = convert_bands(T, phot)
    check_band_columns(bands, size(ft, 2))
    return StellarComponent{T}(ft, grid.velscale, T(σ_min), T(σ_max), T(vmax), T(σ_templ), bands)
end


function nparams(::StellarComponent)
    return 2
end


"""
    ncolumns(component)

Return the number of design-matrix columns of a component.
"""
function ncolumns(c::StellarComponent)
    return size(c.ft, 2)
end


"""
    columns(component, grid, θ)

The component's design-matrix columns on the model grid, `(nmodel, ncolumns)`, at
parameters `θ`.
"""
function columns(c::StellarComponent, grid::KinematicsGrid, θ::AbstractVector)
    check_nparams(c, θ)
    v = θ[1]
    σ = θ[2]
    return convolve_templates(grid, c.ft, v, σ; σ_templ=c.σ_templ)
end


function in_bounds(c::StellarComponent, θ::AbstractVector)
    check_nparams(c, θ)
    v = θ[1]
    σ = θ[2]
    if !isfinite(v) || !isfinite(σ)
        return false
    end
    return abs(v) <= c.vmax && c.σ_min <= σ <= c.σ_max
end


function to_free(c::StellarComponent, θ::AbstractVector)
    check_nparams(c, θ)
    v = θ[1]
    σ = θ[2]
    if σ <= c.σ_min
        throw(ArgumentError("initial σ must exceed σ_min = $(c.σ_min)"))
    end
    return [v / c.vscale, log(σ - c.σ_min)]
end


function from_free(c::StellarComponent, u::AbstractVector)
    check_nparams(c, u)
    v = u[1] * c.vscale
    σ = c.σ_min + exp(u[2])
    return [v, σ]
end


function check_component(c::StellarComponent, npix::Integer)
    return nothing
end


# ---------------------------------------------------------------- polynomial
"""
    legendre_basis(npix, degree; T=Float64)

Legendre polynomials P₀…P_degree sampled uniformly on [-1, 1], as `(npix, degree+1)`.
"""
function legendre_basis(npix::Integer, degree::Integer; T::Type=Float64)
    if npix < 2
        throw(ArgumentError("need at least two pixels"))
    end
    if degree < 0
        throw(ArgumentError("degree must be ≥ 0"))
    end

    x = collect(range(-one(T), one(T); length=npix))
    P = Matrix{T}(undef, npix, degree + 1)
    P[:, 1] .= 1
    if degree >= 1
        P[:, 2] .= x
    end
    # Bonnet's recursion: k P_k = (2k − 1) x P_{k−1} − (k − 1) P_{k−2}
    for k in 2:degree
        @views P[:, k + 1] .= ((2k - 1) .* x .* P[:, k] .- (k - 1) .* P[:, k - 1]) ./ k
    end
    return P
end


"""
    PolynomialComponent(npix, degree)

Additive Legendre polynomial with *signed* coefficients, implemented as the
non-negative column pairs `[P  -P]`; `physical_weights` returns `w₊ - w₋`.
"""
struct PolynomialComponent{T<:AbstractFloat} <: Component
    cols::Matrix{T}
end


function PolynomialComponent(npix::Integer, degree::Integer)
    if degree < 0
        throw(ArgumentError("degree must be ≥ 0"))
    end

    P = legendre_basis(npix, degree)
    return PolynomialComponent(hcat(P, -P))
end


function ncolumns(c::PolynomialComponent)
    return size(c.cols, 2)
end


function columns(c::PolynomialComponent, grid::KinematicsGrid, θ::AbstractVector)
    return c.cols
end


function physical_weights(c::PolynomialComponent, w::AbstractVector)
    if length(w) != ncolumns(c)
        throw(ArgumentError("expected $(ncolumns(c)) weights, got $(length(w))"))
    end

    n = length(w) ÷ 2
    positive = w[1:n]
    negative = w[n+1:end]
    return positive .- negative
end


function check_component(c::PolynomialComponent, npix::Integer)
    if size(c.cols, 1) != npix
        throw(ArgumentError("polynomial has $(size(c.cols, 1)) pixels, model has $npix"))
    end
    return nothing
end


# ---------------------------------------------------------------- photometry interface
# phot_columns(c, phot, θ) -> (nfilt, ncolumns): band fluxes of each column, in the
# same order and units as columns(c, grid, θ). check_photometry validates at set-up.

"""
    check_photometry(component, phot)

Throw an `ArgumentError` unless the component can predict the photometry `phot`
(components without photometry support always throw).
"""
function check_photometry(c::Component, phot)
    throw(ArgumentError("$(nameof(typeof(c))) does not support photometry"))
end

function check_photometry(c::StellarComponent, phot)
    if isnothing(c.phot)
        throw(ArgumentError("stellar component has no band templates; pass phot=band_templates(...)"))
    end
    if c.phot.names != band_names(phot)
        throw(ArgumentError("band templates are for $(c.phot.names), photometry has $(band_names(phot))"))
    end
    return nothing
end


"""
    phot_columns(component, phot, θ)

Band fluxes `(nfilt, ncolumns)` of each of the component's columns at parameters `θ`.
"""
function phot_columns(c::StellarComponent, phot, θ::AbstractVector)
    return c.phot.M
end


# The additive polynomial is a spectral-calibration nuisance: it adds nothing to photometry.
function check_photometry(::PolynomialComponent, phot)
    return nothing
end

function phot_columns(c::PolynomialComponent, phot, θ::AbstractVector)
    return zeros(eltype(c.cols), nphot(phot), size(c.cols, 2))
end


# ---------------------------------------------------------------- change of variables
# log |∂θ_c/∂u_c| for the map u -> θ implemented by from_free; needed to put priors
# defined on physical parameters into free (sampler) coordinates.

"""
    log_jacobian(component, u)

`log |∂θ/∂u|` of the component's `from_free` map at free coordinates `u` (0 by default).
"""
function log_jacobian(::Component, u::AbstractVector)
    return 0.0
end

function log_jacobian(c::StellarComponent, u::AbstractVector)
    # v = vscale·u₁, σ = σ_min + e^{u₂}
    return log(c.vscale) + u[2]
end
