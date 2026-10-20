# Gaussian emission lines. Lines sharing a fixed flux ratio form a Multiplet (one
# linear amplitude); Multiplets sharing (v, σ) form a GasGroup (two nonlinear
# parameters). Amplitudes are integrated line fluxes, in data flux-density × Å.
#
# Frames: `dwave` and the rest wavelengths are rest-frame vacuum Å (data wave divided
# by 1 + z_sys). Build the LSF from observed wave/FWHM: σ in km/s is frame independent.

# ---------------------------------------------------------------- line definitions
"""
    Multiplet(name, wave)
    Multiplet(name, waves, ratios)

Lines with fixed relative fluxes `ratios` that share one amplitude (the flux of a
line whose ratio is 1). A singlet has one line.
"""
struct Multiplet
    name::String
    waves::Vector{Float64}
    ratios::Vector{Float64}

    function Multiplet(name, waves, ratios)
        if length(waves) != length(ratios)
            throw(ArgumentError("waves and ratios must have equal length"))
        end
        if isempty(waves)
            throw(ArgumentError("a multiplet needs at least one line"))
        end
        if !all(>(0), waves) || !all(>(0), ratios)
            throw(ArgumentError("wavelengths and ratios must be positive"))
        end
        return new(String(name), Vector{Float64}(waves), Vector{Float64}(ratios))
    end
end


function Multiplet(name::AbstractString, wave::Real)
    return Multiplet(name, [wave], [1.0])
end


"""
    GasGroup(name, multiplets...)

Multiplets that share one velocity and one dispersion.
"""
struct GasGroup
    name::String
    multiplets::Vector{Multiplet}
end


function GasGroup(name::AbstractString, multiplets::Multiplet...)
    return GasGroup(String(name), collect(multiplets))
end


"""
    default_gas_groups()

Balmer, low-ionisation and high-ionisation groups with vacuum wavelengths. The
[OIII] (2.98) and [NII] (3.05) doublet ratios and the line list are starting points:
check them against your preferred reference and edit as needed.
"""
function default_gas_groups()
    balmer = GasGroup("balmer",
                      Multiplet("Halpha", 6564.614),
                      Multiplet("Hbeta", 4862.683),
                      Multiplet("Hgamma", 4341.692),
                      Multiplet("Hdelta", 4102.892))
    low_ionisation = GasGroup("low-ion",
                              Multiplet("[NII]", [6549.86, 6585.27], [1.0, 3.05]),
                              Multiplet("[SII]6718", 6718.29),
                              Multiplet("[SII]6733", 6732.67),
                              Multiplet("[OI]", [6365.536, 6302.046], [1.0, 3.0]),
                              Multiplet("[OII]3727", 3727.092),
                              Multiplet("[OII]3729", 3729.875))
    high_ionisation = GasGroup("high-ion",
                               Multiplet("[OIII]", [4960.295, 5008.240], [1.0, 2.98]),
                               Multiplet("[NeIII]", 3869.86))
    return [balmer, low_ionisation, high_ionisation]
end


# ---------------------------------------------------------------- range selection
"""
    restrict(multiplet, lo, hi)
    restrict(group, lo, hi)

Keep only the lines within `[lo, hi]` Å. A multiplet with no lines left becomes
`nothing` (and is dropped from its group); a group with no lines left is an error.
"""
function restrict(m::Multiplet, lo::Real, hi::Real)
    keep = findall(w -> lo <= w <= hi, m.waves)
    if isempty(keep)
        return nothing
    end
    return Multiplet(m.name, m.waves[keep], m.ratios[keep])
end

function restrict(g::GasGroup, lo::Real, hi::Real)
    restricted = [restrict(m, lo, hi) for m in g.multiplets]
    kept = filter(!isnothing, restricted)
    if isempty(kept)
        throw(ArgumentError("group '$(g.name)' has no lines in [$lo, $hi] Å"))
    end
    return GasGroup(g.name, Vector{Multiplet}(kept))
end


# ---------------------------------------------------------------- component
"""
    GasComponent(dwave, groups, lsf=no broadening; σ_min=1, σ_max=1000, vmax=1000)

Emission lines on the data grid `dwave` (rest frame). θ = (v₁, σ₁, v₂, σ₂, …), one
pair per group, in km/s; σ is the intrinsic dispersion. Instrumental broadening
comes from `lsf`:

- `GaussianLSF`: added in quadrature at each line's wavelength;
- `BandedLSF`:   the resolution matrix is applied exactly to the intrinsic profiles
                 (it acts on the pixel flux-density vector).

Without `lsf` no instrumental broadening is applied: use this when a resolution matrix
in the model's `projection` (`pixel_projection(...; resolution=R)`) supplies it.

Lines outside the data are dropped; a group left empty is an error. `σ_min` must be > 0.
"""
struct GasComponent{T<:AbstractFloat,L<:LSF} <: Component
    groups::Vector{GasGroup}
    wave::Vector{T}
    lnedges::Vector{T}
    widths::Vector{T}      # pixel widths [Å]
    lsf::L
    vscale::T
    σ_min::T
    σ_max::T
    vmax::T
end


function GasComponent(dwave::AbstractVector{<:Real}, groups::AbstractVector{GasGroup},
                      lsf::LSF=GaussianLSF(zeros(float(eltype(dwave)), length(dwave)));
                      σ_min::Real=1.0, σ_max::Real=1000.0, vmax::Real=1000.0)
    if npix(lsf) != length(dwave)
        throw(ArgumentError("lsf has $(npix(lsf)) pixels, data has $(length(dwave))"))
    end
    if σ_min <= 0
        throw(ArgumentError("σ_min must be > 0"))
    end
    if σ_max <= σ_min
        throw(ArgumentError("σ_max = $σ_max must exceed σ_min = $σ_min"))
    end
    if vmax <= 0
        throw(ArgumentError("vmax must be positive"))
    end
    if isempty(groups)
        throw(ArgumentError("need at least one gas group"))
    end

    T = float(eltype(dwave))
    edges = log_pixel_edges(dwave)
    kept = [restrict(g, first(edges), last(edges)) for g in groups]
    lnedges = T.(log.(edges))
    widths = T.(diff(edges))
    vscale = T(median_velscale(dwave))
    return GasComponent{T,typeof(lsf)}(kept, Vector{T}(dwave), lnedges, widths, lsf, vscale,
                                       T(σ_min), T(σ_max), T(vmax))
end


"""
    ngroups(c::GasComponent)

Return the number of gas groups (each has its own `(v, σ)`).
"""
function ngroups(c::GasComponent)
    return length(c.groups)
end


function nparams(c::GasComponent)
    return 2 * ngroups(c)
end


function ncolumns(c::GasComponent)
    return sum(g -> length(g.multiplets), c.groups)
end


"""
    line_names(c::GasComponent)

Multiplet names in column order (after dropping lines outside the data).
"""
function line_names(c::GasComponent)
    return [m.name for g in c.groups for m in g.multiplets]
end


"""
    group_names(c::GasComponent)

Group names in parameter order: group `g` has θ[2g-1] = v, θ[2g] = σ.
"""
function group_names(c::GasComponent)
    return [g.name for g in c.groups]
end


function check_component(c::GasComponent, npix::Integer)
    if length(c.widths) != npix
        throw(ArgumentError("gas component has $(length(c.widths)) pixels, model has $npix"))
    end
    return nothing
end


# ---------------------------------------------------------------- line profiles
"""
    pixel_window(lnedges, μ, s; nsig=8)

Pixels within ±`nsig`σ of `μ` (all in ln λ); empty if the line is off the grid.
"""
function pixel_window(lnedges::AbstractVector, μ::Real, s::Real; nsig::Real=8)
    lo = searchsortedlast(lnedges, μ - nsig * s)
    hi = searchsortedfirst(lnedges, μ + nsig * s) - 1
    first_pixel = max(lo, 1)
    last_pixel = min(hi, length(lnedges) - 1)
    return first_pixel:last_pixel
end


"""
    total_sigma(c, λ, σ)

Observed dispersion [km/s] of a line at `λ`: intrinsic ⊕ instrumental for a Gaussian
LSF; just the intrinsic σ for a banded LSF (applied to the profiles afterwards).
"""
function total_sigma(c::GasComponent{T,<:GaussianLSF}, λ::Real, σ::Real) where {T}
    σ_instrument = interp_linear(λ, c.wave, c.lsf.sigma_kms)
    return hypot(σ, σ_instrument)
end

function total_sigma(c::GasComponent{T,<:BandedLSF}, λ::Real, σ::Real) where {T}
    return σ
end


"""
    add_line!(col, c, λ0, ratio, v, σ)

Add `ratio ×` a unit-flux, pixel-integrated Gaussian (in ln λ, i.e. velocity) at rest
wavelength `λ0` to `col`.
"""
function add_line!(col::AbstractVector, c::GasComponent, λ0::Real, ratio::Real, v::Real, σ::Real)
    μ = log(λ0) + v / C_KMS
    λ_observed = λ0 * exp(v / C_KMS)
    s = total_sigma(c, λ_observed, σ) / C_KMS

    for i in pixel_window(c.lnedges, μ, s)
        upper = gauss_cdf((c.lnedges[i+1] - μ) / s)
        lower = gauss_cdf((c.lnedges[i] - μ) / s)
        mass = upper - lower
        col[i] += ratio * mass / c.widths[i]
    end
    return col
end


"""
    add_multiplet!(col, c, m, v, σ)

Add every line of multiplet `m`, with its ratio, to `col`.
"""
function add_multiplet!(col::AbstractVector, c::GasComponent, m::Multiplet, v::Real, σ::Real)
    for (λ0, ratio) in zip(m.waves, m.ratios)
        add_line!(col, c, λ0, ratio, v, σ)
    end
    return col
end


"""
    finish_columns(c, cols)

Apply the instrumental resolution matrix to the line columns (banded LSF only).
"""
function finish_columns(c::GasComponent{T,<:GaussianLSF}, cols::AbstractMatrix) where {T}
    return cols
end

function finish_columns(c::GasComponent{T,<:BandedLSF}, cols::AbstractMatrix) where {T}
    return apply_banded(c.lsf, cols)
end


function columns(c::GasComponent, grid::KinematicsGrid, θ::AbstractVector)
    check_nparams(c, θ)

    element_type = promote_type(eltype(c.widths), eltype(θ))
    cols = zeros(element_type, length(c.widths), ncolumns(c))
    column = 0
    for (g, group) in enumerate(c.groups)
        v = θ[2g-1]
        σ = θ[2g]
        for m in group.multiplets
            column += 1
            add_multiplet!(view(cols, :, column), c, m, v, σ)
        end
    end
    return finish_columns(c, cols)
end


# ---------------------------------------------------------------- parameter maps
function in_bounds(c::GasComponent, θ::AbstractVector)
    check_nparams(c, θ)

    for g in 1:ngroups(c)
        v = θ[2g-1]
        σ = θ[2g]
        if !isfinite(v) || !isfinite(σ)
            return false
        end
        if abs(v) > c.vmax || σ < c.σ_min || σ > c.σ_max
            return false
        end
    end
    return true
end


function to_free(c::GasComponent, θ::AbstractVector)
    check_nparams(c, θ)
    if !all(>(c.σ_min), θ[2:2:end])
        throw(ArgumentError("initial σ must exceed σ_min = $(c.σ_min)"))
    end

    u = float.(collect(θ))
    u[1:2:end] ./= c.vscale
    u[2:2:end] .= log.(u[2:2:end] .- c.σ_min)
    return u
end


function from_free(c::GasComponent, u::AbstractVector)
    check_nparams(c, u)

    θ = float.(collect(u))
    θ[1:2:end] .*= c.vscale
    θ[2:2:end] .= c.σ_min .+ exp.(θ[2:2:end])
    return θ
end


# ---------------------------------------------------------------- photometry
function check_photometry(::GasComponent, phot)
    return nothing
end


"""
    phot_columns(c::GasComponent, phot, θ)

Band flux of each multiplet column: a line of integrated flux L contributes
L·T(λ)·λ / ∫Tλ dλ. Line widths are ignored (narrow against a filter); the dependence
on v enters through T at the shifted line wavelength.
"""
function phot_columns(c::GasComponent, phot, θ::AbstractVector)
    check_nparams(c, θ)

    element_type = promote_type(eltype(c.widths), eltype(θ))
    out = zeros(element_type, nphot(phot), ncolumns(c))
    column = 0
    for (g, group) in enumerate(c.groups)
        shift = exp(θ[2g-1] / C_KMS)
        for m in group.multiplets
            column += 1
            for (λ0, ratio) in zip(m.waves, m.ratios)
                λ_observed = λ0 * shift
                for (j, f) in enumerate(phot.rest)
                    out[j, column] += ratio * transmission(f, λ_observed) * λ_observed / f.norm
                end
            end
        end
    end
    return out
end


function log_jacobian(c::GasComponent, u::AbstractVector)
    # v = vscale·u, σ = σ_min + e^u for each group
    return ngroups(c) * log(c.vscale) + sum(u[2:2:end])
end
