"""
    JointModel(model, phot; fit_scale=false, s_min=1e-3, s_max=1e3)

A `SpectralModel` fitted to a spectrum *and* broadband photometry. The stacked design
matrix is `[s·A_spec(θ); A_phot(θ)]`: spectral rows first, then one row per filter.
With `fit_scale=true` an extra nonlinear parameter `s` (last entry of θ, positive)
absorbs the flux calibration/aperture mismatch between spectrum and photometry:
the data spectrum is modelled as `s ×` the photometric-scale model.

Every component must support photometry (`check_photometry`): stellar components need
`phot=band_templates(...)`; gas lines are integrated through the filters; the additive
polynomial contributes nothing to photometry.
"""
struct JointModel{M<:SpectralModel}
    model::M
    phot::Photometry
    fit_scale::Bool
    s_min::Float64
    s_max::Float64

    function JointModel(model::M, phot::Photometry; fit_scale::Bool=false,
                        s_min::Real=1e-3, s_max::Real=1e3) where {M<:SpectralModel}
        for c in model.components
            check_photometry(c, phot)
        end
        if !(0 < s_min < s_max)
            throw(ArgumentError("need 0 < s_min < s_max"))
        end
        return new{M}(model, phot, fit_scale, Float64(s_min), Float64(s_max))
    end
end


# ---------------------------------------------------------------- bookkeeping
function npix(jm::JointModel)
    return npix(jm.model)
end


function nparams(jm::JointModel)
    nscale = jm.fit_scale ? 1 : 0
    return nparams(jm.model) + nscale
end


function ncolumns(jm::JointModel)
    return ncolumns(jm.model)
end


function split_weights(jm::JointModel, w::AbstractVector)
    return split_weights(jm.model, w)
end


"""
    split_theta(jm, θ) -> (θ_model, s)

The spectral model's parameters and the spectral scale (1 when not fitted).
"""
function split_theta(jm::JointModel, θ::AbstractVector)
    if jm.fit_scale
        return θ[1:end-1], θ[end]
    else
        return θ, one(eltype(θ))
    end
end


# ---------------------------------------------------------------- design matrix
"""
    design_matrix(jm::JointModel, θ)

Stacked design matrix: spectral rows (scaled by `s`) then photometric rows.
"""
function design_matrix(jm::JointModel, θ::AbstractVector)
    check_theta(jm, θ)

    θm, s = split_theta(jm, θ)
    A_spectrum = design_matrix(jm.model, θm)
    if jm.fit_scale
        A_spectrum = A_spectrum .* s
    end
    A_photometry = photometry_matrix(jm.model, jm.phot, θm)
    return vcat(A_spectrum, A_photometry)
end


# ---------------------------------------------------------------- parameter maps
function to_free(jm::JointModel, θ::AbstractVector)
    check_theta(jm, θ)

    θm, s = split_theta(jm, θ)
    if !jm.fit_scale
        return to_free(jm.model, θm)
    end
    if s <= 0
        throw(ArgumentError("initial scale must be positive"))
    end
    return vcat(to_free(jm.model, θm), log(s))
end


function from_free(jm::JointModel, u::AbstractVector)
    check_theta(jm, u)

    if !jm.fit_scale
        return from_free(jm.model, u)
    end
    θm = from_free(jm.model, u[1:end-1])
    s = exp(u[end])
    return vcat(θm, s)
end


function in_bounds(jm::JointModel, θ::AbstractVector)
    check_theta(jm, θ)

    θm, s = split_theta(jm, θ)
    if !in_bounds(jm.model, θm)
        return false
    end
    if jm.fit_scale && !(jm.s_min <= s <= jm.s_max)
        return false
    end
    return true
end


# ---------------------------------------------------------------- data and fitting
"""
    FitData(spectrum, photometry)

Stack the spectrum's good pixels and the photometry's good points, as two data blocks.
"""
function FitData(s::Spectrum, p::Photometry)
    spectral = FitData(s)
    idx = findall(p.good)
    inv_noise = 1 ./ p.err[idx]

    nspec = length(spectral.b)
    nphot_good = length(idx)
    b = vcat(spectral.b, p.flux[idx] .* inv_noise)
    all_inv_noise = vcat(spectral.inv_noise, inv_noise)
    rows = vcat(spectral.idx, idx .+ npix(s))
    blocks = filter(!isempty, [1:nspec, nspec+1:nspec+nphot_good])
    return FitData(b, all_inv_noise, rows, blocks)
end


function fit_kinematics(jm::JointModel, s::Spectrum, θ0::AbstractVector; kw...)
    check_compatible(jm.model, s)
    return fit_kinematics(jm, FitData(s, jm.phot), θ0; kw...)
end


# ---------------------------------------------------------------- results
"""
    spectral_scale(jm, result)

Spectral scale `s` of a fit (1 when not fitted).
"""
function spectral_scale(jm::JointModel, r::FitResult)
    _, s = split_theta(jm, r.θ)
    return s
end


"""
    model_spectrum(jm, result)

Best-fit model spectrum on the photometric scale (the data spectrum is `s ×` this).
"""
function model_spectrum(jm::JointModel, r::FitResult)
    θm, _ = split_theta(jm, r.θ)
    return design_matrix(jm.model, θm) * r.weights
end


"""
    model_photometry(jm, result)

Best-fit model band fluxes, one per filter.
"""
function model_photometry(jm::JointModel, r::FitResult)
    θm, _ = split_theta(jm, r.θ)
    return photometry_matrix(jm.model, jm.phot, θm) * r.weights
end


function model_flux(jm::JointModel, r::FitResult)
    return model_spectrum(jm, r)
end


# ---------------------------------------------------------------- Bayesian plumbing
function model_components(jm::JointModel)
    return model_components(jm.model)
end


function column_ranges(jm::JointModel)
    return column_ranges(jm.model)
end


function log_jacobian(jm::JointModel, u::AbstractVector)
    check_theta(jm, u)

    if !jm.fit_scale
        return log_jacobian(jm.model, u)
    end
    # the calibration scale is s = exp(u_end)
    return log_jacobian(jm.model, u[1:end-1]) + u[end]
end
