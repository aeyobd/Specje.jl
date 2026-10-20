"""
    Photometry(filters, flux, err; z=0, good=trues, error_floor=0)

Broadband constraints: band-averaged f_λ `flux` ± `err` in the *same units as the
spectrum*, one per observer-frame filter. `z` is the systemic redshift used to
deredshift the data wavelengths; the filters are shifted accordingly (`rest`).
`error_floor` adds a fractional systematic floor in quadrature (`hypot(err, floor·|flux|)`).
Non-finite values or non-positive errors are flagged bad. Float64 only.
"""
struct Photometry
    filters::Vector{Filter{Float64}}
    rest::Vector{Filter{Float64}}
    flux::Vector{Float64}
    err::Vector{Float64}
    good::BitVector
    z::Float64
end


function Photometry(filters::AbstractVector{<:Filter}, flux::AbstractVector{<:Real},
                    err::AbstractVector{<:Real};
                    z::Real=0.0, good=trues(length(flux)), error_floor::Real=0.0)
    n = length(filters)
    if length(flux) != n || length(err) != n || length(good) != n
        throw(ArgumentError("filters, flux, err and good must have equal length"))
    end
    if error_floor < 0
        throw(ArgumentError("error_floor must be ≥ 0"))
    end

    floor_err = error_floor .* abs.(Float64.(flux))
    total_err = hypot.(Float64.(err), floor_err)
    usable = isfinite.(flux) .& isfinite.(total_err) .& (err .> 0)
    ok = BitVector(good) .& usable

    observed = Vector{Filter{Float64}}(filters)
    rest = [to_rest(f, z) for f in observed]
    return Photometry(observed, rest, Vector{Float64}(flux), total_err, ok, Float64(z))
end


"""
    nphot(p)

Return the number of photometric points (filters).
"""
function nphot(p::Photometry)
    return length(p.flux)
end


"""
    band_names(p)

Return the filter names, in order.
"""
function band_names(p::Photometry)
    return [f.name for f in p.filters]
end
