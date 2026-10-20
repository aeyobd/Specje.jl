"""
    pixel_edges(centers)

Return the pixel edges from a vector of pixel centres.

Use the midpoints, calculating the outermost edges by
using the difference between the outermost centres and midpoints.
This is exact for uniformly spaced pixels.
"""
function pixel_edges(x::AbstractVector{<:Real})
    if length(x) < 2
        throw(ArgumentError("need at least two pixel centres"))
    end
    if !issorted(x)
        throw(ArgumentError("centre array must be sorted"))
    end

    mid = (x[begin:end-1] .+ x[begin+1:end]) ./ 2
    lo = x[begin] - (mid[begin] - x[begin])
    hi = x[end] + (x[end] - mid[end])
    return [lo; mid; hi]
end


"""
    log_pixel_edges(centres)

Return the pixel edges from a vector of pixel centres when spaced log-uniformly.
"""
function log_pixel_edges(x::AbstractVector{<:Real})
    if any(<=(0), x)
        throw(ArgumentError("log-spaced centres must be positive"))
    end

    return exp.(pixel_edges(log.(x)))
end


"""
    nlog(λmin, λmax, dlnλ)

Compute the number of log-spaced pixels centres which fit between λmin and λmax and stepsize `dlnλ`.
"""
function nlog(λmin::Real, λmax::Real, dlnλ::Real)
    return floor(Int, log(λmax / λmin) / dlnλ + 1e-9) + 1
end


"""
    loggrid(λmin, λmax, velscale)

Calculate the pixel centres of a log-uniform grid starting at `λmin`, stepping by
`velscale` km/s per pixel, and ending before or equal to `λmax`.
"""
function loggrid(λmin::Real, λmax::Real, velscale::Real)
    if λmin <= 0
        throw(ArgumentError("λmin must be positive"))
    end
    if λmax < λmin
        throw(ArgumentError("λmax = $λmax must be ≥ λmin = $λmin"))
    end
    if velscale <= 0
        throw(ArgumentError("velscale must be positive"))
    end

    dlnλ = velscale / C_KMS
    n = nlog(λmin, λmax, dlnλ)
    return λmin .* exp.(dlnλ .* (0:n-1))
end
