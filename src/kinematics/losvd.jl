"""
    gaussian_losvd_ft(freqs, shift, σ)

Fourier transform of a Gaussian LOSVD of width `σ` [pixels] displaced by `shift`
[pixels], at frequencies `freqs` [cycles/pixel] (as from `rfftfreq`). Analytic, so
there is no kernel sampling or truncation. Multiplying a template's FFT by this
yields `T(x - shift)` smoothed by the Gaussian.
"""
function gaussian_losvd_ft(freqs::AbstractVector{<:Real}, shift::Real, σ::Real)
    if σ < 0
        throw(ArgumentError("σ must be ≥ 0"))
    end

    return gaussian_losvd_ft_var(freqs, shift, σ^2)
end


"""
    gaussian_losvd_ft_var(freqs, shift, var)

`gaussian_losvd_ft` parameterised by the kernel variance `var` [pixels²] (must be ≥ 0).
Using the variance keeps derivatives finite when the width goes to zero.
"""
function gaussian_losvd_ft_var(freqs::AbstractVector{<:Real}, shift::Real, var::Real)
    broadening = @. exp(-2π^2 * var * freqs^2)
    displacement = @. cis(-2π * freqs * shift)
    return broadening .* displacement
end
