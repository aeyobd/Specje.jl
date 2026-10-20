using FFTW

@testset "gaussian_losvd_ft" begin
    n = 256
    f = rfftfreq(n)

    @testset "matches sampled Gaussian" begin
        g = irfft(gaussian_losvd_ft(f, 10.3, 3.0), n)
        x = 0:n-1
        # irfft gives the n-periodic Gaussian: the left tail wraps onto the last pixels
        G(x) = exp(-x^2 / (2 * 3.0^2)) / (3.0 * sqrt(2π))
        @test g ≈ [sum(G(xi - 10.3 - k * n) for k in -1:1) for xi in x] atol = 1e-10
    end

    @testset "integer shift, σ=0 is a delta" begin
        g = irfft(gaussian_losvd_ft(f, 5, 0.0), n)
        @test g[6] ≈ 1.0 atol = 1e-12
        @test maximum(abs, g[[1:5; 7:n]]) < 1e-12
    end

    @testset "unit DC gain" begin
        @test gaussian_losvd_ft(f, -3.7, 2.0)[1] ≈ 1.0
    end

    @testset "negative shift moves the other way" begin
        g = irfft(gaussian_losvd_ft(f, -4.0, 0.0), n)
        @test g[n - 4 + 1] ≈ 1.0 atol = 1e-12
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "gaussian_losvd_ft (cases)" begin
    n = 128
    f = rfftfreq(n)
    @testset "simple" begin
        @test gaussian_losvd_ft(f, 0.0, 0.0) == ones(ComplexF64, length(f))
    end
    @testset "invalid input" begin
        @test_throws ArgumentError gaussian_losvd_ft(f, 0.0, -1.0)
    end
    @testset "realistic" begin
        # σ = 200 km/s, v = 120 km/s at 50 km/s per pixel: moments of the kernel
        g = irfft(gaussian_losvd_ft(f, 120.0 / 50, 200.0 / 50), n)
        x = collect(0:n-1)
        x[x .> n ÷ 2] .-= n                                   # signed lags
        @test sum(g) ≈ 1.0
        @test sum(x .* g) ≈ 2.4 atol = 1e-8
        @test sum((x .- 2.4) .^ 2 .* g) ≈ 16.0 rtol = 1e-6
    end
    @testset "pathological" begin
        # a shift by the full period is no shift at all
        @test gaussian_losvd_ft(f, n, 2.0) ≈ gaussian_losvd_ft(f, 0.0, 2.0)
        # a kernel far wider than the array smooths everything to the mean
        g = irfft(gaussian_losvd_ft(f, 0.0, 1e4), n)
        @test all(≈(1 / n; rtol=1e-6), g)
        # a sub-pixel shift with σ = 0 is a band-limited (sinc) shift that keeps the sum
        @test sum(irfft(gaussian_losvd_ft(f, 0.5, 0.0), n)) ≈ 1.0
    end
end
