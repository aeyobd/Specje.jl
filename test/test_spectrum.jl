@testset "Spectrum" begin
    w = collect(4000.0:1.0:4009.0)
    f = fill(1.0, 10)
    σ = fill(0.1, 10)

    @testset "construction" begin
        s = Spectrum(w, f, σ)
        @test npix(s) == 10
        @test all(s.good)
        @test s.lsf_fwhm === nothing
        @test Spectrum(Int.(w), f, σ) isa Spectrum{Float64}
    end

    @testset "automatic masking" begin
        f2 = copy(f); f2[3] = NaN
        σ2 = copy(σ); σ2[5] = 0.0
        s = Spectrum(w, f2, σ2)
        @test !s.good[3] && !s.good[5]
        @test count(s.good) == 8
        g = trues(10); g[1] = false
        @test !Spectrum(w, f, σ; good=g).good[1]
    end

    @testset "validation" begin
        @test_throws ArgumentError Spectrum(w, f[1:end-1], σ)
        @test_throws ArgumentError Spectrum(reverse(w), f, σ)
        @test_throws ArgumentError Spectrum(w, f, σ; lsf_fwhm=ones(3))
    end

    @testset "crop" begin
        s = Spectrum(w, f, σ; lsf_fwhm=fill(2.5, 10))
        c = crop(s, 4002.0, 4005.0)
        @test c.wave == [4002.0, 4003.0, 4004.0, 4005.0]
        @test length(c.lsf_fwhm) == 4
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "Spectrum (cases)" begin
    @testset "simple" begin
        s = Spectrum([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        @test s.flux == [1.0, 2.0, 3.0]
        @test s.good == trues(3)
    end

    @testset "invalid input" begin
        w = [1.0, 2.0, 3.0]
        @test_throws ArgumentError Spectrum(w, ones(3), ones(2))
        @test_throws ArgumentError Spectrum(w, ones(3), ones(3); good=trues(2))
        @test_throws ArgumentError Spectrum([1.0, 1.0, 2.0], ones(3), ones(3))
        @test_throws ArgumentError Spectrum([3.0, 2.0, 1.0], ones(3), ones(3))
    end

    @testset "realistic" begin
        # SDSS-like: log-uniform 3600–9000 Å at 69 km/s, with sky-line masks and a few bad pixels
        wave = loggrid(3600.0, 9000.0, 69.0)
        n = length(wave)
        rng = Xoshiro(1)
        flux = 1 .+ 0.1 .* randn(rng, n)
        noise = fill(0.05, n)
        flux[[100, 2000]] .= NaN
        noise[300] = Inf
        sky = @. abs(wave - 5578.5) < 5 || abs(wave - 6301.7) < 5
        s = Spectrum(wave, flux, noise; good=.!sky, lsf_fwhm=fill(2.5, n))
        @test npix(s) == n
        @test !s.good[100] && !s.good[2000] && !s.good[300]
        @test count(!, s.good) == 3 + count(sky)
        @test velscale(s) ≈ 69.0 rtol = 1e-10
    end

    @testset "pathological" begin
        # every pixel unusable: the spectrum exists, but there is nothing to fit
        s = Spectrum([1.0, 2.0], [NaN, NaN], [1.0, 1.0])
        @test !any(s.good)
        @test_throws ArgumentError FitData(s)
        # negative noise is masked, not an error
        @test !Spectrum([1.0, 2.0], [1.0, 1.0], [-1.0, 1.0]).good[1]
        # a single pixel is a valid spectrum
        @test npix(Spectrum([5000.0], [1.0], [0.1])) == 1
        # element type follows the inputs
        @test Spectrum(Float32[1, 2], Float32[1, 1], Float32[1, 1]) isa Spectrum{Float32}
    end
end

@testset "velscale (cases)" begin
    @testset "simple" begin
        @test velscale([1.0, exp(1.0)]) ≈ C_KMS
    end
    @testset "invalid input" begin
        @test_throws ArgumentError velscale([5000.0])
        @test_throws ArgumentError velscale([-1.0, 2.0])
    end
    @testset "realistic" begin
        @test velscale(loggrid(3600.0, 10400.0, 69.0)) ≈ 69.0 rtol = 1e-10
    end
    @testset "pathological" begin
        # on a linear grid velscale is only the *mean* step
        lin = collect(4000.0:1.0:8000.0)
        @test velscale(lin) ≈ C_KMS * log(2) / 4000 rtol = 1e-12
        @test !is_loguniform(lin)
    end
end

@testset "is_loguniform (cases)" begin
    @testset "simple" begin
        @test is_loguniform([1.0, 2.0, 4.0, 8.0])
        @test !is_loguniform([1.0, 2.0, 3.0, 4.0])
    end
    @testset "invalid input" begin
        @test_throws ArgumentError is_loguniform([1.0])
    end
    @testset "realistic" begin
        g = loggrid(3600.0, 9000.0, 69.0)
        @test is_loguniform(g)
        perturbed = copy(g); perturbed[1000] *= 1 + 1e-5
        @test !is_loguniform(perturbed)
    end
    @testset "pathological" begin
        @test is_loguniform([3.0, 7.0])               # two points are trivially uniform
        @test is_loguniform([1.0, 2.0, 4.0 * (1 + 1e-8)]) # within the default rtol
        @test !is_loguniform([1.0, 2.0, 4.0 * (1 + 1e-8)]; rtol=1e-10)
    end
end

@testset "crop (cases)" begin
    s = Spectrum(collect(4000.0:1.0:4009.0), collect(1.0:10.0), fill(0.1, 10);
                 lsf_fwhm=collect(1.0:10.0))
    @testset "simple" begin
        c = crop(s, 4001.5, 4003.5)
        @test c.wave == [4002.0, 4003.0]
        @test c.flux == [3.0, 4.0] && c.lsf_fwhm == [3.0, 4.0]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError crop(s, 4005.0, 4002.0)
        @test_throws ArgumentError crop(s, 5000.0, 6000.0)          # no pixels selected
    end
    @testset "realistic" begin
        g = trues(10); g[4] = false
        masked = Spectrum(s.wave, s.flux, s.noise; good=g)
        c = crop(masked, 4002.0, 4006.0)
        @test c.good == [true, false, true, true, true]
        @test isnothing(c.lsf_fwhm)
    end
    @testset "pathological" begin
        @test crop(s, 4003.0, 4003.0).wave == [4003.0]             # bounds are inclusive
        @test crop(s, -Inf, Inf).wave == s.wave
        @test_throws ArgumentError crop(s, 4003.2, 4003.8)          # between two pixels
    end
end
