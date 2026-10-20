@testset "Photometry" begin
    fl = [gaussian_filter("a", 4500.0, 200.0), gaussian_filter("b", 5700.0, 200.0)]

    @testset "construction" begin
        p = Photometry(fl, [1.0, 2.0], [0.1, 0.2])
        @test nphot(p) == 2
        @test band_names(p) == ["a", "b"]
        @test all(p.good) && p.z == 0.0
        @test p.rest[1].wave == fl[1].wave
        @test p.err == [0.1, 0.2]
    end

    @testset "redshifted filters" begin
        p = Photometry(fl, [1.0, 2.0], [0.1, 0.2]; z=0.1)
        @test p.rest[1].wave ≈ fl[1].wave ./ 1.1
        @test p.rest[2].norm ≈ fl[2].norm / 1.1^2 rtol = 1e-12
        @test p.filters[1].wave == fl[1].wave          # observer-frame originals kept
    end

    @testset "error floor" begin
        p = Photometry(fl, [1.0, 2.0], [0.1, 0.2]; error_floor=0.1)
        @test p.err ≈ [hypot(0.1, 0.1), hypot(0.2, 0.2)]
    end

    @testset "bad points are flagged" begin
        @test Photometry(fl, [1.0, NaN], [0.1, 0.1]).good == [true, false]
        @test Photometry(fl, [1.0, 2.0], [0.1, -1.0]).good == [true, false]
        @test Photometry(fl, [1.0, 2.0], [0.1, 0.0]).good == [true, false]
        @test Photometry(fl, [1.0, 2.0], [0.1, 0.1]; good=[false, true]).good == [false, true]
    end

    @testset "validation" begin
        @test_throws ArgumentError Photometry(fl, [1.0], [0.1, 0.2])
        @test_throws ArgumentError Photometry(fl, [1.0, 2.0], [0.1])
        @test_throws ArgumentError Photometry(fl, [1.0, 2.0], [0.1, 0.2]; good=[true])
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "Photometry (cases)" begin
    fl = [gaussian_filter("g", 4800.0, 300.0), gaussian_filter("r", 6200.0, 300.0)]
    @testset "simple" begin
        p = Photometry(fl, [1.0, 2.0], [0.1, 0.1])
        @test nphot(p) == 2 && band_names(p) == ["g", "r"] && all(p.good)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError Photometry(fl, [1.0, 2.0], [0.1, 0.1]; error_floor=-0.1)
        @test_throws ArgumentError Photometry(fl, [1.0, 2.0], [0.1, 0.1]; z=-1.0)
        @test_throws ArgumentError Photometry(fl, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
    end
    @testset "realistic" begin
        # ugriz-like magnitudes at z = 0.08 with 0.02–0.1 mag errors and a 3 % floor
        bands = [gaussian_filter(n, c, 350.0) for (n, c) in zip(("u", "g", "r", "i", "z"),
                                                                (3550.0, 4700.0, 6200.0, 7500.0, 8900.0))]
        mags = [19.8, 18.4, 17.6, 17.2, 17.0]
        dmag = [0.10, 0.03, 0.02, 0.02, 0.05]
        F = [ab_to_flambda(m, b; unit=1e-17) for (m, b) in zip(mags, bands)]
        σ = mag_err_to_flux_err.(F, dmag)
        p = Photometry(bands, F, σ; z=0.08, error_floor=0.03)
        @test p.err ≈ hypot.(σ, 0.03 .* F)
        @test all(p.err .>= σ)
        @test pivot_wavelength(p.rest[3]) ≈ pivot_wavelength(bands[3]) / 1.08 rtol = 1e-12
    end
    @testset "pathological" begin
        # zero flux with a floor keeps its measurement error; a non-detection with
        # zero error is unusable
        p = Photometry(fl, [0.0, 1.0], [0.1, 0.0]; error_floor=0.05)
        @test p.err[1] ≈ 0.1 && p.good == [true, false]
        # infinite error is flagged bad rather than giving zero weight silently
        @test !Photometry(fl, [1.0, 1.0], [Inf, 0.1]).good[1]
    end
end
