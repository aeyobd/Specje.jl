@testset "filters" begin
    @testset "construction and validation" begin
        w = collect(4000.0:10.0:5000.0)
        f = Filter("x", w, ones(length(w)))
        @test f.name == "x"
        @test f.norm ≈ (5000.0^2 - 4000.0^2) / 2 rtol = 1e-12
        @test_throws ArgumentError Filter("bad", w, ones(5))
        @test_throws ArgumentError Filter("bad", reverse(w), ones(length(w)))
        @test_throws ArgumentError Filter("bad", w, -ones(length(w)))
        @test_throws ArgumentError Filter("bad", w, zeros(length(w)))
        @test_throws ArgumentError Filter("bad", [5000.0], [1.0])
    end

    @testset "transmission" begin
        f = gaussian_filter("g", 5000.0, 100.0)
        @test transmission(f, 5000.0) ≈ 1.0 atol = 1e-12
        @test transmission(f, 4400.0) == 0.0
        @test transmission(f, 5600.0) == 0.0
        @test transmission(f, first(f.wave)) == 0.0
        @test 0 < transmission(f, 5050.0) < 1
        @test transmission(f, 5050.0) ≈ exp(-0.125) rtol = 1e-3
    end

    @testset "norm and pivot" begin
        g = gaussian_filter("g", 5000.0, 100.0)
        @test g.norm ≈ 5000.0 * 100.0 * sqrt(2π) rtol = 1e-5
        t = tophat_filter("t", 5000.0, 6000.0)
        @test pivot_wavelength(t) ≈ sqrt((6000.0^2 - 5000.0^2) / 2 / log(6 / 5)) rtol = 1e-6
    end

    @testset "to_rest" begin
        f = gaussian_filter("g", 5000.0, 100.0)
        r = to_rest(f, 0.25)
        @test r.wave ≈ f.wave ./ 1.25
        @test r.trans == f.trans
        @test r.norm ≈ f.norm / 1.25^2 rtol = 1e-12
        @test r.name == "g"
    end

    @testset "magnitudes and units" begin
        f = gaussian_filter("g", 5000.0, 100.0)
        @test flambda_to_ab(ab_to_flambda(22.5, f), f) ≈ 22.5
        @test jy_to_flambda(3631.0, f) ≈ ab_to_flambda(0.0, f) rtol = 1e-3
        @test ab_to_flambda(20.0, f; unit=1e-17) ≈ ab_to_flambda(20.0, f) * 1e17
        @test ab_to_flambda(21.0, f) < ab_to_flambda(20.0, f)
        @test ab_to_flambda(20.0, f) / ab_to_flambda(25.0, f) ≈ 100.0
        @test mag_err_to_flux_err(2.0, 0.1) ≈ 2.0 * 0.1 * log(10) / 2.5
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "Filter and transmission (cases)" begin
    @testset "simple" begin
        f = Filter("box", [1000.0, 2000.0], [1.0, 1.0])
        @test f.norm ≈ (2000.0^2 - 1000.0^2) / 2
        @test transmission(f, 1500.0) == 1.0
    end
    @testset "invalid input" begin
        @test_throws ArgumentError Filter("x", [1000.0, 1000.0], [1.0, 1.0])
        @test_throws ArgumentError gaussian_filter("g", 5000.0, 0.0)
        @test_throws ArgumentError gaussian_filter("g", 5000.0, 100.0; n=1)
        @test_throws ArgumentError gaussian_filter("g", 100.0, 100.0)          # extends below 0 Å
        @test_throws ArgumentError tophat_filter("t", 6000.0, 5000.0)
        @test_throws ArgumentError tophat_filter("t", 0.0, 5000.0)
    end
    @testset "realistic" begin
        # an asymmetric, tabulated response: interpolation, norm and pivot behave
        w = collect(4000.0:25.0:5500.0)
        t = @. ifelse(w < 4500, (w - 4000) / 500, exp(-(w - 4500) / 400))
        f = Filter("asym", w, t)
        @test transmission(f, 4250.0) ≈ 0.5
        @test transmission(f, 4512.5) ≈ (t[21] + t[22]) / 2
        @test 4000 < pivot_wavelength(f) < 5500
        @test pivot_wavelength(to_rest(f, 0.3)) ≈ pivot_wavelength(f) / 1.3 rtol = 1e-12
    end
    @testset "pathological" begin
        # a single non-zero tabulated point still has throughput
        f = Filter("spike", [4999.0, 5000.0, 5001.0], [0.0, 1.0, 0.0])
        @test f.norm ≈ 5000.0
        @test transmission(f, 5000.5) ≈ 0.5
        @test to_rest(f, 0.0).wave == f.wave                           # z = 0: unchanged
        @test_throws ArgumentError to_rest(f, -1.0)
        @test transmission(f, -Inf) == 0.0 && transmission(f, Inf) == 0.0
    end
end

@testset "magnitude and flux conversions (cases)" begin
    f = tophat_filter("t", 5000.0, 6000.0)
    @testset "simple" begin
        @test flambda_to_ab(ab_to_flambda(0.0, f), f) ≈ 0.0 atol = 1e-12
    end
    @testset "invalid input" begin
        @test_throws ArgumentError ab_to_flambda(20.0, f; unit=0.0)
        @test_throws ArgumentError flambda_to_ab(-1.0, f)
        @test_throws ArgumentError flambda_to_ab(0.0, f)
        @test_throws ArgumentError jy_to_flambda(1.0, f; unit=-1e-17)
        @test_throws ArgumentError mag_err_to_flux_err(1.0, -0.1)
    end
    @testset "realistic" begin
        # an r = 17.5 galaxy in 1e-17 erg/s/cm²/Å (SDSS units), with a 0.03 mag error
        r = tophat_filter("r", 5500.0, 6900.0)
        F = ab_to_flambda(17.5, r; unit=1e-17)
        λp = pivot_wavelength(r)
        # AB: m = −2.5 log10 f_ν − 48.60 (zero point ≈ 3630.8 Jy)
        @test F ≈ 10^(-0.4 * (17.5 + 48.60)) * 2.99792458e18 / λp^2 / 1e-17 rtol = 1e-12
        @test flambda_to_ab(F, r; unit=1e-17) ≈ 17.5
        @test mag_err_to_flux_err(F, 0.03) / F ≈ 0.0276 atol = 1e-4   # ≈ 2.8 %
        @test jy_to_flambda(3631 * 10^(-0.4 * 17.5), r; unit=1e-17) ≈ F rtol = 1e-3
    end
    @testset "pathological" begin
        # very bright and very faint magnitudes stay finite and invertible
        for m in (-30.0, 40.0)
            @test flambda_to_ab(ab_to_flambda(m, f), f) ≈ m
        end
        @test mag_err_to_flux_err(5.0, 0.0) == 0.0
    end
end
