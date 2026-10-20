@testset "library resampling" begin
    lib = flat_library()

    @testset "rebin_log" begin
        out = rebin_log(lib; velscale=200.0)
        @test is_loguniform(out.wave)
        @test velscale(out.wave) ≈ 200.0 rtol = 1e-8
        @test size(out.flux) == (nwave(out), nage(lib), nmet(lib))
        @test first(out.wave) >= first(lib.wave) && last(out.wave) <= last(lib.wave)
    end

    @testset "metadata preserved" begin
        out = rebin_log(lib; velscale=200.0)
        @test out.log_age == lib.log_age
        @test out.logzsol == lib.logzsol
        @test out.mass_frac == lib.mass_frac
        @test out.norm == lib.norm
    end

    @testset "constant SSPs unchanged" begin
        out = rebin_log(lib; velscale=200.0)
        for a in 1:nage(lib), z in 1:nmet(lib)
            @test all(≈(float(a * z)), out.flux[:, a, z])
        end
    end

    @testset "wavelength range" begin
        out = rebin_log(lib; velscale=200.0, λmin=4200.0, λmax=4800.0)
        @test first(out.wave) >= 4200.0 && last(out.wave) <= 4800.0
    end

    @testset "flux conservation" begin
        t = toy_library(; nw=800)
        out = rebin_log(t; velscale=100.0)
        # log vs linear pixel weighting differs slightly for a ramp, hence the loose tolerance
        @test mean(out.flux[:, 2, 2]) ≈ mean(t.flux[:, 2, 2]) rtol = 2e-2
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "rebin_log(::SSPLibrary) (cases)" begin
    @testset "simple" begin
        out = rebin_log(flat_library(); velscale=100.0)
        @test all(≈(1.0), out.flux[:, 1, 1])
    end
    @testset "invalid input" begin
        lib = flat_library()
        @test_throws ArgumentError rebin_log(lib; velscale=0.0)
        @test_throws ArgumentError rebin_log(lib; velscale=100.0, λmin=4800.0, λmax=4200.0)
        @test_throws ArgumentError rebin_log(lib; velscale=100.0, λmin=6000.0, λmax=7000.0)
    end
    @testset "realistic" begin
        # absorption-line SSPs on a linear 0.5 Å grid: equivalent widths survive log rebinning
        wave = collect(4000.0:0.5:5000.0)
        f = [1 - 0.5 * exp(-(λ - 4500)^2 / (2 * 1.2^2)) for λ in wave, a in 1:2, z in 1:1]
        lib = SSPLibrary(wave, f, [9.0, 10.0], [0.0], ones(2, 1); lsf_fwhm=fill(2.5, length(wave)))
        out = rebin_log(lib; velscale=30.0)
        ew(wv, fl) = sum((1 .- fl) .* diff(pixel_edges(wv)))
        @test ew(out.wave, out.flux[:, 1, 1]) ≈ ew(wave, f[:, 1, 1]) rtol = 1e-3
        @test all(≈(2.5), out.lsf_fwhm)
    end
    @testset "pathological" begin
        lib = flat_library()                                 # 4000–5000 Å ≈ 67 000 km/s
        @test_throws ArgumentError rebin_log(lib; velscale=1e5)   # no two pixels fit
        out = rebin_log(lib; velscale=30_000.0)
        @test nwave(out) == 2
    end
end
