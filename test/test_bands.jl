import Specje: trapz_weights

@testset "band templates" begin
    wave = collect(3000.0:1.0:9000.0)
    fl = [gaussian_filter("a", 4500.0, 200.0; n=2001), gaussian_filter("b", 6000.0, 150.0; n=2001)]

    @testset "trapz_weights" begin
        @test sum(trapz_weights(wave)) ≈ 6000.0
        x = [0.0, 1.0, 3.0, 4.0]
        @test trapz_weights(x) ≈ [0.5, 1.5, 1.5, 0.5]
    end

    @testset "bandpass_weights" begin
        W = bandpass_weights(fl, wave)
        @test size(W) == (2, length(wave))
        @test W * ones(length(wave)) ≈ ones(2) rtol = 1e-5       # constant spectrum
        expected = [4500.0 + 200.0^2 / 4500.0, 6000.0 + 150.0^2 / 6000.0]
        @test W * wave ≈ expected rtol = 1e-5                    # ⟨λ⟩ over a Gaussian band
    end

    @testset "coverage" begin
        far = [gaussian_filter("far", 3100.0, 200.0)]
        @test_throws ArgumentError bandpass_weights(far, wave)
    end

    @testset "redshift invariance of band averages" begin
        z = 0.1
        pz = Photometry(fl, ones(2), ones(2); z=z)
        p0 = Photometry(fl, ones(2), ones(2))
        @test bandpass_weights(pz.rest, wave) * ones(length(wave)) ≈ ones(2) rtol = 1e-5
        # F(λ') = λ' observed through shifted filters picks up 1/(1+z)
        @test bandpass_weights(pz.rest, wave) * wave ≈ (bandpass_weights(p0.rest, wave) * wave) ./ (1 + z) rtol = 1e-4
    end

    @testset "band_templates" begin
        narrow = flat_library()                                   # F = a·z per template
        wide = SSPLibrary(collect(3000.0:2.0:9000.0),
                          [float(a * z) for _ in 1:3001, a in 1:3, z in 1:2],
                          narrow.log_age, narrow.logzsol, narrow.mass_frac)
        phot = Photometry(fl, ones(2), ones(2))

        bt = band_templates(wide, narrow, phot)
        @test bt.names == ["a", "b"] && size(bt.M) == (2, 6)
        @test bt.M ≈ repeat(reshape(narrow.flux[1, :, :], 1, :), 2) rtol = 1e-4

        normed = normalize_window(narrow; window=(4400.0, 4600.0))
        @test all(≈(1.0; rtol=1e-4), band_templates(wide, normed, phot).M)

        bad = SSPLibrary(wide.wave, wide.flux, wide.log_age .+ 0.5, wide.logzsol, wide.mass_frac)
        @test_throws ArgumentError band_templates(bad, narrow, phot)
        short = subset_library(wide; log_age=(8.0, 9.0))
        @test_throws ArgumentError band_templates(short, narrow, phot)
    end

    @testset "BandTemplates helpers" begin
        b = BandTemplates(["a"], ones(1, 3))
        @test Specje.convert_bands(Float32, b).M isa Matrix{Float32}
        @test Specje.convert_bands(Float64, nothing) === nothing
        @test_throws ArgumentError Specje.check_band_columns(b, 4)
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "bandpass_weights (cases)" begin
    @testset "simple" begin
        g = gaussian_filter("g", 5500.0, 200.0)
        W = bandpass_weights([g], collect(4000.0:1.0:7000.0))
        @test sum(W) ≈ 1.0 rtol = 1e-6                # a constant spectrum averages to itself
    end
    @testset "invalid input" begin
        t = tophat_filter("t", 5000.0, 6000.0)
        @test_throws ArgumentError bandpass_weights([t], collect(5500.0:1.0:7000.0))
        @test_throws ArgumentError bandpass_weights([t], [7000.0, 6000.0, 5000.0])
        @test_throws ArgumentError trapz_weights([1.0])
        @test_throws ArgumentError BandTemplates(["a", "b"], ones(1, 3))
    end
    @testset "realistic" begin
        # a power-law SED f_λ ∝ λ^-2 (flat in f_ν) through a smooth band, against a
        # fine-quadrature reference on the filter's own grid
        g = gaussian_filter("g", 5500.0, 300.0; n=8001)
        exact = Specje.trapz(g.wave, g.trans .* g.wave .^ -1) / g.norm
        wave = collect(3000.0:0.5:9000.0)
        @test only(bandpass_weights([g], wave) * wave .^ -2) ≈ exact rtol = 1e-6
    end
    @testset "pathological" begin
        # a top-hat's sharp edges are only resolved to one library pixel: on a 1 Å grid a
        # 1000 Å-wide band loses Δλ / width = 0.1 % of a constant spectrum
        t = tophat_filter("t", 5000.0, 6000.0)
        W = bandpass_weights([t], collect(4000.0:1.0:7000.0))
        @test sum(W) ≈ 1 - 1 / 1000 rtol = 1e-6
        # a filter exactly as wide as the library is accepted
        wave = collect(5000.0:1.0:6000.0)
        @test size(bandpass_weights([t], wave)) == (1, 1001)
        # a library sampled on just two points still integrates linearly
        @test trapz_weights([2.0, 5.0]) == [1.5, 1.5]
    end
end
