using LogDensityProblems
using LinearAlgebra: I
using SparseArrays: nnz, nonzeros
import Specje: nmodel

@testset "forward-model projection" begin
    s = synthetic_setup()
    mw = s.dwave                                            # log model grid, 400 px
    star = StellarComponent(s.grid, s.T; σ_min=25.0, vmax=1000.0)
    native = collect(range(mw[3], mw[end-2]; length=150))   # coarser linear pixels inside
    P = pixel_projection(mw, native)

    @testset "model_grid" begin
        lin = collect(range(4000.0, 5000.0; length=500))
        g = model_grid(lin; oversample=2)
        e, ge = pixel_edges(lin), log_pixel_edges(g)
        @test is_loguniform(g)
        @test first(ge) <= first(e) && last(ge) >= last(e)
        @test velscale(g) ≈ C_KMS * minimum(diff(log.(e))) / 2 rtol = 1e-9
        @test velscale(model_grid(lin; oversample=3)) ≈ 2 / 3 * velscale(g) rtol = 1e-9
        @test_throws ArgumentError model_grid(lin; oversample=0.5)
    end

    @testset "pixel_projection" begin
        f = randn(Xoshiro(1), length(mw))
        @test size(P) == (150, 400)
        @test P * f ≈ rebin_flux(pixel_edges(native), log_pixel_edges(mw), f) rtol = 1e-12
        @test all(sum(P; dims=2) .≈ 1)
        @test P * ones(length(mw)) ≈ ones(150)
        @test_throws ArgumentError pixel_projection(mw, [mw[1] - 5.0; native])
    end

    @testset "GaussianLSF on another grid" begin
        sp = Spectrum(native, ones(150), ones(150); lsf_fwhm=collect(range(2.0, 3.0; length=150)))
        l = GaussianLSF(sp, mw)
        @test npix(l) == length(mw)
        @test l.sigma_kms[200] ≈ fwhm_to_sigma_kms(Specje.interp_linear(mw[200], native, sp.lsf_fwhm), mw[200])
        @test_throws ArgumentError GaussianLSF(Spectrum(native, ones(150), ones(150)), mw)
    end

    @testset "projected model" begin
        poly = PolynomialComponent(length(mw), 1)
        m0 = SpectralModel(s.grid, star, poly)
        m = SpectralModel(s.grid, star, poly; projection=P)
        θ = [120.0, 200.0]
        @test npix(m) == 150 && nmodel(m) == 400 && npix(m0) == 400
        @test design_matrix(m, θ) ≈ P * design_matrix(m0, θ)
        @test_throws DimensionMismatch SpectralModel(s.grid, star; projection=P[:, 1:399])
        @test_throws ArgumentError SpectralModel(s.grid, star, PolynomialComponent(150, 1); projection=P)
        @test_throws ArgumentError fit_kinematics(m, Spectrum(mw, ones(400), ones(400)), θ)
    end

    @testset "fit on native pixels recovers the truth" begin
        m = SpectralModel(s.grid, star; projection=P)
        w = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
        clean = design_matrix(m, [120.0, 200.0]) * w
        σn = 0.005 * mean(clean)
        sp = Spectrum(native, clean .+ σn .* randn(Xoshiro(2), 150), fill(σn, 150))
        r = fit_kinematics(m, sp, [100.0, 150.0])
        @test r.θ[1] ≈ 120.0 atol = 3.0
        @test r.θ[2] ≈ 200.0 atol = 6.0
        @test length(model_flux(m, r)) == 150
    end

    @testset "gradient through the projection" begin
        lsf = GaussianLSF(fill(60.0, length(mw)))
        gas = GasComponent(mw, [GasGroup("A", Multiplet("a", 4260.0))], lsf; σ_min=10.0, vmax=500.0)
        m = SpectralModel(s.grid, star, gas; projection=P)
        θ = [120.0, 200.0, 30.0, 100.0]
        clean = design_matrix(m, θ) * [0.0, 2.0, 0.0, 1.0, 0.5, 0.0, 3.0]
        σn = 0.02 * mean(clean)
        sp = Spectrum(native, clean .+ σn .* randn(Xoshiro(3), 150), fill(σn, 150))
        bm = BayesModel(m, FitData(sp), θ; fit_noise=true)
        u = initial_free(bm, θ) .+ 0.03
        lp, g = LogDensityProblems.logdensity_and_gradient(bm, u)
        h = 1e-3          # the density is O(10³): smaller steps lose digits to rounding
        fd = [(log_posterior(bm, u .+ h .* (1:5 .== i)) - log_posterior(bm, u .- h .* (1:5 .== i))) / 2h
              for i in 1:5]
        @test lp ≈ log_posterior(bm, u)
        @test g ≈ fd rtol = 1e-3
    end
end
