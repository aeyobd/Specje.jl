using Statistics
using Distributions: Normal, LogNormal, logpdf, params, UnivariateDistribution
import Specje: log_prior, build_priors

@testset "priors" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0, σ_max=600.0, vmax=1000.0)
    lsf = GaussianLSF(fill(60.0, nd))
    gas = GasComponent(s.dwave, [GasGroup("A", Multiplet("a", 4260.0)),
                                 GasGroup("B", Multiplet("b", 4400.0))], lsf;
                       σ_min=10.0, vmax=500.0)
    poly = PolynomialComponent(nd, 1)
    model = SpectralModel(s.grid, star, gas, poly)

    @testset "default priors per component" begin
        @test length(default_priors(star)) == 2
        @test length(default_priors(gas)) == 4
        @test isempty(default_priors(poly))
        pr = default_priors(model)
        @test length(pr) == nparams(model) == 6
        @test all(p -> p isa UnivariateDistribution, pr)
    end

    @testset "default priors respect the bounds" begin
        v, σ = default_priors(star)
        @test isfinite(logpdf(v, 0.0)) && logpdf(v, 2000.0) == -Inf
        @test isfinite(logpdf(σ, 200.0)) && logpdf(σ, 10.0) == -Inf && logpdf(σ, 900.0) == -Inf
    end

    @testset "joint model adds a calibration prior" begin
        f = joint_fixture(fit_scale=true)
        pr = default_priors(f.jm)
        @test length(pr) == nparams(f.jm) == 3
        @test pr[3] isa LogNormal
        @test length(default_priors(JointModel(f.stars, f.phot))) == 2
    end

    # data for the weight-scale tests: one stellar component, exact weights
    θ, w = [120.0, 200.0], [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    stars = SpectralModel(s.grid, star)
    clean = design_matrix(stars, θ) * w
    fd = FitData(Spectrum(s.dwave, clean, fill(0.01 * mean(clean), nd)))

    @testset "default_tau" begin
        @test default_tau(stars, fd, θ; factor=5.0) ≈ [10.0] rtol = 1e-3
        with_poly = SpectralModel(s.grid, star, PolynomialComponent(nd, 0))
        τ = default_tau(with_poly, fd, [θ; Float64[]]; factor=5.0)
        @test τ[1] ≈ 10.0 rtol = 1e-3
        @test τ[2] ≈ 1.0 atol = 0.05                 # floor: 10% of the largest weight
    end

    @testset "build_priors" begin
        τ = [10.0]
        p = build_priors(nothing, stars, τ, 1)
        @test length(p.θ) == 2 && length(p.logτ) == 1 && length(p.logs) == 1
        @test p.logτ[1] isa Normal && params(p.logτ[1]) == (log(10.0), 1.0)
        @test build_priors(p, stars, τ, 1) === p
        q = build_priors(default_priors(stars), stars, τ, 2)
        @test length(q.logs) == 2
    end

    @testset "log_prior" begin
        p = build_priors(nothing, stars, [10.0], 1)
        @test log_prior(p, θ, Float64[], Float64[]) ≈ sum(logpdf.(p.θ, θ))
        @test log_prior(p, θ, [0.3], [0.1]) ≈
              sum(logpdf.(p.θ, θ)) + logpdf(p.logτ[1], 0.3) + logpdf(p.logs[1], 0.1)
        @test log_prior(p, [5000.0, 200.0], Float64[], Float64[]) == -Inf
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "default_priors, default_tau, log_prior (cases)" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0, σ_max=600.0, vmax=1000.0)
    model = SpectralModel(s.grid, star)
    w = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    clean = design_matrix(model, [120.0, 200.0]) * w
    fd = FitData(Spectrum(s.dwave, clean, fill(0.01, nd)))
    @testset "simple" begin
        pr = default_priors(model)
        @test length(pr) == 2
        @test all(isfinite, logpdf.(pr, [0.0, 150.0]))
    end
    @testset "invalid input" begin
        @test_throws ArgumentError default_tau(model, fd, [120.0, 200.0]; factor=0.0)
        @test_throws ArgumentError default_tau(model, fd, [120.0])
        pr = build_priors(nothing, model, [1.0], 1)
        @test_throws ArgumentError log_prior(pr, [0.0], Float64[], Float64[])
    end
    @testset "realistic" begin
        # τ is factor × the largest NNLS weight of each component
        lsf = GaussianLSF(fill(60.0, nd))
        gas = GasComponent(s.dwave, [GasGroup("A", Multiplet("a", 4260.0))], lsf; σ_min=5.0)
        m = SpectralModel(s.grid, star, gas)
        flux = design_matrix(m, [120.0, 200.0, 0.0, 50.0]) * [w; 4.0]
        τ = default_tau(m, FitData(Spectrum(s.dwave, flux, fill(0.01, nd))), [120.0, 200.0, 0.0, 50.0])
        @test τ ≈ [5 * 2.0, 5 * 4.0] rtol = 1e-4
        @test length(default_priors(m)) == 4
    end
    @testset "pathological" begin
        # no signal: every NNLS weight is zero and the floor of 1 applies
        zero_fd = FitData(Spectrum(s.dwave, zeros(nd), fill(0.01, nd)))
        @test default_tau(model, zero_fd, [0.0, 150.0]) == [5.0]
        # outside the truncated support the prior density is zero
        pr = build_priors(nothing, model, [1.0], 1)
        @test log_prior(pr, [2000.0, 150.0], Float64[], Float64[]) == -Inf
        @test log_prior(pr, [0.0, 10.0], Float64[], Float64[]) == -Inf
    end
end
