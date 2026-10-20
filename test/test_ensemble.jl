using LinearAlgebra
using Random
using Statistics
using LogDensityProblems
import Specje: stretch_factor, logp_columns

struct GaussTarget
    Σinv::Matrix{Float64}
end
LogDensityProblems.logdensity(g::GaussTarget, x) = -0.5 * dot(x, g.Σinv * x)
LogDensityProblems.dimension(g::GaussTarget) = size(g.Σinv, 1)
LogDensityProblems.capabilities(::Type{GaussTarget}) = LogDensityProblems.LogDensityOrder{0}()

struct WallTarget end
LogDensityProblems.logdensity(::WallTarget, x) = x[1] < 0 ? -Inf : -0.5 * dot(x, x)
LogDensityProblems.dimension(::WallTarget) = 2
LogDensityProblems.capabilities(::Type{WallTarget}) = LogDensityProblems.LogDensityOrder{0}()

function ar1(rng, φ, n)
    x = zeros(n)
    for t in 2:n
        x[t] = φ * x[t-1] + randn(rng)
    end
    return x
end

@testset "ensemble sampler" begin
    Σ = [1.0 0.8; 0.8 2.0]
    target = GaussTarget(inv(Σ))

    @testset "stretch factor" begin
        rng = Xoshiro(1)
        zs = [stretch_factor(rng, 2.0) for _ in 1:20_000]
        @test all(x -> 0.5 <= x <= 2.0, zs)
        @test mean(zs) ≈ 7 / 6 atol = 0.02              # (a^{3/2}-a^{-3/2}) / (3(a^{1/2}-a^{-1/2}))
    end

    @testset "logp_columns" begin
        Y = [0.0 1.0 2.0; 0.0 1.0 -1.0]
        expected = [LogDensityProblems.logdensity(target, Y[:, k]) for k in 1:3]
        @test logp_columns(target, Y) ≈ expected
        @test logp_columns(target, Y; threaded=true) ≈ expected
    end

    @testset "recovers a correlated Gaussian" begin
        chain = sample_ensemble(Xoshiro(10), target, [0.0, 0.0]; nwalkers=32, nsteps=3000)
        @test size(chain.samples) == (2, 32, 3000)
        @test size(chain.logp) == (32, 3000)
        @test 0.1 < acceptance_fraction(chain) < 0.9
        flat = flatten_chain(chain; burn=500)
        @test size(flat) == (2, 32 * 2500)
        @test vec(mean(flat; dims=2)) ≈ [0.0, 0.0] atol = 0.15
        @test cov(flat') ≈ Σ atol = 0.25
        τ = autocorr_time(chain)
        @test length(τ) == 2 && all(>(1.0), τ) && all(<(600.0), τ)
    end

    @testset "thinning, recorded log densities, determinism" begin
        a = sample_ensemble(Xoshiro(3), target, [0.0, 0.0]; nwalkers=16, nsteps=100, thin=10)
        @test size(a.samples) == (2, 16, 10)
        @test a.logp[:, end] ≈ [LogDensityProblems.logdensity(target, a.samples[:, w, end]) for w in 1:16]
        b = sample_ensemble(Xoshiro(3), target, [0.0, 0.0]; nwalkers=16, nsteps=100, thin=10)
        @test a.samples == b.samples
        c = sample_ensemble(Xoshiro(3), target, [0.0, 0.0]; nwalkers=16, nsteps=100, thin=10, threaded=true)
        @test a.samples == c.samples
    end

    @testset "explicit initial walkers" begin
        X0 = randn(Xoshiro(2), 2, 20)
        chain = sample_ensemble(Xoshiro(3), target, X0; nsteps=50)
        @test size(chain.samples) == (2, 20, 50)
    end

    @testset "input checks" begin
        @test_throws ArgumentError sample_ensemble(Xoshiro(1), target, randn(2, 7); nsteps=5)
        @test_throws ArgumentError sample_ensemble(Xoshiro(1), target, randn(3, 8); nsteps=5)
        @test_throws ArgumentError sample_ensemble(Xoshiro(1), target, randn(2, 8); nsteps=5, a=1.0)
        @test_throws ArgumentError sample_ensemble(Xoshiro(1), WallTarget(), [-10.0, 0.0]; nsteps=5)
    end

    @testset "respects hard walls" begin
        chain = sample_ensemble(Xoshiro(5), WallTarget(), [1.0, 0.0]; nwalkers=16, nsteps=500, ball_scale=0.1)
        @test all(>=(0), chain.samples[1, :, :])
        @test all(isfinite, chain.logp)
    end

    @testset "autocorrelation estimators" begin
        rng = Xoshiro(7)
        x = ar1(rng, 0.6, 200_000)
        acf = autocorr_function(x)
        @test acf[1] == 1.0
        @test acf[2] ≈ 0.6 atol = 0.02
        @test autocorr_time_1d(x) ≈ 4.0 rtol = 0.15      # (1+φ)/(1−φ)
        @test autocorr_time_1d(randn(rng, 100_000)) ≈ 1.0 atol = 0.3
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "sample_ensemble (cases)" begin
    target = GaussTarget(Matrix(1.0I, 2, 2))
    @testset "simple" begin
        c = sample_ensemble(Xoshiro(1), target, [0.0, 0.0]; nsteps=50)
        @test size(c.samples) == (2, 16, 50) && size(c.logp) == (16, 50)
        @test 0 < acceptance_fraction(c) < 1
    end
    @testset "invalid input" begin
        rng = Xoshiro(2)
        @test_throws ArgumentError sample_ensemble(rng, target, randn(rng, 2, 5))        # odd
        @test_throws ArgumentError sample_ensemble(rng, target, randn(rng, 2, 2))        # < 4
        @test_throws ArgumentError sample_ensemble(rng, target, randn(rng, 3, 8))        # dimension
        @test_throws ArgumentError sample_ensemble(rng, target, randn(rng, 2, 8); a=1.0)
        @test_throws ArgumentError sample_ensemble(rng, target, randn(rng, 2, 8); nsteps=0)
        @test_throws ArgumentError sample_ensemble(rng, target, randn(rng, 2, 8); nsteps=10, thin=20)
        @test_throws ArgumentError sample_ensemble(rng, target, [0.0, 0.0]; ball_scale=0.0)
        @test_throws ArgumentError sample_ensemble(rng, WallTarget(), [-1.0 -1 -1 -1; 0 1 2 3])   # zero density
    end
    @testset "realistic" begin
        # a strongly correlated, badly scaled 3-D Gaussian: affine invariance copes
        Σ = [1.0 0.95 0.0; 0.95 1.0 0.0; 0.0 0.0 1e4]
        c = sample_ensemble(Xoshiro(3), GaussTarget(inv(Σ)), zeros(3); nsteps=4000, ball_scale=0.1)
        X = flatten_chain(c; burn=1000)
        @test cov(X') ≈ Σ rtol = 0.15
        @test all(autocorr_time(c) .< 200)
    end
    @testset "pathological" begin
        # walkers that all start at one point can never move apart: rejected up front
        @test_throws ArgumentError sample_ensemble(Xoshiro(4), target, zeros(2, 8))
        # thinning keeps every k-th state and the counts still cover every proposal
        c = sample_ensemble(Xoshiro(5), target, [0.0, 0.0]; nsteps=100, thin=10)
        @test size(c.samples, 3) == 10 && c.nprop == 100 * 16
        # a hard wall: no recorded sample ever crosses it
        cw = sample_ensemble(Xoshiro(6), WallTarget(), [1.0, 0.0]; nsteps=300, ball_scale=0.1)
        @test all(>=(0), cw.samples[1, :, :])
    end
end

@testset "flatten_chain and autocorrelation (cases)" begin
    @testset "simple" begin
        c = EnsembleChain(reshape(collect(1.0:12.0), 2, 2, 3), zeros(2, 3), 0, 1)
        @test flatten_chain(c) == reshape(collect(1.0:12.0), 2, 6)
        @test flatten_chain(c; burn=2) == [9.0 11.0; 10.0 12.0]
    end
    @testset "invalid input" begin
        c = EnsembleChain(zeros(2, 2, 3), zeros(2, 3), 0, 1)
        @test_throws ArgumentError flatten_chain(c; burn=3)
        @test_throws ArgumentError flatten_chain(c; burn=-1)
        @test_throws ArgumentError autocorr_function([1.0])
    end
    @testset "realistic" begin
        # AR(1) with φ = 0.9: τ = (1 + φ) / (1 − φ) = 19
        x = ar1(Xoshiro(7), 0.9, 200_000)
        @test autocorr_time_1d(x) ≈ 19.0 rtol = 0.1
        acf = autocorr_function(x)
        @test acf[1] == 1.0
        @test acf[2] ≈ 0.9 atol = 0.01
    end
    @testset "pathological" begin
        # white noise has τ ≈ 1; a constant series has no defined autocorrelation
        @test autocorr_time_1d(randn(Xoshiro(8), 100_000)) ≈ 1.0 atol = 0.05
        @test_throws ArgumentError autocorr_function(fill(3.0, 10))
        # two samples is the minimum
        @test autocorr_function([1.0, -1.0])[1] == 1.0
    end
end
