using LinearAlgebra
using Random
using Statistics
import Specje: gibbs_sweep!, gaussian_draw

@testset "weight posterior" begin
    rng = Xoshiro(8)
    n, m = 30, 3
    A = randn(rng, n, m)
    d = A * [1.0, 2.0, 0.5] .+ 0.1 .* randn(rng, n)
    S = combine(block_stats_all(A, FitData(d, ones(n), collect(1:n))), [1.0])

    @testset "conditional_weights is the ridge solution" begin
        wp = conditional_weights(S, fill(2.0, m))
        Λ = A' * A + I / 4
        @test wp.prec ≈ Λ
        @test wp.mean ≈ Λ \ (A' * d)
        @test wp.chol.U' * wp.chol.U ≈ wp.prec
    end

    @testset "flat-prior limit is least squares" begin
        @test conditional_weights(S, fill(1e6, m)).mean ≈ A \ d rtol = 1e-6
    end

    @testset "unconstrained draws have the right moments" begin
        wp = conditional_weights(S, fill(2.0, m))
        draws = sample_weights(rng, wp, 40_000; positive=false)
        @test size(draws) == (m, 40_000)
        @test vec(mean(draws; dims=2)) ≈ wp.mean atol = 0.01
        @test cov(draws') ≈ inv(Symmetric(wp.prec)) atol = 0.004
    end

    manual(Λ, μ) = WeightPosterior(μ, Λ, cholesky(Symmetric(Λ)))

    @testset "positive sampling ≈ Gaussian when far from the boundary" begin
        Λ = [2.0 0.5; 0.5 1.0]
        wp = manual(Λ, [3.0, 4.0])
        draws = sample_weights(rng, wp, 30_000; positive=true)
        @test all(>=(0), draws)
        @test vec(mean(draws; dims=2)) ≈ [3.0, 4.0] atol = 0.05
        @test cov(draws') ≈ inv(Λ) atol = 0.05
    end

    @testset "truncation moves the mean as in the truncated normal" begin
        wp = manual([1.0 0.0; 0.0 1.0], [-1.0, 2.0])
        draws = sample_weights(rng, wp, 20_000; positive=true)
        @test all(>=(0), draws)
        @test mean(draws[1, :]) ≈ 0.5251 atol = 0.03      # -1 + φ(1)/(1-Φ(1))
        @test mean(draws[2, :]) ≈ 2.0552 atol = 0.03      #  2 + φ(-2)/(1-Φ(-2))
    end

    @testset "far-tail truncation stays finite and non-negative" begin
        wp = manual([1.0 0.0; 0.0 1.0], [-8.0, 1.0])
        draws = sample_weights(rng, wp, 500; positive=true)
        @test all(isfinite, draws) && all(>=(0), draws)
    end

    @testset "unconstrained draws can be negative" begin
        wp = manual([1.0 0.0; 0.0 1.0], [-1.0, 2.0])
        @test any(<(0), sample_weights(rng, wp, 2000; positive=false)[1, :])
    end

    @testset "init and sweeps" begin
        wp = manual([2.0 0.5; 0.5 1.0], [1.0, 1.0])
        draws = sample_weights(rng, wp, 10; positive=true, burn=5, init=[-3.0, 0.5])
        @test all(>=(0), draws)
        w = [1.0, 1.0]
        @test gibbs_sweep!(rng, w, wp) === w
        @test all(>=(0), w)
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "conditional_weights and sample_weights (cases)" begin
    rng = Xoshiro(71)
    stats(A, d) = combine(block_stats_all(A, FitData(d, ones(length(d)), collect(1:length(d)))), [1.0])
    @testset "simple" begin
        S = stats(reshape([1.0, 1.0], 2, 1), [1.0, 3.0])
        wp = conditional_weights(S, [1e8])
        @test only(wp.mean) ≈ 2.0 rtol = 1e-6               # mean of the data
    end
    @testset "invalid input" begin
        S = stats(randn(rng, 10, 2), randn(rng, 10))
        @test_throws DimensionMismatch conditional_weights(S, [1.0])
        wp = conditional_weights(S, [1.0, 1.0])
        @test_throws ArgumentError sample_weights(rng, wp, 0)
        @test_throws ArgumentError sample_weights(rng, wp, 10; burn=-1)
        @test_throws ArgumentError sample_weights(rng, wp, 10; init=[1.0])
    end
    @testset "realistic" begin
        # two nearly degenerate SSP-like columns: the posterior is strongly anti-correlated,
        # and the positive draws respect the sum that the data constrain
        x = range(0, 1; length=200)
        A = [exp.(-x) exp.(-1.05 .* x)]
        d = A * [1.0, 1.0] .+ 0.01 .* randn(rng, 200)
        wp = conditional_weights(stats(A ./ 0.01, d ./ 0.01), [10.0, 10.0])
        C = inv(Symmetric(wp.prec))
        @test C[1, 2] / sqrt(C[1, 1] * C[2, 2]) < -0.99
        draws = sample_weights(rng, wp, 3000; positive=true, burn=500)
        @test all(>=(0), draws)
        @test mean(sum(draws; dims=1)) ≈ 2.0 rtol = 0.02
    end
    @testset "pathological" begin
        # a weight the data do not constrain is drawn from its prior
        A = [randn(rng, 50) zeros(50)]
        wp = conditional_weights(stats(A, randn(rng, 50)), [1.0, 3.0])
        draws = sample_weights(rng, wp, 20_000; positive=false)
        @test std(draws[2, :]) ≈ 3.0 rtol = 0.03
        # positive draws of that weight follow the half-normal of the prior
        pos = sample_weights(rng, wp, 20_000; positive=true)
        @test mean(pos[2, :]) ≈ 3.0 * sqrt(2 / π) rtol = 0.05
    end
end
