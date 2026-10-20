using LinearAlgebra
using Random
import Specje: ridge_precision

@testset "marginal likelihood" begin
    rng = Xoshiro(5)
    n, m = 14, 4
    A = randn(rng, n, m)
    σ = 0.5 .+ rand(rng, n)
    d = randn(rng, n)
    τ = [1.0, 2.0, 0.5, 3.0]
    rows = collect(1:n)

    # Dense reference: d ~ N(0, diag((s σ)²) + A D Aᵀ) on the chosen rows.
    function brute(r, sv)
        C = Diagonal((sv .* σ[r]) .^ 2) + A[r, :] * Diagonal(τ .^ 2) * A[r, :]'
        return -(length(r) * log(2π) + logdet(C) + dot(d[r], C \ d[r])) / 2
    end
    fd = FitData(d ./ σ, 1 ./ σ, rows)
    ll(fd, s) = marginal_loglike(combine(block_stats_all(A, fd), s), τ)

    @testset "block statistics" begin
        b = only(block_stats_all(A, fd))
        Aw = A ./ σ
        @test b.AtA ≈ Aw' * Aw
        @test b.Atb ≈ Aw' * (d ./ σ)
        @test b.btb ≈ sum(abs2, d ./ σ)
        @test b.n == n && b.logσ ≈ sum(log, σ)
    end

    @testset "agrees with the dense Gaussian" begin
        @test ll(fd, [1.0]) ≈ brute(rows, ones(n)) rtol = 1e-10
    end

    @testset "noise scale" begin
        @test ll(fd, [1.7]) ≈ brute(rows, fill(1.7, n)) rtol = 1e-10
    end

    @testset "two blocks with different scales" begin
        fd2 = FitData(d ./ σ, 1 ./ σ, rows, [1:9, 10:14])
        sv = vcat(fill(0.8, 9), fill(2.5, 5))
        @test ll(fd2, [0.8, 2.5]) ≈ brute(rows, sv) rtol = 1e-10
    end

    @testset "masked rows" begin
        good = [1:5; 7:n]
        fdm = FitData(d[good] ./ σ[good], 1 ./ σ[good], good)
        @test ll(fdm, [1.0]) ≈ brute(good, ones(length(good))) rtol = 1e-10
    end

    @testset "ridge precision" begin
        S = combine(block_stats_all(A, fd), [1.0])
        @test ridge_precision(S, τ) ≈ S.AtA + Diagonal(1 ./ τ .^ 2)
    end

    @testset "input checks" begin
        S = combine(block_stats_all(A, fd), [1.0])
        @test_throws DimensionMismatch marginal_loglike(S, τ[1:3])
        @test_throws ArgumentError combine(block_stats_all(A, fd), [1.0, 2.0])
    end

    @testset "flat prior limit prefers the true noise level" begin
        wtrue = [1.0, -0.5, 2.0, 0.3]
        dd = A * wtrue .+ 0.2 .* randn(rng, n)
        f = FitData(dd, ones(n), rows)
        τbig = fill(10.0, m)
        score(s) = marginal_loglike(combine(block_stats_all(A, f), [s]), τbig)
        @test score(0.2) > score(1.5)
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "block_stats_all, combine, marginal_loglike (cases)" begin
    @testset "simple" begin
        # one column, one row: closed form N(d | 0, σ² + a²τ²)
        fd = FitData([2.0], [1.0], [1])
        S = combine(block_stats_all(reshape([1.0], 1, 1), fd), [1.0])
        @test marginal_loglike(S, [1.0]) ≈ -0.5 * (log(2π) + log(2.0) + 4.0 / 2.0)
    end
    @testset "invalid input" begin
        fd = FitData([1.0, 2.0], [1.0, 1.0], [1, 2])
        @test_throws DimensionMismatch block_stats_all(ones(1, 3), fd)
        @test_throws ArgumentError combine(Specje.BlockStats[], Float64[])
        S = combine(block_stats_all(ones(2, 3), fd), [1.0])
        @test_throws DimensionMismatch marginal_loglike(S, [1.0, 1.0])
    end
    @testset "realistic" begin
        # spectrum (400 rows) + photometry (5 rows) blocks with their own noise scales,
        # against the dense covariance
        rng = Xoshiro(61)
        n1, n2, m = 400, 5, 6
        A = randn(rng, n1 + n2, m)
        σ = 0.5 .+ rand(rng, n1 + n2)
        d = A * rand(rng, m) .+ σ .* randn(rng, n1 + n2)
        τ = fill(2.0, m)
        fd = FitData(d ./ σ, 1 ./ σ, collect(1:n1+n2), [1:n1, n1+1:n1+n2])
        s = [1.2, 0.7]
        S = combine(block_stats_all(A, fd), s)
        scale = vcat(fill(s[1], n1), fill(s[2], n2))
        C = Diagonal((scale .* σ) .^ 2) + A * Diagonal(τ .^ 2) * A'
        dense = -(length(d) * log(2π) + logdet(C) + dot(d, C \ d)) / 2
        @test marginal_loglike(S, τ) ≈ dense rtol = 1e-10
    end
    @testset "pathological" begin
        rng = Xoshiro(62)
        a = randn(rng, 20)
        fd = FitData(randn(rng, 20), ones(20), collect(1:20))
        # duplicate columns with an effectively flat prior: Λ is singular → -Inf, not NaN
        S = combine(block_stats_all([a a], fd), [1.0])
        @test marginal_loglike(S, [1e12, 1e12]) == -Inf
        # with a proper prior the same problem is fine
        @test isfinite(marginal_loglike(S, [1.0, 1.0]))
        # the evidence falls off for absurd noise scales in either direction
        ll(s) = marginal_loglike(combine(block_stats_all(reshape(a, :, 1), fd), [s]), [1.0])
        @test ll(1.0) > ll(1e-3) && ll(1.0) > ll(1e3)
    end
end
