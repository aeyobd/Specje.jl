using LinearAlgebra
using Random

@testset "nnls" begin
    rng = Xoshiro(2)

    @testset "exact non-negative solution" begin
        A = rand(rng, 40, 8)
        x = [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, 0.0, 3.0]
        @test nnls(A, A * x) ≈ x atol = 1e-8
    end

    @testset "identity clips negatives" begin
        @test nnls(Matrix(1.0I, 4, 4), [1.0, -2.0, 3.0, -0.5]) ≈ [1.0, 0.0, 3.0, 0.0]
    end

    @testset "agrees with least squares when that is positive" begin
        A = rand(rng, 50, 5)
        b = A * fill(1.0, 5) .+ 1e-3 .* randn(rng, 50)
        @test nnls(A, b) ≈ A \ b atol = 1e-9
    end

    @testset "KKT conditions on a random problem" begin
        A, b = randn(rng, 60, 12), randn(rng, 60)
        x = nnls(A, b)
        w = A' * (b - A * x)
        @test all(>=(0), x)
        @test all(abs.(w[x .> 0]) .< 1e-8)
        @test all(w[x .== 0] .<= 1e-8)
    end

    @testset "degenerate cases" begin
        @test nnls(rand(rng, 5, 3), zeros(5)) == zeros(3)
        @test nnls(ones(4, 2), -ones(4)) == zeros(2)
        @test_throws DimensionMismatch nnls(ones(3, 2), ones(4))
    end

    @testset "duplicate columns" begin
        a = rand(rng, 30)
        x = nnls(hcat(a, a), 2 .* a)
        @test sum(x) ≈ 2 atol = 1e-8
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "nnls (cases)" begin
    @testset "simple" begin
        @test nnls([1.0 0.0; 0.0 1.0], [2.0, 3.0]) ≈ [2.0, 3.0]
    end
    @testset "invalid input" begin
        @test_throws DimensionMismatch nnls(ones(3, 2), ones(2))
        @test_throws ArgumentError nnls(ones(3, 2), ones(3); maxiter=0)
    end
    @testset "realistic" begin
        # 48 SSP-like, strongly correlated columns; a sparse truth with noise
        rng = Xoshiro(11)
        x = collect(range(0, 1; length=400))
        A = reduce(hcat, [exp.(-((x .- c) ./ 0.3) .^ 2) .* (1 .+ 0.1 .* sin.(20 .* x .+ c)) for c in range(0, 1; length=48)])
        truth = zeros(48); truth[[5, 20, 41]] = [1.0, 0.5, 2.0]
        b = A * truth .+ 1e-4 .* randn(rng, 400)
        sol = nnls(A, b)
        gradient = A' * (b - A * sol)
        @test all(>=(0), sol)
        @test all(abs.(gradient[sol .> 0]) .< 1e-6)                 # KKT: stationary on support
        @test all(gradient[sol .== 0] .<= 1e-6)                     # KKT: no improving direction
        @test norm(A * sol - b) <= norm(A * truth - b) * (1 + 1e-9)
    end
    @testset "pathological" begin
        @test nnls(zeros(5, 3), ones(5)) == zeros(3)                 # no information at all
        @test nnls(ones(1, 1), [-4.0]) == [0.0]
        # more unknowns than equations: still feasible and optimal
        rng = Xoshiro(12)
        A, b = rand(rng, 3, 10), rand(rng, 3)
        sol = nnls(A, b)
        @test all(>=(0), sol) && norm(A * sol - b) < 1e-8
        # a huge overall scale does not change the solution
        A2 = rand(rng, 20, 4); b2 = A2 * [1.0, 0.0, 2.0, 0.5]
        @test nnls(1e12 .* A2, 1e12 .* b2) ≈ [1.0, 0.0, 2.0, 0.5] atol = 1e-6
    end
end
