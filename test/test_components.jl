using LinearAlgebra: Diagonal, det
import Specje: columns, in_bounds, to_free, from_free, physical_weights, check_component

@testset "components" begin
    s = synthetic_setup()
    grid = s.grid

    @testset "legendre_basis" begin
        x = range(-1, 1; length=5)
        P = legendre_basis(5, 3)
        @test size(P) == (5, 4)
        @test P[:, 1] == ones(5)
        @test P[:, 2] ≈ collect(x)
        @test P[:, 3] ≈ (3 .* x .^ 2 .- 1) ./ 2
        @test P[:, 4] ≈ (5 .* x .^ 3 .- 3 .* x) ./ 2
        @test all(≈(1.0), legendre_basis(7, 5)[end, :])
    end

    @testset "StellarComponent" begin
        c = StellarComponent(grid, s.T; σ_min=25.0, σ_max=600.0, vmax=800.0)
        @test nparams(c) == 2 && ncolumns(c) == 6
        out = columns(c, grid, [100.0, 150.0])
        @test size(out) == (length(grid.crop), 6)
        @test out ≈ convolve_templates(grid, transform_templates(grid, s.T), 100.0, 150.0)
    end

    @testset "StellarComponent parameter maps" begin
        c = StellarComponent(grid, s.T; σ_min=25.0)
        θ = [120.0, 200.0]
        @test from_free(c, to_free(c, θ)) ≈ θ
        @test to_free(c, θ)[1] ≈ 120.0 / s.vs
        @test from_free(c, [0.0, -50.0])[2] >= 25.0          # σ never drops below σ_min
        @test_throws ArgumentError to_free(c, [0.0, 20.0])
    end

    @testset "StellarComponent bounds" begin
        c = StellarComponent(grid, s.T; σ_min=25.0, σ_max=600.0, vmax=800.0)
        @test in_bounds(c, [0.0, 100.0])
        @test !in_bounds(c, [900.0, 100.0])
        @test !in_bounds(c, [0.0, 700.0])
        @test !in_bounds(c, [0.0, 10.0])
        @test !in_bounds(c, [NaN, 100.0])
    end

    @testset "PolynomialComponent" begin
        c = PolynomialComponent(10, 2)
        @test nparams(c) == 0 && ncolumns(c) == 6
        @test columns(c, grid, Float64[]) == hcat(legendre_basis(10, 2), -legendre_basis(10, 2))
        @test physical_weights(c, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5]) == [0.5, 1.5, 2.5]
        @test_throws ArgumentError PolynomialComponent(10, -1)
        @test_throws ArgumentError check_component(c, 11)
    end

    @testset "default interface" begin
        c = PolynomialComponent(10, 1)
        @test in_bounds(c, Float64[])
        @test to_free(c, Float64[]) == Float64[]
        @test from_free(c, Float64[]) == Float64[]
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "legendre_basis (cases)" begin
    @testset "simple" begin
        @test legendre_basis(3, 0) == ones(3, 1)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError legendre_basis(1, 2)
        @test_throws ArgumentError legendre_basis(10, -1)
    end
    @testset "realistic" begin
        # orthogonal on [-1, 1] with ∫P_k² = 2/(2k+1) (fine sampling, trapezoid weights)
        n = 20001
        P = legendre_basis(n, 6)
        w = fill(2 / (n - 1), n); w[[1, n]] ./= 2
        G = P' * (w .* P)
        @test G ≈ Diagonal(2 ./ (2 .* (0:6) .+ 1)) atol = 1e-6
    end
    @testset "pathological" begin
        P = legendre_basis(500, 40)                 # high degree stays bounded by 1
        @test maximum(abs, P) <= 1 + 1e-9
        @test legendre_basis(2, 3)[:, 4] ≈ [-1.0, 1.0]
    end
end

@testset "StellarComponent (cases)" begin
    s = synthetic_setup()
    grid = s.grid
    @testset "simple" begin
        c = StellarComponent(grid, s.T)
        @test (nparams(c), ncolumns(c)) == (2, 6)
        @test c.σ_min == 0.0 && c.σ_templ == 0.0
    end
    @testset "invalid input" begin
        @test_throws ArgumentError StellarComponent(grid, s.T; σ_min=100.0, σ_max=50.0)
        @test_throws ArgumentError StellarComponent(grid, s.T; σ_min=-1.0)
        @test_throws ArgumentError StellarComponent(grid, s.T; vmax=0.0)
        @test_throws ArgumentError StellarComponent(grid, s.T[1:1000, :])      # wrong template grid
        c = StellarComponent(grid, s.T; σ_min=25.0)
        @test_throws ArgumentError columns(c, grid, [0.0])
        @test_throws ArgumentError in_bounds(c, [0.0, 100.0, 1.0])
        @test_throws ArgumentError from_free(c, [0.0])
    end
    @testset "realistic" begin
        # free-coordinate round trip over a range of physical values
        c = StellarComponent(grid, s.T; σ_min=25.0, σ_max=600.0, vmax=1000.0)
        for v in (-900.0, -12.5, 0.0, 333.0), σ in (25.5, 80.0, 450.0)
            @test from_free(c, to_free(c, [v, σ])) ≈ [v, σ]
        end
        # the log-Jacobian matches the numerical derivative of from_free
        u = to_free(c, [100.0, 150.0])
        h = 1e-6
        J = [(from_free(c, u .+ h .* (1:2 .== j)) .- from_free(c, u .- h .* (1:2 .== j))) ./ 2h for j in 1:2]
        @test Specje.log_jacobian(c, u) ≈ log(abs(det(hcat(J...)))) rtol = 1e-6
    end
    @testset "pathological" begin
        c = StellarComponent(grid, s.T; σ_min=25.0, σ_max=600.0, vmax=1000.0)
        @test in_bounds(c, [1000.0, 600.0])          # bounds are inclusive
        @test !in_bounds(c, [0.0, Inf])
        @test from_free(c, [0.0, -800.0])[2] == 25.0  # e^u underflows: σ sits on σ_min
        @test_throws ArgumentError to_free(c, [0.0, 25.0])   # exactly σ_min has no free coordinate
    end
end

@testset "PolynomialComponent (cases)" begin
    @testset "simple" begin
        c = PolynomialComponent(4, 0)
        @test c.cols == [ones(4) -ones(4)]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError PolynomialComponent(1, 2)
        @test_throws ArgumentError physical_weights(PolynomialComponent(10, 1), ones(3))
    end
    @testset "realistic" begin
        # any signed polynomial is representable with non-negative pair weights
        c = PolynomialComponent(200, 3)
        coeffs = [0.3, -1.2, 0.0, 0.7]
        w = [max.(coeffs, 0); max.(-coeffs, 0)]
        @test c.cols * w ≈ legendre_basis(200, 3) * coeffs
        @test physical_weights(c, w) ≈ coeffs
    end
    @testset "pathological" begin
        # both halves of a pair active: only their difference is physical
        c = PolynomialComponent(10, 0)
        @test physical_weights(c, [5.0, 5.0]) == [0.0]
    end
end
