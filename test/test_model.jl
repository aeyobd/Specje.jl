import Specje: to_free, from_free, in_bounds
import Specje: columns, to_free, from_free, in_bounds

@testset "SpectralModel" begin
    s = synthetic_setup()
    grid = s.grid
    nd = length(grid.crop)
    c1 = StellarComponent(grid, s.T[:, 1:3]; σ_min=25.0)
    c2 = StellarComponent(grid, s.T[:, 4:6]; σ_min=25.0)
    poly = PolynomialComponent(nd, 1)
    m = SpectralModel(grid, c1, c2, poly)
    θ = [0.0, 100.0, 150.0, 200.0]

    @testset "bookkeeping" begin
        @test npix(m) == nd
        @test nparams(m) == 4
        @test ncolumns(m) == 3 + 3 + 4
        @test param_ranges(m) == [1:2, 3:4, 5:4]
        @test column_ranges(m) == [1:3, 4:6, 7:10]
    end

    @testset "design_matrix" begin
        A = design_matrix(m, θ)
        @test size(A) == (nd, 10)
        @test A[:, 1:3] ≈ columns(c1, grid, θ[1:2])
        @test A[:, 4:6] ≈ columns(c2, grid, θ[3:4])
        @test A[:, 7:10] == poly.cols
        @test_throws ArgumentError design_matrix(m, θ[1:3])
    end

    @testset "one-component convenience constructor" begin
        m1 = SpectralModel(grid, c1)
        @test nparams(m1) == 2 && ncolumns(m1) == 3
        @test design_matrix(m1, θ[1:2]) ≈ columns(c1, grid, θ[1:2])
    end

    @testset "split_weights" begin
        w = [1.0, 2, 3, 4, 5, 6, 0.3, 0.0, 0.0, 0.1]
        parts = split_weights(m, w)
        @test parts[1] == [1.0, 2, 3] && parts[2] == [4.0, 5, 6]
        @test parts[3] ≈ [0.3, -0.1]
        @test_throws ArgumentError split_weights(m, w[1:5])
    end

    @testset "parameter maps and bounds" begin
        @test from_free(m, to_free(m, θ)) ≈ θ
        @test length(to_free(m, θ)) == 4
        @test in_bounds(m, θ)
        @test !in_bounds(m, [0.0, 100.0, 150.0, 5000.0])
    end

    @testset "construction checks" begin
        @test_throws ArgumentError SpectralModel(grid, PolynomialComponent(nd + 1, 1))
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "SpectralModel and design_matrix (cases)" begin
    s = synthetic_setup()
    grid = s.grid
    nd = length(grid.crop)
    star = StellarComponent(grid, s.T; σ_min=25.0)
    @testset "simple" begin
        m = SpectralModel(grid, PolynomialComponent(nd, 0))
        @test design_matrix(m, Float64[]) == [ones(nd) -ones(nd)]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError SpectralModel(grid)                          # no components
        @test_throws ArgumentError SpectralModel(grid, PolynomialComponent(nd - 1, 0))
        @test_throws DimensionMismatch SpectralModel(grid, star; projection=ones(5, nd + 1))
        m = SpectralModel(grid, star)
        @test_throws ArgumentError design_matrix(m, [1.0, 2.0, 3.0])
        @test_throws ArgumentError to_free(m, [1.0])
        @test_throws ArgumentError in_bounds(m, Float64[])
        @test_throws ArgumentError split_weights(m, ones(7))
    end
    @testset "realistic" begin
        # bulge + disc + gas + continuum: columns, parameters and weights line up
        lsf = GaussianLSF(fill(60.0, nd))
        gas = GasComponent(s.dwave, [GasGroup("A", Multiplet("a", 4260.0))], lsf; σ_min=5.0)
        bulge = StellarComponent(grid, s.T[:, 1:3]; σ_min=25.0)
        disc = StellarComponent(grid, s.T[:, 4:6]; σ_min=25.0)
        poly = PolynomialComponent(nd, 2)
        m = SpectralModel(grid, bulge, disc, gas, poly)
        θ = [50.0, 220.0, -80.0, 60.0, 10.0, 40.0]
        @test param_ranges(m) == [1:2, 3:4, 5:6, 7:6]
        @test column_ranges(m) == [1:3, 4:6, 7:7, 8:13]
        A = design_matrix(m, θ)
        @test A[:, 7] ≈ columns(gas, grid, θ[5:6])
        @test from_free(m, to_free(m, θ)) ≈ θ
        w = collect(1.0:13.0)
        parts = split_weights(m, w)
        @test parts[3] == [7.0] && parts[4] ≈ [8.0, 9.0, 10.0] .- [11.0, 12.0, 13.0]
    end
    @testset "pathological" begin
        # a component with no parameters gets an empty parameter slice
        m = SpectralModel(grid, PolynomialComponent(nd, 1), star)
        @test param_ranges(m) == [1:0, 1:2]
        @test size(design_matrix(m, [0.0, 100.0])) == (nd, 4 + 6)
        # an all-zero projection is allowed: the model predicts nothing
        mz = SpectralModel(grid, star; projection=zeros(3, nd))
        @test design_matrix(mz, [0.0, 100.0]) == zeros(3, 6)
    end
end
