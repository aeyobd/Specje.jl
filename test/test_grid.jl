@testset "grid" begin
    @testset "pixel_edges" begin
        @test pixel_edges([1.0, 2.0, 3.0]) ≈ [0.5, 1.5, 2.5, 3.5]
        @test length(pixel_edges(rand(7) .+ (1:7))) == 8
    end

    @testset "loggrid" begin
        g = loggrid(3000.0, 9000.0, 70.0)
        @test first(g) == 3000.0
        @test last(g) <= 9000.0
        @test velscale(g) ≈ 70.0 rtol = 1e-10
        @test is_loguniform(g)
        @test !is_loguniform(collect(3000.0:10.0:4000.0))
    end

    @testset "log_pixel_edges" begin
        g = loggrid(3000.0, 4000.0, 100.0)
        e = log_pixel_edges(g)
        @test all(@. e[1:end-1] < g < e[2:end])
        @test @.(sqrt(e[1:end-1] * e[2:end])) ≈ g
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "pixel_edges (cases)" begin
    @testset "simple" begin
        @test pixel_edges([0.0, 1.0]) == [-0.5, 0.5, 1.5]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError pixel_edges([1.0])
        @test_throws ArgumentError pixel_edges([1.0, 3.0, 2.0])
    end
    @testset "realistic" begin
        # irregular centres: interior edges are midpoints, outer edges mirror them
        x = [1.0, 2.0, 4.0, 8.0]
        @test pixel_edges(x) == [0.5, 1.5, 3.0, 6.0, 10.0]
    end
    @testset "pathological" begin
        # large offsets: edges stay ordered despite limited relative precision
        x = 1e8 .+ collect(0.0:1.0:5.0)
        e = pixel_edges(x)
        @test issorted(e) && length(e) == 7
        @test e[2] - e[1] ≈ 1.0
    end
end

@testset "log_pixel_edges (cases)" begin
    @testset "simple" begin
        @test log_pixel_edges([1.0, 4.0]) ≈ [0.5, 2.0, 8.0]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError log_pixel_edges([0.0, 1.0])
        @test_throws ArgumentError log_pixel_edges([-2.0, -1.0])
    end
    @testset "realistic" begin
        g = loggrid(3600.0, 9000.0, 69.0)
        e = log_pixel_edges(g)
        @test all(≈(69.0; rtol=1e-9), C_KMS .* diff(log.(e)))   # every pixel 69 km/s wide
    end
    @testset "pathological" begin
        # two centres a factor 1e6 apart still give geometric midpoints
        @test log_pixel_edges([1.0, 1e6])[2] ≈ 1e3
    end
end

@testset "loggrid (cases)" begin
    @testset "simple" begin
        g = loggrid(1.0, exp(2 * 100.0 / C_KMS), 100.0)
        @test length(g) == 3
    end
    @testset "invalid input" begin
        @test_throws ArgumentError loggrid(0.0, 10.0, 50.0)
        @test_throws ArgumentError loggrid(10.0, 5.0, 50.0)
        @test_throws ArgumentError loggrid(1.0, 10.0, 0.0)
        @test_throws ArgumentError loggrid(1.0, 10.0, -50.0)
    end
    @testset "realistic" begin
        g = loggrid(3600.0, 10400.0, 69.0)          # SDSS
        @test length(g) == floor(Int, C_KMS * log(10400 / 3600) / 69.0) + 1
        @test last(g) <= 10400.0 && last(g) * exp(69.0 / C_KMS) > 10400.0
    end
    @testset "pathological" begin
        @test loggrid(5000.0, 5000.0, 50.0) == [5000.0]        # zero-width range: one pixel
        @test loggrid(5000.0, 5000.1, 1e4) == [5000.0]         # step larger than the range
    end
end
