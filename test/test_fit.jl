using Statistics
using Random

@testset "fitting" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=0.5 * s.vs, vmax=1000.0)
    model = SpectralModel(s.grid, star)
    θtrue = [120.0, 200.0]
    wtrue = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    clean = design_matrix(model, θtrue) * wtrue
    quiet(flux; level) = Spectrum(s.dwave, flux, fill(level * mean(flux), nd))

    @testset "FitData and solve_weights" begin
        spec = quiet(clean; level=0.01)
        fd = FitData(spec)
        @test length(fd.b) == nd
        w, chi2 = solve_weights(design_matrix(model, θtrue), fd)
        @test w ≈ wtrue atol = 1e-6
        @test chi2 < 1e-6
    end

    @testset "masked pixels are excluded" begin
        good = trues(nd); good[1:20] .= false
        spec = Spectrum(s.dwave, clean, fill(0.01, nd); good=good)
        @test length(FitData(spec).idx) == nd - 20
        @test_throws ArgumentError FitData(Spectrum(s.dwave, clean, fill(0.01, nd); good=falses(nd)))
    end

    @testset "chi2 is minimal at the truth" begin
        fd = FitData(quiet(clean; level=0.01))
        c0 = profile_chi2(model, fd, θtrue)
        @test c0 < profile_chi2(model, fd, θtrue .+ [30.0, 0.0])
        @test c0 < profile_chi2(model, fd, θtrue .+ [0.0, 50.0])
        @test profile_chi2(model, fd, [5000.0, 200.0]) == Inf
    end

    @testset "noiseless recovery" begin
        res = fit_kinematics(model, quiet(clean; level=0.01), [0.0, 100.0])
        @test res.θ[1] ≈ 120.0 atol = 0.5
        @test res.θ[2] ≈ 200.0 atol = 1.0
        @test res.weights ≈ wtrue atol = 0.1
        @test all(>=(0), res.weights)
        @test res.chi2 < 10
        @test res.nactive == count(>(0), res.weights)
        @test model_flux(model, res) ≈ clean rtol = 1e-2
    end

    @testset "noisy recovery" begin
        rng = Xoshiro(7)
        σn = 0.02 * mean(clean)
        spec = Spectrum(s.dwave, clean .+ σn .* randn(rng, nd), fill(σn, nd))
        res = fit_kinematics(model, spec, [0.0, 100.0])
        @test res.θ[1] ≈ 120.0 atol = 15.0
        @test res.θ[2] ≈ 200.0 atol = 30.0
        @test 0.8 < reduced_chi2(res) < 1.25
    end

    @testset "masked fit" begin
        good = trues(nd); good[rand(Xoshiro(3), 1:nd, 40)] .= false
        spec = Spectrum(s.dwave, clean, fill(0.01 * mean(clean), nd); good=good)
        res = fit_kinematics(model, spec, [0.0, 100.0])
        @test res.θ[2] ≈ 200.0 atol = 2.0
    end

    @testset "polynomial terms recovered at fixed kinematics" begin
        poly = PolynomialComponent(nd, 2)
        m = SpectralModel(s.grid, star, poly)
        c = [0.2, -0.1, 0.05]
        pos, neg = max.(c, 0), max.(-c, 0)
        flux = design_matrix(m, θtrue) * [wtrue; pos; neg]
        w, _ = solve_weights(design_matrix(m, θtrue), FitData(quiet(flux; level=0.01)))
        parts = split_weights(m, w)
        @test parts[1] ≈ wtrue atol = 1e-4
        @test parts[2] ≈ c atol = 1e-4
    end

    @testset "input checks" begin
        short = Spectrum(s.dwave[1:10], ones(10), ones(10))
        @test_throws ArgumentError fit_kinematics(model, short, θtrue)
        @test_throws ArgumentError fit_kinematics(model, quiet(clean; level=0.01), [0.0])
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "FitData (cases)" begin
    @testset "simple" begin
        fd = FitData(Spectrum([1.0, 2.0], [4.0, 6.0], [2.0, 3.0]))
        @test fd.b == [2.0, 2.0] && fd.inv_noise == [0.5, 1 / 3] && fd.blocks == [1:2]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError FitData([1.0, 2.0], [1.0], [1, 2])
        @test_throws ArgumentError FitData([1.0, 2.0], [1.0, 1.0], [1, 2], [1:1])   # blocks miss a row
        @test_throws ArgumentError FitData(Spectrum([1.0, 2.0], [NaN, NaN], [1.0, 1.0]))
    end
    @testset "realistic" begin
        # masked pixels drop out; idx points back at the right rows
        good = trues(10); good[[2, 7]] .= false
        fd = FitData(Spectrum(collect(1.0:10.0), collect(1.0:10.0), fill(0.5, 10); good=good))
        @test fd.idx == [1, 3, 4, 5, 6, 8, 9, 10]
        @test fd.b == 2 .* fd.idx
    end
    @testset "pathological" begin
        # a single good pixel is enough to build the data
        good = falses(5); good[3] = true
        fd = FitData(Spectrum(collect(1.0:5.0), ones(5), ones(5); good=good))
        @test fd.idx == [3] && fd.blocks == [1:1]
    end
end

@testset "solve_weights and profile_chi2 (cases)" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0, vmax=1000.0)
    model = SpectralModel(s.grid, star)
    θ = [120.0, 200.0]
    w = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    clean = design_matrix(model, θ) * w
    fd = FitData(Spectrum(s.dwave, clean, fill(0.01, nd)))
    @testset "simple" begin
        weights, chi2 = solve_weights(design_matrix(model, θ), fd)
        @test weights ≈ w atol = 1e-8
        @test chi2 < 1e-10
    end
    @testset "invalid input" begin
        @test_throws DimensionMismatch solve_weights(ones(10, 6), fd)
        @test_throws ArgumentError profile_chi2(model, fd, [120.0])
    end
    @testset "realistic" begin
        # χ² at the truth for noisy data is ≈ N, and its curvature in σ is positive
        rng = Xoshiro(21)
        σn = 0.02 * mean(clean)
        noisy = FitData(Spectrum(s.dwave, clean .+ σn .* randn(rng, nd), fill(σn, nd)))
        c0 = profile_chi2(model, noisy, θ)
        @test abs(c0 - nd) < 4sqrt(2nd)
        curvature = profile_chi2(model, noisy, θ .+ [0, 5]) - 2c0 + profile_chi2(model, noisy, θ .- [0, 5])
        @test curvature > 0
    end
    @testset "pathological" begin
        @test profile_chi2(model, fd, [NaN, 200.0]) == Inf
        @test profile_chi2(model, fd, [0.0, 1e6]) == Inf
        # all-zero data: the best weights are zero and the χ² vanishes
        zero_fd = FitData(Spectrum(s.dwave, zeros(nd), fill(0.01, nd)))
        weights, chi2 = solve_weights(design_matrix(model, θ), zero_fd)
        @test all(iszero, weights) && chi2 == 0
    end
end

@testset "fit_kinematics (cases)" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0, vmax=1000.0)
    model = SpectralModel(s.grid, star)
    w = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    clean = design_matrix(model, [120.0, 200.0]) * w
    spec = Spectrum(s.dwave, clean, fill(0.005 * mean(clean), nd))
    @testset "simple" begin
        r = fit_kinematics(model, spec, [100.0, 180.0])
        @test r.θ ≈ [120.0, 200.0] atol = 0.5
        @test r.converged
    end
    @testset "invalid input" begin
        @test_throws ArgumentError fit_kinematics(model, spec, [0.0, 150.0]; iterations=0)
        @test_throws ArgumentError fit_kinematics(model, spec, [0.0, 150.0]; tol=0.0)
        @test_throws ArgumentError fit_kinematics(model, spec, [0.0, 20.0])     # σ0 below σ_min
        @test_throws ArgumentError fit_kinematics(model, spec, [0.0, 150.0, 1.0])
    end
    @testset "realistic" begin
        # stars + gas + continuum, noisy, started 60 km/s and 50 km/s off
        lsf = GaussianLSF(fill(60.0, nd))
        gas = GasComponent(s.dwave, [GasGroup("A", Multiplet("a", 4260.0), Multiplet("b", 4400.0))],
                           lsf; σ_min=5.0, vmax=500.0)
        poly = PolynomialComponent(nd, 1)
        m = SpectralModel(s.grid, star, gas, poly)
        θ = [120.0, 200.0, 40.0, 80.0]
        flux = design_matrix(m, θ) * [w; 3.0; 1.0; 0.2; 0.0; 0.0; 0.05]
        σn = 0.01 * mean(flux)
        noisy = Spectrum(s.dwave, flux .+ σn .* randn(Xoshiro(31), nd), fill(σn, nd))
        r = fit_kinematics(m, noisy, [60.0, 150.0, 0.0, 60.0])
        @test r.θ[1:2] ≈ θ[1:2] atol = 10.0
        @test r.θ[3:4] ≈ θ[3:4] atol = 10.0
        @test 0.8 < reduced_chi2(r) < 1.25
        @test length(model_flux(m, r)) == nd
    end
    @testset "pathological" begin
        # a model without nonlinear parameters needs no search
        pm = SpectralModel(s.grid, PolynomialComponent(nd, 2))
        r = fit_kinematics(pm, Spectrum(s.dwave, fill(2.0, nd), fill(0.1, nd)), Float64[])
        @test r.θ == Float64[] && r.converged && r.iterations == 0
        @test split_weights(pm, r.weights)[1] ≈ [2.0, 0.0, 0.0] atol = 1e-10
        # one iteration is not enough to converge, and the result says so
        r1 = fit_kinematics(model, spec, [0.0, 150.0]; iterations=1)
        @test !r1.converged && r1.iterations == 1
        # zero signal: weights vanish and χ² is zero, wherever θ ends up
        rz = fit_kinematics(model, Spectrum(s.dwave, zeros(nd), fill(0.01, nd)), [0.0, 150.0])
        @test all(iszero, rz.weights) && rz.chi2 == 0 && rz.nactive == 0
    end
end
