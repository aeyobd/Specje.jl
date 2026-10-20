using LinearAlgebra
using Random
using Statistics
using LogDensityProblems
import Specje: log_jacobian, loglike, log_prior, column_scales, split_free

@testset "BayesModel" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0, vmax=1000.0)
    model = SpectralModel(s.grid, star)
    θtrue, wtrue = [120.0, 200.0], [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    clean = design_matrix(model, θtrue) * wtrue
    σn = 0.02 * mean(clean)
    spec = Spectrum(s.dwave, clean .+ σn .* randn(Xoshiro(4), nd), fill(σn, nd))
    fd = FitData(spec)
    bm = BayesModel(model, fd, θtrue)
    u0 = initial_free(bm, θtrue)
    ll = LogDensityProblems.logdensity

    @testset "layout and interface" begin
        @test LogDensityProblems.dimension(bm) == 2
        @test LogDensityProblems.dimension(BayesModel(model, fd, θtrue; fit_tau=true)) == 3
        @test LogDensityProblems.dimension(BayesModel(model, fd, θtrue; fit_noise=true)) == 3
        @test LogDensityProblems.capabilities(bm) == LogDensityProblems.LogDensityOrder{1}()
        @test_throws ArgumentError split_free(bm, [1.0])
    end

    @testset "coordinates" begin
        bma = BayesModel(model, fd, θtrue; fit_tau=true, fit_noise=true)
        ua = initial_free(bma, θtrue)
        @test length(ua) == 4 && ua[3] ≈ log(bma.tau[1]) && ua[4] == 0.0
        ph = to_physical(bma, ua)
        @test ph.θ ≈ θtrue && ph.τ ≈ bma.tau && ph.s == [1.0]
        ph0 = to_physical(bm, u0)
        @test ph0.τ == bm.tau && ph0.s == [1.0]
    end

    @testset "density is finite, peaked, and walled" begin
        @test isfinite(ll(bm, u0))
        @test ll(bm, u0) > ll(bm, initial_free(bm, [170.0, 200.0]))
        @test ll(bm, u0) > ll(bm, initial_free(bm, [120.0, 320.0]))
        @test ll(bm, [100.0, 0.0]) == -Inf          # v = 5000 km/s, beyond vmax
        @test ll(bm, [0.0, 20.0]) == -Inf           # σ beyond σ_max
        @test ll(bm, [NaN, 0.0]) == -Inf
    end

    @testset "posterior = prior + Jacobian + marginal likelihood" begin
        ph = to_physical(bm, u0)
        um = first(split_free(bm, u0))
        manual = log_prior(bm.priors, ph.θ, Float64[], Float64[]) +
                 log_jacobian(model, um) + loglike(bm, ph.θ, ph.τ, ph.s)
        @test ll(bm, u0) ≈ manual
    end

    @testset "marginal likelihood equals the dense Gaussian" begin
        A = design_matrix(model, θtrue)
        Aw = A[fd.idx, :] .* fd.inv_noise
        τ = column_scales(bm, bm.tau)
        K = Symmetric(I + Aw * Diagonal(τ .^ 2) * Aw')
        b = fd.b
        brute = -(length(b) * log(2π) + logdet(K) + dot(b, K \ b)) / 2 + sum(log, fd.inv_noise)
        @test loglike(bm, θtrue, bm.tau, [1.0]) ≈ brute rtol = 1e-8
    end

    @testset "noise scale is identified" begin
        f(sv) = loglike(bm, θtrue, bm.tau, [sv])
        @test f(1.0) > f(3.0) && f(1.0) > f(0.3)
    end

    @testset "hyperparameters enter the density" begin
        bma = BayesModel(model, fd, θtrue; fit_tau=true, fit_noise=true)
        ua = initial_free(bma, θtrue)
        @test isfinite(ll(bma, ua))
        @test ll(bma, ua) != ll(bma, [ua[1:2]; ua[3] + 1.0; ua[4]])
        @test ll(bma, ua) > ll(bma, [ua[1:3]; 1.5])       # noise inflated 4.5×
    end

    @testset "constructor checks" begin
        @test_throws ArgumentError BayesModel(model, fd, θtrue; priors=default_priors(model)[1:1])
        @test_throws ArgumentError BayesModel(model, fd, θtrue; tau=[1.0, 2.0])
        @test_throws ArgumentError BayesModel(model, fd, θtrue; tau=[-1.0])
        @test BayesModel(model, fd, θtrue; priors=default_priors(model)) isa BayesModel
        @test BayesModel(model, fd, θtrue; tau=[7.0]).tau == [7.0]
    end

    @testset "MAP" begin
        start = u0 .+ [0.5, -0.2]
        r = map_estimate(bm, start)
        @test r.logp > ll(bm, start)
        @test r.u[1] * s.vs ≈ 120.0 atol = 10.0
        @test 25.0 + exp(r.u[2]) ≈ 200.0 atol = 40.0
    end

    @testset "gradient matches finite differences" begin
        fdiff(f, u; h=1e-5) = [(f(u .+ h .* (eachindex(u) .== i)) -
                                f(u .- h .* (eachindex(u) .== i))) / 2h for i in eachindex(u)]
        check(b, u) = (g = last(LogDensityProblems.logdensity_and_gradient(b, u));
                       isapprox(g, fdiff(v -> ll(b, v), u); rtol=1e-4))
        lp, g = LogDensityProblems.logdensity_and_gradient(bm, u0)
        @test lp ≈ ll(bm, u0) && length(g) == 2
        @test check(bm, u0 .+ 0.03)
        bma = BayesModel(model, fd, θtrue; fit_tau=true, fit_noise=true)
        @test check(bma, initial_free(bma, θtrue) .+ 0.03)
        @test LogDensityProblems.logdensity_and_gradient(bm, [0.0, 20.0]) == (-Inf, [0.0, 0.0])

        lsf = GaussianLSF(fill(60.0, nd))
        for l in (lsf, banded_gaussian(fill(60.0 / s.vs, nd)))
            gas = GasComponent(s.dwave, [GasGroup("A", Multiplet("a", 4260.0))], l;
                               σ_min=10.0, vmax=500.0)
            mg = SpectralModel(s.grid, star, gas, PolynomialComponent(nd, 1))
            θg = [120.0, 200.0, 30.0, 100.0]
            fg = design_matrix(mg, θg) * [wtrue; 3.0; 0.1; 0.0; 0.0; 0.0]
            sg = Spectrum(s.dwave, fg .+ σn .* randn(Xoshiro(5), nd), fill(σn, nd))
            bg = BayesModel(mg, FitData(sg), θg)
            @test check(bg, initial_free(bg, θg) .+ 0.03)
        end
    end

    @testset "weights given the nonlinear parameters" begin
        wp = weight_posterior(bm, u0)
        @test length(wp.mean) == 6
        @test maximum(abs, wp.mean .- wtrue) < 1.0
        @test sum(wp.mean) ≈ sum(wtrue) atol = 0.5
        draws = sample_weights(Xoshiro(1), wp, 50; positive=true, burn=20)
        @test size(draws) == (6, 50) && all(>=(0), draws)
        @test length(split_weights(model, vec(mean(draws; dims=2)))) == 1
    end

    @testset "short ensemble run" begin
        chain = sample_ensemble(Xoshiro(2), bm, u0; nwalkers=8, nsteps=120, ball_scale=0.02)
        @test size(chain.samples) == (2, 8, 120)
        @test all(isfinite, chain.logp)
        @test 0.02 < acceptance_fraction(chain) < 0.99
        late = flatten_chain(chain; burn=60)
        @test mean(late[1, :]) * s.vs ≈ 120.0 atol = 10.0
    end

    @testset "joint model" begin
        f = joint_fixture(fit_scale=true)
        θ = [120.0, 200.0, 1.0]
        flux = design_matrix(f.stars, θ[1:2]) * f.ws
        sp = Spectrum(f.j.dwave, flux, fill(0.01 * mean(flux), length(flux)))
        fdj = FitData(sp, f.phot)
        bmj = BayesModel(f.jm, fdj, θ; fit_noise=true)
        @test length(fdj.blocks) == 2 && length(bmj.priors.logs) == 2
        @test LogDensityProblems.dimension(bmj) == 3 + 2
        uj = initial_free(bmj, θ)
        @test isfinite(ll(bmj, uj))
        ph = to_physical(bmj, uj)
        @test ph.θ ≈ θ && ph.s == [1.0, 1.0]
        @test ll(bmj, uj) > ll(bmj, initial_free(bmj, [120.0, 200.0, 1.5]))
        g = last(LogDensityProblems.logdensity_and_gradient(bmj, uj .+ 0.03))
        h = 1e-5
        @test g[3] ≈ (ll(bmj, uj .+ 0.03 .+ h .* (1:5 .== 3)) -
                      ll(bmj, uj .+ 0.03 .- h .* (1:5 .== 3))) / 2h rtol = 1e-4
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "BayesModel and log_posterior (cases)" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0, σ_max=600.0, vmax=1000.0)
    model = SpectralModel(s.grid, star)
    θ = [120.0, 200.0]
    w = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    clean = design_matrix(model, θ) * w
    σn = 0.02 * mean(clean)
    fd = FitData(Spectrum(s.dwave, clean .+ σn .* randn(Xoshiro(81), nd), fill(σn, nd)))
    @testset "simple" begin
        bm = BayesModel(model, fd, θ)
        @test LogDensityProblems.dimension(bm) == 2
        @test to_physical(bm, initial_free(bm, θ)).θ ≈ θ
    end
    @testset "invalid input" begin
        @test_throws ArgumentError BayesModel(model, fd, [120.0])
        @test_throws ArgumentError BayesModel(model, fd, θ; tau=[-1.0])
        @test_throws ArgumentError BayesModel(model, fd, θ; tau=[1.0, 2.0])
        @test_throws ArgumentError BayesModel(model, fd, θ; priors=default_priors(model)[1:1])
        bm = BayesModel(model, fd, θ)
        @test_throws ArgumentError log_posterior(bm, [0.0])
        @test_throws ArgumentError to_physical(bm, [0.0, 1.0, 2.0])
    end
    @testset "realistic" begin
        # all hyperparameters on: layout, round trip, and a finite gradient
        bm = BayesModel(model, fd, θ; fit_tau=true, fit_noise=true)
        u = initial_free(bm, θ)
        @test length(u) == 4
        ph = to_physical(bm, u)
        @test ph.θ ≈ θ && ph.τ ≈ bm.tau && ph.s == [1.0]
        lp, g = log_posterior_and_gradient(bm, u)
        @test isfinite(lp) && all(isfinite, g) && lp ≈ log_posterior(bm, u)
        # the posterior prefers the true noise level to one twice as large
        @test log_posterior(bm, u) > log_posterior(bm, [u[1:3]; log(2.0)])
    end
    @testset "pathological" begin
        bm = BayesModel(model, fd, θ)
        # just inside the σ_max wall is allowed, just beyond is not (exp(log(x)) is not
        # exactly x, so the wall itself cannot be hit through the free coordinate)
        u_wall = [θ[1] / s.vs, log(600.0 - 25.0)]
        @test isfinite(log_posterior(bm, u_wall .- [0, 1e-9]))
        @test log_posterior(bm, u_wall .+ [0, 1e-9]) == -Inf
        @test log_posterior_and_gradient(bm, u_wall .+ [0, 1.0]) == (-Inf, [0.0, 0.0])
        # an absurd τ makes the evidence collapse but stays a number
        bm_tiny = BayesModel(model, fd, θ; tau=[1e-12])
        @test isfinite(log_posterior(bm_tiny, initial_free(bm_tiny, θ)))
        @test log_posterior(bm_tiny, initial_free(bm_tiny, θ)) < log_posterior(bm, initial_free(bm, θ))
    end
end

@testset "map_estimate and weight_posterior (cases)" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0, σ_max=600.0, vmax=1000.0)
    model = SpectralModel(s.grid, star)
    θ = [120.0, 200.0]
    w = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    clean = design_matrix(model, θ) * w
    σn = 0.01 * mean(clean)
    fd = FitData(Spectrum(s.dwave, clean .+ σn .* randn(Xoshiro(82), nd), fill(σn, nd)))
    bm = BayesModel(model, fd, θ)
    @testset "simple" begin
        m = map_estimate(bm, initial_free(bm, [100.0, 170.0]))
        @test to_physical(bm, m.u).θ ≈ θ atol = 10.0
        @test m.logp ≈ log_posterior(bm, m.u)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError map_estimate(bm, [0.0])
        @test_throws ArgumentError map_estimate(bm, [1e3, 0.0])        # zero density start
        @test_throws ArgumentError map_estimate(bm, [NaN, 0.0])
    end
    @testset "realistic" begin
        # at the MAP the gradient is ≈ 0, and the weights are close to the truth
        u_start = initial_free(bm, [100.0, 170.0])
        m = map_estimate(bm, u_start)
        _, g_start = log_posterior_and_gradient(bm, u_start)
        _, g_map = log_posterior_and_gradient(bm, m.u)
        @test norm(g_map) < 1e-3 * norm(g_start)
        wp = weight_posterior(bm, m.u)
        @test sum(wp.mean) ≈ sum(w) rtol = 0.05
    end
    @testset "pathological" begin
        # starting exactly at the mode: the search stays there
        m = map_estimate(bm, initial_free(bm, [100.0, 170.0]))
        again = map_estimate(bm, m.u)
        @test again.logp >= m.logp - 1e-8
        @test to_physical(bm, again.u).θ ≈ to_physical(bm, m.u).θ atol = 0.5
    end
end
