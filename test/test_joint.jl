import Specje: columns, phot_columns, split_theta, to_free, from_free, in_bounds

struct NoPhotComponent <: Component end
Specje.ncolumns(::NoPhotComponent) = 1
Specje.columns(::NoPhotComponent, grid, θ) = zeros(length(grid.crop), 1)

@testset "photometry in the model" begin
    j = joint_setup()
    nd, vs = length(j.dwave), j.vs
    T = templates(j.narrow)
    star = StellarComponent(j.grid, T; σ_min=25.0, vmax=1000.0, phot=j.bands)
    lsf = GaussianLSF(fill(60.0, nd))
    gas = GasComponent(j.dwave, [GasGroup("A", Multiplet("a", 4260.0), Multiplet("b", 4340.0))],
                       lsf; σ_min=10.0, vmax=500.0)
    poly = PolynomialComponent(nd, 1)
    stars = SpectralModel(j.grid, star)
    θs, ws = [120.0, 200.0], [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    bands_true = j.bands.M * ws
    phot = Photometry(j.filters, bands_true, 0.02 .* bands_true)

    @testset "synthetic band fluxes are sane" begin
        @test all(>(0), j.bands.M)
        @test photometry_matrix(stars, j.phot0, θs) == j.bands.M
    end

    @testset "component photometry columns" begin
        pc = phot_columns(poly, phot, Float64[])
        @test size(pc) == (4, 4) && all(iszero, pc)
        g = phot_columns(gas, phot, [0.0, 100.0])
        f = phot.rest[1]
        @test size(g) == (4, 2)
        @test g[1, 1] ≈ transmission(f, 4260.0) * 4260.0 / f.norm
        @test g[1, 2] ≈ transmission(f, 4340.0) * 4340.0 / f.norm
        @test all(iszero, g[2:4, :])                              # redder filters miss these lines
        shift = exp(300.0 / C_KMS)
        g300 = phot_columns(gas, phot, [300.0, 100.0])
        @test g300[1, 1] ≈ transmission(f, 4260.0 * shift) * 4260.0 * shift / f.norm
    end

    @testset "photometry_matrix blocks" begin
        full = SpectralModel(j.grid, star, gas, poly)
        P = photometry_matrix(full, phot, [θs; 30.0; 100.0])
        @test size(P) == (4, 12)
        @test P[:, 1:6] == j.bands.M
        @test P[:, 7:8] == phot_columns(gas, phot, [30.0, 100.0])
        @test all(iszero, P[:, 9:12])
    end

    @testset "JointModel bookkeeping and stacking" begin
        jm = JointModel(stars, phot)
        @test (nparams(jm), ncolumns(jm), npix(jm)) == (2, 6, nd)
        A = design_matrix(jm, θs)
        @test size(A) == (nd + 4, 6)
        @test A[1:nd, :] ≈ design_matrix(stars, θs)
        @test A[nd+1:end, :] ≈ j.bands.M
        @test split_weights(jm, ws) == split_weights(stars, ws)

        jms = JointModel(stars, phot; fit_scale=true)
        @test nparams(jms) == 3
        As = design_matrix(jms, [θs; 1.3])
        @test As[1:nd, :] ≈ 1.3 .* design_matrix(stars, θs)
        @test As[nd+1:end, :] ≈ j.bands.M
        @test_throws ArgumentError design_matrix(jms, θs)
    end

    @testset "component checks" begin
        @test_throws ArgumentError JointModel(SpectralModel(j.grid, StellarComponent(j.grid, T)), phot)
        flipped = Photometry(j.filters[end:-1:1], reverse(bands_true), reverse(bands_true))
        @test_throws ArgumentError JointModel(stars, flipped)
        @test_throws ArgumentError StellarComponent(j.grid, T; phot=BandTemplates(["x"], ones(4, 3)))
        @test_throws ArgumentError JointModel(SpectralModel(j.grid, NoPhotComponent()), phot)
        @test_throws ArgumentError JointModel(stars, phot; s_min=2.0, s_max=1.0)
        @test JointModel(SpectralModel(j.grid, star, gas, poly), phot) isa JointModel
    end

    @testset "parameter maps" begin
        jms = JointModel(stars, phot; fit_scale=true, s_min=0.1, s_max=10.0)
        θ = [120.0, 200.0, 1.3]
        @test from_free(jms, to_free(jms, θ)) ≈ θ
        @test to_free(jms, θ)[end] ≈ log(1.3)
        @test split_theta(jms, θ) == ([120.0, 200.0], 1.3)
        @test in_bounds(jms, θ)
        @test !in_bounds(jms, [120.0, 200.0, 50.0])
        @test !in_bounds(jms, [120.0, 200.0, 0.01])
        @test_throws ArgumentError to_free(jms, [120.0, 200.0, -1.0])
        jm = JointModel(stars, phot)
        @test from_free(jm, to_free(jm, θs)) ≈ θs
        @test split_theta(jm, θs) == (θs, 1.0)
    end

    @testset "FitData stacking" begin
        flux = design_matrix(stars, θs) * ws
        spec = Spectrum(j.dwave, flux, fill(0.01 * mean(flux), nd))
        fd = FitData(spec, phot)
        @test length(fd.b) == nd + 4
        @test fd.idx == collect(1:nd+4)
        @test fd.inv_noise[end] ≈ 1 / phot.err[end]
        bad = Photometry(j.filters, [bands_true[1:3]; NaN], 0.02 .* bands_true)
        fdb = FitData(spec, bad)
        @test length(fdb.b) == nd + 3 && last(fdb.idx) == nd + 3
    end

    @testset "joint fit, consistent calibration" begin
        flux = design_matrix(stars, θs) * ws
        spec = Spectrum(j.dwave, flux, fill(0.01 * mean(flux), nd))
        jm = JointModel(stars, phot)
        res = fit_kinematics(jm, spec, [0.0, 100.0])
        @test res.θ[1] ≈ 120.0 atol = 1.0
        @test res.θ[2] ≈ 200.0 atol = 2.0
        @test res.weights ≈ ws atol = 0.1
        @test res.ndata == nd + 4
        @test model_photometry(jm, res) ≈ bands_true rtol = 0.02
        @test length(model_spectrum(jm, res)) == nd
        @test model_flux(jm, res) == model_spectrum(jm, res)
        @test spectral_scale(jm, res) == 1.0
    end

    @testset "spectrum off by a calibration factor" begin
        flux = 1.3 .* (design_matrix(stars, θs) * ws)
        spec = Spectrum(j.dwave, flux, fill(0.01 * mean(flux), nd))

        jms = JointModel(stars, phot; fit_scale=true)
        res = fit_kinematics(jms, spec, [0.0, 100.0, 1.0])
        @test spectral_scale(jms, res) ≈ 1.3 rtol = 0.02
        @test res.θ[1] ≈ 120.0 atol = 2.0
        @test res.θ[2] ≈ 200.0 atol = 4.0
        @test res.weights ≈ ws atol = 0.15

        naive = fit_kinematics(JointModel(stars, phot), spec, [0.0, 100.0])
        @test reduced_chi2(naive) > reduced_chi2(res)                             # tension without the scale
        @test reduced_chi2(res) < 1
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "JointModel (cases)" begin
    f = joint_fixture(fit_scale=true)
    nd = length(f.j.dwave)
    θ = [120.0, 200.0, 1.0]
    flux = design_matrix(f.stars, θ[1:2]) * f.ws
    spec = Spectrum(f.j.dwave, flux, fill(0.01 * mean(flux), nd))
    @testset "simple" begin
        @test nparams(f.jm) == 3 && npix(f.jm) == nd && ncolumns(f.jm) == 6
        @test size(design_matrix(f.jm, θ)) == (nd + 4, 6)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError design_matrix(f.jm, θ[1:2])
        @test_throws ArgumentError in_bounds(f.jm, [θ; 1.0])
        @test_throws ArgumentError from_free(f.jm, [0.0])
        @test_throws ArgumentError fit_kinematics(f.jm, Spectrum(f.j.dwave[1:10], ones(10), ones(10)), θ)
    end
    @testset "realistic" begin
        # spectrum 30 % brighter than the photometry (aperture loss), with noise
        rng = Xoshiro(51)
        σn = 0.01 * mean(flux)
        noisy = Spectrum(f.j.dwave, 1.3 .* flux .+ σn .* randn(rng, nd), fill(σn, nd))
        r = fit_kinematics(f.jm, noisy, [100.0, 150.0, 1.0])
        @test spectral_scale(f.jm, r) ≈ 1.3 rtol = 0.03
        @test model_photometry(f.jm, r) ≈ f.j.bands.M * f.ws rtol = 0.05
        # the model spectrum is on the photometric scale: the data are s × it
        @test model_spectrum(f.jm, r) ≈ design_matrix(f.stars, r.θ[1:2]) * r.weights
        @test model_flux(f.jm, r) == model_spectrum(f.jm, r)
    end
    @testset "pathological" begin
        # the scale bounds are inclusive
        jm = JointModel(f.stars, f.phot; fit_scale=true, s_min=0.5, s_max=2.0)
        @test in_bounds(jm, [120.0, 200.0, 0.5]) && in_bounds(jm, [120.0, 200.0, 2.0])
        # no usable photometric point: the data reduce to the spectrum alone (one block)
        dead = Photometry(f.j.filters, fill(NaN, 4), ones(4))
        fd = FitData(spec, dead)
        @test fd.blocks == [1:nd] && length(fd.b) == nd
        r = fit_kinematics(JointModel(f.stars, dead), spec, [100.0, 150.0])
        @test r.θ ≈ [120.0, 200.0] atol = 1.0
    end
end
