using FFTW
using LogDensityProblems
import Specje: sparse_matrix, columns

@testset "resolution-matrix projection" begin
    s = synthetic_setup()
    mw, vs = s.dwave, s.vs
    native = collect(range(mw[3], mw[end-2]; length=150))
    # asymmetric kernel whose width varies along the spectrum
    offs = collect(-4:4)
    bands = [exp(-(k - 0.3)^2 / (2 * (1.0 + 0.4i / 150)^2)) * (k < 0 ? 1.3 : 1.0)
             for k in offs, i in 1:150]
    R = normalize_rows(BandedLSF(bands, offs))
    # templates pre-broadened by a constant σ_templ (exactly, in Fourier space)
    broaden(T, σ) = irfft(transform_templates(s.grid, T) .* gaussian_losvd_ft(s.grid.freqs, 0.0, σ / vs),
                          s.grid.nfft, 1)[1:size(T, 1), :]
    σt = 40.0
    Tlib = broaden(s.T, σt)

    @testset "sparse resolution matrix" begin
        @test Matrix(sparse_matrix(R)) == dense_matrix(R)
    end

    @testset "projection options" begin
        P = pixel_projection(mw, native)
        @test pixel_projection(mw, native; resolution=R) ≈ sparse_matrix(R) * P
        @test_throws DimensionMismatch pixel_projection(mw, native; resolution=banded_gaussian(fill(1.0, 10)))
        @test_throws ArgumentError pixel_projection(mw, native; sampling=:nearest)

        Pp = pixel_projection(mw, native; sampling=:point)
        f = log.(mw)
        @test Pp * f ≈ [Specje.interp_linear(x, mw, f) for x in native] rtol = 1e-12
        @test all(sum(Pp; dims=2) .≈ 1)
        @test_throws ArgumentError pixel_projection(mw, [mw[1] - 1.0; native]; sampling=:point)
    end

    @testset "gas without its own LSF" begin
        c = GasComponent(mw, [GasGroup("g", Multiplet("x", 4350.0))])
        m = columns(c, s.grid, [0.0, 100.0])[:, 1] .* c.widths
        @test sum(m) ≈ 1.0 rtol = 1e-9
        @test profile_variance(m) ≈ (100.0 / vs)^2 + 1 / 12 rtol = 1e-4
    end

    @testset "σ_templ removes the templates' own width" begin
        sharp = StellarComponent(s.grid, s.T; σ_min=1.0)
        broad = StellarComponent(s.grid, Tlib; σ_min=σt, σ_templ=σt)
        θ = [120.0, 90.0]
        @test columns(broad, s.grid, θ) ≈ columns(sharp, s.grid, θ) rtol = 1e-8
        @test columns(broad, s.grid, [0.0, σt]) ≈ columns(StellarComponent(s.grid, Tlib), s.grid, [0.0, 0.0]) rtol = 1e-10
        @test_throws ArgumentError StellarComponent(s.grid, Tlib; σ_min=10.0, σ_templ=σt)
        @test_throws ArgumentError StellarComponent(s.grid, Tlib; σ_templ=-1.0)
    end

    gas = GasComponent(mw, [GasGroup("A", Multiplet("a", 4260.0), Multiplet("b", 4400.0))];
                       σ_min=5.0, vmax=500.0)
    θtrue = [120.0, 90.0, 30.0, 40.0]
    wtrue = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0, 3.0, 1.5]

    @testset "fit through R·P recovers the truth ($sampling)" for sampling in (:average, :point)
        P = pixel_projection(mw, native; resolution=R, sampling=sampling)
        truth = SpectralModel(s.grid, StellarComponent(s.grid, s.T; σ_min=1.0), gas; projection=P)
        clean = design_matrix(truth, θtrue) * wtrue
        sp = Spectrum(native, clean, fill(0.002 * mean(clean), 150))
        m = SpectralModel(s.grid, StellarComponent(s.grid, Tlib; σ_min=σt, σ_templ=σt), gas; projection=P)
        r = fit_kinematics(m, sp, [100.0, 120.0, 0.0, 60.0])
        @test r.θ ≈ θtrue rtol = 1e-3
        @test r.weights[7:8] ≈ wtrue[7:8] rtol = 1e-3
        # ignoring the templates' width biases σ★ to √(σ² − σ_templ²), with an equally good fit
        m0 = SpectralModel(s.grid, StellarComponent(s.grid, Tlib; σ_min=1.0), gas; projection=P)
        r0 = fit_kinematics(m0, sp, [100.0, 120.0, 0.0, 60.0])
        @test r0.θ[2] ≈ sqrt(90.0^2 - σt^2) rtol = 1e-3
    end

    @testset "gradient through R·P" begin
        P = pixel_projection(mw, native; resolution=R)
        m = SpectralModel(s.grid, StellarComponent(s.grid, Tlib; σ_min=σt, σ_templ=σt), gas; projection=P)
        clean = design_matrix(m, θtrue) * wtrue
        σn = 0.02 * mean(clean)
        sp = Spectrum(native, clean .+ σn .* randn(Xoshiro(7), 150), fill(σn, 150))
        bm = BayesModel(m, FitData(sp), θtrue)
        u = initial_free(bm, θtrue) .+ 0.03
        lp, g = LogDensityProblems.logdensity_and_gradient(bm, u)
        h = 1e-3
        fd = [(log_posterior(bm, u .+ h .* (1:4 .== i)) - log_posterior(bm, u .- h .* (1:4 .== i))) / 2h
              for i in 1:4]
        @test g ≈ fd rtol = 1e-3
    end

    @testset "constant velocity resolution" begin
        lib = log_library()                                   # 2.5 Å FWHM, log grid
        σlib = library_sigma_kms(lib)
        @test σlib ≈ fwhm_to_sigma_kms(2.5, lib.wave)
        T = match_resolution(lib, 200.0)
        ref = match_resolution(templates(lib), lib.wave, lib.lsf_fwhm, lib.wave,
                               GaussianLSF(fill(200.0, nwave(lib))), velscale(lib.wave))
        @test T ≈ ref
        @test_throws ArgumentError match_resolution(lib, 0.5 * minimum(σlib))
        @test_throws ArgumentError library_sigma_kms(flat_library())
    end
end
