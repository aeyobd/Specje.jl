@testset "resolution matching" begin
    vs = 50.0
    n = 600
    twave = 4000.0 .* exp.((0:n-1) .* vs ./ C_KMS)
    σt = 100.0
    tfwhm = sigma_kms_to_fwhm(σt, twave)
    line = [exp(-(i - 300.0)^2 / (2 * 4.0^2)) for i in 0:n-1]    # σ_line = 4 px

    @testset "sigma_diff_kms" begin
        @test sigma_diff_kms(100.0, 60.0) ≈ 80.0
        @test_throws ArgumentError sigma_diff_kms([100.0, 50.0], [60.0, 60.0])
        @test sigma_diff_kms([100.0, 50.0], [60.0, 60.0]; strict=false) ≈ [80.0, 0.0]
    end

    @testset "Gaussian data LSF" begin
        dlsf = GaussianLSF(fill(200.0, n))
        out = match_resolution(reshape(line, :, 1), twave, tfwhm, twave, dlsf, vs)
        σdiff_pix2 = (200.0^2 - σt^2) / vs^2
        # exactly σ_diff² is added (the kernel corrects for its 1/12 px² of pixel integration)
        @test profile_variance(out[:, 1]) ≈ 16 + σdiff_pix2 rtol = 1e-4
        @test sum(out) ≈ sum(line) rtol = 1e-10        # flux preserved
    end

    @testset "banded data LSF" begin
        R = banded_gaussian(fill(4.0, n))                # σ = 200 km/s
        σeff = inst_sigma_kms(R, vs)[n÷2]
        # strict=false: truncated rows at the spectrum edges look slightly sharper
        out = match_resolution(reshape(line, :, 1), twave, tfwhm, twave, R, vs; strict=false)
        @test profile_variance(out[:, 1]) ≈ 16 + (σeff^2 - σt^2) / vs^2 rtol = 1e-4
    end

    @testset "data frame offset uses interpolation" begin
        dwave = twave[1:2:end]                            # coarser, different length
        dlsf = GaussianLSF(fill(200.0, length(dwave)))
        out = match_resolution(reshape(line, :, 1), twave, tfwhm, dwave, dlsf, vs)
        @test size(out) == (n, 1)
    end

    @testset "library version" begin
        lib = log_library()
        dlsf = GaussianLSF(lib.wave, 4.0)
        T = match_resolution(lib, lib.wave, dlsf)
        @test size(T) == (nwave(lib), ntemplates(lib))
        @test T[50:550, 1] ≈ templates(lib)[50:550, 1] rtol = 1e-9   # linear ramp survives
        @test_throws ArgumentError match_resolution(lib, lib.wave, GaussianLSF(lib.wave, 2.0))
        @test_throws ArgumentError match_resolution(flat_library(), lib.wave, dlsf)
        @test_throws ArgumentError match_resolution(toy_library(), lib.wave, dlsf)
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "sigma_diff_kms (cases)" begin
    @testset "simple" begin
        @test sigma_diff_kms(5.0, 3.0) ≈ 4.0
    end
    @testset "invalid input" begin
        @test_throws ArgumentError sigma_diff_kms(-1.0, 0.0)
        @test_throws ArgumentError sigma_diff_kms(10.0, [5.0, -5.0])
        @test_throws ArgumentError sigma_diff_kms(50.0, 60.0)            # strict
    end
    @testset "realistic" begin
        # MILES (2.5 Å) templates to a spectrograph whose FWHM grows from 3 to 4 Å
        wave = collect(4000.0:500.0:7000.0)
        σ_data = fwhm_to_sigma_kms(collect(range(3.0, 4.0; length=7)), wave)
        σ_lib = fwhm_to_sigma_kms(2.5, wave)
        d = sigma_diff_kms(σ_data, σ_lib)
        @test d .^ 2 .+ σ_lib .^ 2 ≈ σ_data .^ 2
        @test all(>(0), d)
    end
    @testset "pathological" begin
        @test sigma_diff_kms(60.0, 60.0) == 0.0                         # already matched
        @test sigma_diff_kms(0.0, 0.0) == 0.0
        @test sigma_diff_kms([50.0, 70.0], 60.0; strict=false) ≈ [0.0, sqrt(70.0^2 - 60.0^2)]
    end
end

@testset "match_resolution (cases)" begin
    lib = log_library()                       # 2.5 Å, 50 km/s
    @testset "simple" begin
        # target equal to the library resolution: nothing to do
        same = GaussianLSF(lib.wave, lib.lsf_fwhm)
        @test match_resolution(lib, lib.wave, same) ≈ templates(lib)
    end
    @testset "invalid input" begin
        T = templates(lib)
        good_lsf = GaussianLSF(lib.wave, 4.0)
        @test_throws DimensionMismatch match_resolution(T[1:10, :], lib.wave, 2.5, lib.wave, good_lsf, 50.0)
        @test_throws DimensionMismatch match_resolution(T, lib.wave, 2.5, lib.wave[1:10], good_lsf, 50.0)
        @test_throws ArgumentError match_resolution(T, lib.wave, 2.5, lib.wave, good_lsf, 0.0)
    end
    @testset "realistic" begin
        # a narrow absorption line ends up at the data resolution
        n = nwave(lib)
        line = [1 - 0.5 * exp(-(i - 300)^2 / 2) for i in 1:n]      # σ = 1 px intrinsic
        twave = lib.wave
        T = reshape(line, :, 1)
        σ_target = 200.0
        out = match_resolution(T, twave, fill(0.0, n), twave, GaussianLSF(fill(σ_target, n)), 50.0)
        depth = 1 .- out[:, 1]
        @test profile_variance(depth[250:350]) ≈ 1 + (σ_target / 50)^2 rtol = 1e-3
        @test sum(depth) ≈ sum(1 .- line) rtol = 1e-8                  # equivalent width kept
    end
    @testset "pathological" begin
        # data sharper than the library in a few pixels: strict refuses, non-strict leaves
        # those pixels unsmoothed
        σ_data = fill(200.0, nwave(lib)); σ_data[1:10] .= 10.0
        dlsf = GaussianLSF(σ_data)
        @test_throws ArgumentError match_resolution(lib, lib.wave, dlsf)
        out = match_resolution(lib, lib.wave, dlsf; strict=false)
        @test size(out) == size(templates(lib))
    end
end

@testset "match_resolution to a constant σ and library_sigma_kms (cases)" begin
    lib = log_library()
    @testset "simple" begin
        @test library_sigma_kms(lib) ≈ fwhm_to_sigma_kms(2.5, lib.wave)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError match_resolution(lib, 0.0)
        @test_throws ArgumentError match_resolution(lib, -10.0)
        @test_throws ArgumentError library_sigma_kms(flat_library())
    end
    @testset "realistic" begin
        # broaden to the library's coarsest resolution: allowed everywhere
        σc = maximum(library_sigma_kms(lib))
        T = match_resolution(lib, σc)
        @test size(T) == (nwave(lib), ntemplates(lib))
    end
    @testset "pathological" begin
        # one km/s below the coarsest resolution is already sharper than the library somewhere
        σc = maximum(library_sigma_kms(lib)) - 1.0
        @test_throws ArgumentError match_resolution(lib, σc)
        @test size(match_resolution(lib, σc; strict=false)) == size(templates(lib))
    end
end

@testset "minimum_sigma_templ and match_resolution_auto (cases)" begin
    s = synthetic_setup()
    vs, nt, nd = s.vs, length(s.twave), length(s.dwave)
    na = size(s.T, 2)
    library(T, fwhm) = SSPLibrary(s.twave, reshape(T, nt, na, 1), collect(range(8.0, 10.0; length=na)),
                                  [0.0], ones(na, 1); lsf_fwhm=fwhm)
    broadened(σ_kms) = Specje.apply_banded(Specje.matching_kernel(σ_kms ./ vs), s.T)
    w = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0]
    sharp = StellarComponent(s.grid, s.T)
    fitted_σ(star, data) = fit_kinematics(SpectralModel(s.grid, star), data, [100.0, 150.0]).θ[2]
    lib60 = library(broadened(fill(60.0, nt)), sigma_kms_to_fwhm(60.0, s.twave))   # 60 km/s everywhere

    @testset "simple" begin
        @test minimum_sigma_templ(lib60, s.dwave, GaussianLSF(fill(30.0, nd))) ≈ sqrt(60.0^2 - 30.0^2)
        @test minimum_sigma_templ(lib60, s.dwave, GaussianLSF(fill(80.0, nd))) == 0.0
        @test minimum_sigma_templ(lib60, s.dwave) ≈ 60.0          # no LSF: instrument applied later
    end
    @testset "invalid input" begin
        lsf = GaussianLSF(fill(30.0, nd))
        @test_throws ArgumentError match_resolution_auto(lib60, s.dwave, lsf; σ_templ=40.0)
        @test_throws DimensionMismatch match_resolution_auto(lib60, s.dwave, GaussianLSF(fill(30.0, 5)))
        @test_throws ArgumentError minimum_sigma_templ(lib60, [9000.0, 9001.0])     # no overlap
        @test_throws ArgumentError minimum_sigma_templ(flat_library(), s.dwave)     # no lsf_fwhm
        @test_throws ArgumentError StellarComponent(s.grid, s.T; σ_templ=50.0, σ_min=40.0)
    end
    @testset "realistic" begin
        # library (60 km/s) broader than the data (30 km/s): the fit is exact; the
        # non-strict workaround is biased to √(σ² + σ_inst² − σ_lib²)
        lsf = GaussianLSF(fill(30.0, nd))
        clean = Specje.columns(sharp, s.grid, [120.0, hypot(90.0, 30.0)]) * w
        data = Spectrum(s.dwave, clean, fill(1e-3 * mean(clean), nd))
        m = match_resolution_auto(lib60, s.dwave, lsf)
        star = StellarComponent(s.grid, m.templates; σ_templ=m.σ_templ)
        @test star.σ_min == m.σ_templ                            # σ_min defaults to σ_templ
        @test fitted_σ(star, data) ≈ 90.0 rtol = 1e-3
        naive = StellarComponent(s.grid, match_resolution(lib60, s.dwave, lsf; strict=false))
        @test fitted_σ(naive, data) ≈ sqrt(90.0^2 + 30.0^2 - 60.0^2) rtol = 1e-3

        # a 2.5 Å library against an instrument going from 1.2 to 3.0 Å: the deficit
        # varies along the spectrum and its maximum sets σ_c
        σlib = fwhm_to_sigma_kms(2.5, s.twave)
        libB = library(broadened(σlib), fill(2.5, nt))
        σinst = fwhm_to_sigma_kms(collect(range(1.2, 3.0; length=nd)), s.dwave)
        galaxy = Specje.columns(sharp, s.grid, [120.0, 70.0]) * w
        observed = Specje.apply_banded(Specje.matching_kernel(σinst ./ vs), galaxy)
        dataB = Spectrum(s.dwave, observed, fill(1e-3 * mean(observed), nd))
        mB = match_resolution_auto(libB, s.dwave, GaussianLSF(σinst))
        deficit = sqrt.(max.(Specje.sigma_on_grid(σlib, s.twave, s.dwave) .^ 2 .- σinst .^ 2, 0))
        @test mB.σ_templ ≈ maximum(deficit) rtol = 1e-3
        @test fitted_σ(StellarComponent(s.grid, mB.templates; σ_templ=mB.σ_templ), dataB) ≈ 70.0 rtol = 0.01
    end
    @testset "pathological" begin
        lsf = GaussianLSF(fill(80.0, nd))
        # a library sharper than the data everywhere: σ_c = 0, identical to match_resolution
        m = match_resolution_auto(lib60, s.dwave, lsf)
        @test m.σ_templ == 0.0
        @test m.templates ≈ match_resolution(lib60, s.dwave, lsf)
        # a larger σ_templ than needed is allowed and still exact
        lsf30 = GaussianLSF(fill(30.0, nd))
        clean = Specje.columns(sharp, s.grid, [120.0, hypot(90.0, 30.0)]) * w
        data = Spectrum(s.dwave, clean, fill(1e-3 * mean(clean), nd))
        big = match_resolution_auto(lib60, s.dwave, lsf30; σ_templ=70.0)
        @test fitted_σ(StellarComponent(s.grid, big.templates; σ_templ=70.0), data) ≈ 90.0 rtol = 1e-3
        # without an LSF it is the constant-σ matching used with a resolution matrix
        r = match_resolution_auto(lib60, s.dwave)
        @test r.σ_templ ≈ 60.0
        @test r.templates ≈ match_resolution(lib60, 60.0)
        # a galaxy below σ_c cannot be fitted: the start value is refused
        star = StellarComponent(s.grid, m.templates; σ_templ=51.0)
        @test_throws ArgumentError fit_kinematics(SpectralModel(s.grid, star), data, [100.0, 40.0])
    end
end

@testset "sigma_diff_kms round-off (cases)" begin
    @testset "pathological" begin
        # equal resolutions that differ by round-off are not "broader than the data"
        σ = fwhm_to_sigma_kms(sigma_kms_to_fwhm(fill(60.0, 100), collect(4000.0:10.0:4990.0)),
                              collect(4000.0:10.0:4990.0))
        @test sigma_diff_kms(fill(60.0, 100), σ) ≈ zeros(100) atol = 1e-3
        # a genuine 0.1 % shortfall still is
        @test_throws ArgumentError sigma_diff_kms(60.0, 60.06)
    end
end

@testset "resolution_deficit_kms (cases)" begin
    lib = log_library()                          # 2.5 Å, σ_lib falls with λ
    nw = nwave(lib)
    σ_lib = library_sigma_kms(lib)
    @testset "simple" begin
        d = resolution_deficit_kms(lib, lib.wave, GaussianLSF(fill(0.0, nw)))
        @test d.wave == lib.wave && d.deficit ≈ σ_lib
        @test maximum(d.deficit) == minimum_sigma_templ(lib, lib.wave, GaussianLSF(fill(0.0, nw)))
    end
    @testset "invalid input" begin
        @test_throws ArgumentError resolution_deficit_kms(lib, [1.0, 2.0])
        @test_throws DimensionMismatch resolution_deficit_kms(lib, lib.wave, GaussianLSF(ones(3)))
    end
    @testset "realistic" begin
        # data covering only part of the library: only that part counts
        dwave = lib.wave[200:400]
        d = resolution_deficit_kms(lib, dwave, GaussianLSF(fill(60.0, 201)))
        @test d.wave == dwave
        @test d.deficit ≈ sqrt.(max.(σ_lib[200:400] .^ 2 .- 60.0^2, 0))
    end
    @testset "pathological" begin
        # one spuriously sharp LSF value (e.g. a truncated resolution-matrix row) shows
        # up as a spike that sets σ_c on its own
        σ_data = fill(100.0, nw); σ_data[1] = 1.0
        d = resolution_deficit_kms(lib, lib.wave, GaussianLSF(σ_data))
        @test argmax(d.deficit) == 1
        @test count(>(0), d.deficit) == 1
        @test minimum_sigma_templ(lib, lib.wave, GaussianLSF(σ_data)) ≈ sqrt(σ_lib[1]^2 - 1.0)
    end
end
