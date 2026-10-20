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
        @test profile_variance(out[:, 1]) ≈ 16 + σdiff_pix2 + 1 / 12 rtol = 1e-4
        @test sum(out) ≈ sum(line) rtol = 1e-10        # flux preserved
    end

    @testset "banded data LSF" begin
        R = banded_gaussian(fill(4.0, n))                # σ = 200 km/s
        σeff = inst_sigma_kms(R, vs)[n÷2]
        # strict=false: truncated rows at the spectrum edges look slightly sharper
        out = match_resolution(reshape(line, :, 1), twave, tfwhm, twave, R, vs; strict=false)
        @test profile_variance(out[:, 1]) ≈ 16 + (σeff^2 - σt^2) / vs^2 + 1 / 12 rtol = 1e-4
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
        @test profile_variance(depth[250:350]) ≈ 1 + (σ_target / 50)^2 + 1 / 12 rtol = 1e-3
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
