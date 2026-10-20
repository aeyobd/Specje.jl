@testset "SSPLibrary lsf_fwhm" begin
    lib = toy_library()

    @testset "carried through selection and normalisation" begin
        @test lib.lsf_fwhm == fill(2.5, 50)
        @test subset_library(lib; log_age=(8.5, 10.0)).lsf_fwhm == lib.lsf_fwhm
        @test length(crop_wave(lib, 4300.0, 4700.0).lsf_fwhm) ==
              length(crop_wave(lib, 4300.0, 4700.0).wave)
        @test normalize_window(lib; window=(4400.0, 4600.0)).lsf_fwhm == lib.lsf_fwhm
    end

    @testset "validation" begin
        @test_throws ArgumentError SSPLibrary(lib.wave, lib.flux, lib.log_age, lib.logzsol,
                                              lib.mass_frac; lsf_fwhm=ones(3))
        @test flat_library().lsf_fwhm === nothing
    end

    @testset "rebinning" begin
        out = rebin_log(lib; velscale=200.0)
        @test all(≈(2.5), out.lsf_fwhm)
        @test length(out.lsf_fwhm) == nwave(out)
    end

    @testset "HDF5" begin
        path = tempname() * ".h5"
        write_ssp_hdf5(path, lib)
        @test read_fsps_hdf5(path).lsf_fwhm == lib.lsf_fwhm
        write_ssp_hdf5(path, flat_library())
        @test read_fsps_hdf5(path).lsf_fwhm === nothing
        rm(path)
    end
end
