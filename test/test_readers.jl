using HDF5

@testset "readers" begin
    lib = normalize_window(toy_library(); window=(4400.0, 4600.0))

    @testset "round trip" begin
        path = tempname() * ".h5"
        write_ssp_hdf5(path, lib)
        back = read_fsps_hdf5(path)
        for name in fieldnames(SSPLibrary)
            @test getfield(back, name) == getfield(lib, name)
        end
        rm(path)
    end

    @testset "missing norm defaults to one" begin
        path = tempname() * ".h5"
        h5open(path, "w") do f
            for name in (:wave, :flux, :log_age, :logzsol, :mass_frac)
                write(f, String(name), getfield(lib, name))
            end
        end
        back = read_fsps_hdf5(path)
        @test all(==(1.0), back.norm)
        @test back.flux == lib.flux
        rm(path)
    end

    @testset "bad file" begin
        path = tempname() * ".h5"
        h5open(path, "w") do f
            write(f, "wave", [1.0, 2.0])
        end
        @test_throws Exception read_fsps_hdf5(path)
        rm(path)
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "read_fsps_hdf5 and write_ssp_hdf5 (cases)" begin
    @testset "simple" begin
        lib = toy_library()
        path = tempname() * ".h5"
        @test write_ssp_hdf5(path, lib) == path
        @test read_fsps_hdf5(path).flux == lib.flux
        rm(path)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError read_fsps_hdf5(tempname() * ".h5")   # no such file
        path = tempname() * ".h5"
        lib = toy_library()
        h5open(path, "w") do f                               # inconsistent shapes
            write(f, "wave", lib.wave)
            write(f, "flux", lib.flux[:, :, 1:2])
            write(f, "log_age", lib.log_age)
            write(f, "logzsol", lib.logzsol)
            write(f, "mass_frac", lib.mass_frac)
        end
        @test_throws ArgumentError read_fsps_hdf5(path)
        rm(path)
    end
    @testset "realistic" begin
        # cache a log-rebinned, normalised library and read it back unchanged
        lib = normalize_window(rebin_log(toy_library(; nw=400); velscale=60.0);
                               window=(4400.0, 4600.0))
        path = tempname() * ".h5"
        write_ssp_hdf5(path, lib)
        back = read_fsps_hdf5(path)
        @test back.wave == lib.wave && back.flux == lib.flux && back.norm == lib.norm
        @test is_loguniform(back.wave)
        rm(path)
    end
    @testset "pathological" begin
        # overwriting an existing file replaces it completely (no stale lsf_fwhm)
        path = tempname() * ".h5"
        write_ssp_hdf5(path, toy_library())
        write_ssp_hdf5(path, flat_library())
        @test read_fsps_hdf5(path).lsf_fwhm === nothing
        rm(path)
    end
end
