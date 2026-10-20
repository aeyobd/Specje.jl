# HDF5 I/O. HDF5.jl reverses dimension order relative to h5py, so the
# Python exporter's (nmet, nage, nwave) cube arrives here as (nwave, nage, nmet).

"""
    read_optional(f, name, default)

Read dataset `name` from the open HDF5 file `f`, or return `default` if it is absent.
"""
function read_optional(f, name::AbstractString, default)
    if haskey(f, name)
        return read(f, name)
    else
        return default
    end
end


"""
    read_fsps_hdf5(path) -> SSPLibrary

Read a library written by `export_fsps.py` (or `write_ssp_hdf5`).
Datasets: `wave`, `flux`, `log_age`, `logzsol`, `mass_frac`; optional `norm`, `lsf_fwhm`.
"""
function read_fsps_hdf5(path::AbstractString)
    if !isfile(path)
        throw(ArgumentError("no such file: $path"))
    end

    lib = h5open(path, "r") do f
        wave = read(f, "wave")
        flux = read(f, "flux")
        log_age = read(f, "log_age")
        logzsol = read(f, "logzsol")
        mass_frac = read(f, "mass_frac")
        norm = read_optional(f, "norm", ones(size(mass_frac)))
        lsf_fwhm = read_optional(f, "lsf_fwhm", nothing)
        SSPLibrary(wave, flux, log_age, logzsol, mass_frac; norm=norm, lsf_fwhm=lsf_fwhm)
    end
    return lib
end


"""
    write_ssp_hdf5(path, lib)

Write an `SSPLibrary` (e.g. a log-rebinned one, to cache it) in the layout
`read_fsps_hdf5` reads. Returns `path`.
"""
function write_ssp_hdf5(path::AbstractString, lib::SSPLibrary)
    h5open(path, "w") do f
        for name in (:wave, :flux, :log_age, :logzsol, :mass_frac, :norm)
            write(f, String(name), getfield(lib, name))
        end
        if !isnothing(lib.lsf_fwhm)
            write(f, "lsf_fwhm", lib.lsf_fwhm)
        end
    end
    return path
end
