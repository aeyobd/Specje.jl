#!/usr/bin/env python
"""Export an FSPS SSP grid to HDF5 for Specje.jl.

    python export_fsps.py --out fsps_ssp.h5 [--imf-type 1] [--wmin 3525 --wmax 7500]

Layout (h5py/C order; HDF5.jl sees it with dimensions reversed):
    wave       (nwave,)               Å, vacuum
    flux       (nmet, nage, nwave)    L_sun/Å per M_sun formed
    log_age    (nage,)                log10(yr)
    logzsol    (nmet,)                log10(Z/Z_sun)
    mass_frac  (nmet, nage)           surviving stellar mass per M_sun formed

The default wavelength window is the MILES range; outside it FSPS splices in a
much lower-resolution library, so a constant `lsf_fwhm` would be wrong there.
2.5 Å is the usual MILES value (Falcon-Barroso et al. 2011); check your build.
"""
import argparse

import fsps
import h5py
import numpy as np

C_KMS = 299792.458


def make_population(imf_type):
    # zcontinuous=0: use the library's native metallicity grid. No dust, no nebular emission.
    return fsps.StellarPopulation(zcontinuous=0, sfh=0, imf_type=imf_type,
                                  add_neb_emission=False, dust_type=0)


def ssp_for_metallicity(sp, imin):
    """All-age SSP spectra and surviving mass for the (1-indexed) metallicity `imin`."""
    _, spec = sp.get_spectrum(zmet=imin, tage=0.0, peraa=True)
    return np.asarray(spec, float), np.asarray(sp.stellar_mass, float)


def build_grid(sp, wmin, wmax):
    wave = np.asarray(sp.wavelengths, float)
    keep = (wave >= wmin) & (wave <= wmax)
    log_age = np.asarray(sp.log_age, float)
    zleg = np.asarray(sp.zlegend, float)
    flux = np.empty((len(zleg), len(log_age), keep.sum()))
    mass_frac = np.empty((len(zleg), len(log_age)))
    for i in range(len(zleg)):
        spec, mass = ssp_for_metallicity(sp, i + 1)
        assert spec.shape == (len(log_age), len(wave)), spec.shape
        assert mass.shape == (len(log_age),), mass.shape
        flux[i], mass_frac[i] = spec[:, keep], mass
    logzsol = np.log10(zleg / sp.solar_metallicity)

    res = np.asarray(sp.resolutions, float)[keep]
    sigma_A = np.abs(res) / C_KMS * wave[keep]
    lsf_fwhm = 2*np.sqrt(2*np.log(2)) * sigma_A

    return wave[keep], flux, log_age, logzsol, mass_frac, lsf_fwhm


def write_hdf5(path, sp, grid):
    names = ("wave", "flux", "log_age", "logzsol", "mass_frac", "lsf_fwhm")
    with h5py.File(path, "w") as f:
        for name, data in zip(names, grid):
            f.create_dataset(name, data=data)
        f.attrs["isoc_library"] = str(sp.isoc_library)
        f.attrs["spec_library"] = str(sp.spec_library)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imf-type", type=int, default=2, help="0 Salpeter, 1 Chabrier, 2 Kroupa")
    ap.add_argument("--wmin", type=float, default=2000.0)
    ap.add_argument("--wmax", type=float, default=12_000.0)
    args = ap.parse_args()
    sp = make_population(args.imf_type)
    write_hdf5(args.out, sp, build_grid(sp, args.wmin, args.wmax))


if __name__ == "__main__":
    main()
