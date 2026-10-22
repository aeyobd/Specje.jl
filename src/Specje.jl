module Specje

using Statistics
using LinearAlgebra
using SparseArrays: sparse, dropzeros!, SparseMatrixCSC
using HDF5
using FFTW
using SpecialFunctions: erf
import Optim
import LogDensityProblems
import ForwardDiff
import DiffResults
using Distributions: Normal, LogNormal, truncated, logpdf, UnivariateDistribution

include("spectrum.jl")
include("grid.jl")
include("rebin.jl")
include("templates/library.jl")
include("templates/resample.jl")
include("templates/readers.jl")
include("kinematics/lsf.jl")
include("kinematics/match.jl")
include("kinematics/losvd.jl")
include("kinematics/convolve.jl")
include("photometry/filters.jl")
include("photometry/photometry.jl")
include("photometry/bands.jl")
include("model/nnls.jl")
include("model/components.jl")
include("model/gas.jl")
include("model/model.jl")
include("model/fit.jl")
include("model/joint.jl")
include("bayes/stats.jl")
include("bayes/marginal.jl")
include("bayes/weights.jl")
include("bayes/priors.jl")
include("bayes/posterior.jl")
include("bayes/ensemble.jl")

export C_KMS, Spectrum, npix, velscale, is_loguniform, crop
export pixel_edges, log_pixel_edges, loggrid
export rebin_flux, rebin_noise, rebin_good, rebin_log, model_grid, pixel_projection
export SSPLibrary, nwave, nage, nmet, ntemplates, age_gyr
export subset_library, crop_wave, normalize_window
export templates, template_params, reshape_weights, formed_mass, stellar_mass
export read_fsps_hdf5, write_ssp_hdf5
export LSF, GaussianLSF, BandedLSF, fwhm_to_sigma_kms, sigma_kms_to_fwhm
export inst_sigma_kms, apply_banded, dense_matrix, normalize_rows
export banded_gaussian, banded_from_dia, to_banded
export sigma_diff_kms, match_resolution, library_sigma_kms
export match_resolution_auto, minimum_sigma_templ, resolution_deficit_kms
export gaussian_losvd_ft
export KinematicsGrid, kinematics_grid, shift_pixels
export transform_templates, convolve_templates, convolve_blocks
export nnls
export Component, StellarComponent, PolynomialComponent, legendre_basis
export Multiplet, GasGroup, GasComponent, default_gas_groups
export line_names, group_names
export SpectralModel, nparams, ncolumns, param_ranges, column_ranges
export design_matrix, split_weights
export FitData, FitResult, solve_weights, profile_chi2, fit_kinematics
export model_flux, reduced_chi2
export Filter, transmission, pivot_wavelength, to_rest, gaussian_filter, tophat_filter
export ab_to_flambda, flambda_to_ab, jy_to_flambda, mag_err_to_flux_err
export Photometry, nphot, band_names
export bandpass_weights, BandTemplates, band_templates
export photometry_matrix, JointModel, spectral_scale, model_spectrum, model_photometry
export BlockStats, block_stats_all, combine, marginal_loglike
export WeightPosterior, conditional_weights, sample_weights
export Priors, default_priors, default_tau
export BayesModel, log_posterior, log_posterior_and_gradient, to_physical, initial_free, map_estimate, weight_posterior
export EnsembleChain, sample_ensemble, acceptance_fraction, flatten_chain
export autocorr_function, autocorr_time_1d, autocorr_time

end
