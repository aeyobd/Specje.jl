# Build:  julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#         julia --project=docs docs/make.jl
using Documenter
using Specje

makedocs(
    sitename = "Specje.jl",
    modules = [Specje],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true", edit_link = nothing),
    checkdocs = :none,
    remotes = nothing,           # not a git repository: no source links
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Concepts and conventions" => "concepts.md",
        "Guide" => [
            "Spectra and templates" => "guide/data.md",
            "Instrumental resolution" => "guide/resolution.md",
            "Building a model" => "guide/model.md",
            "χ² fitting" => "guide/fitting.md",
            "Photometry" => "guide/photometry.md",
            "Bayesian inference" => "guide/bayes.md",
        ],
        "Troubleshooting" => "troubleshooting.md",
        "API reference" => [
            "Spectra, grids and rebinning" => "api/spectra_grids_and_rebinning.md",
            "SSP libraries" => "api/ssp_libraries.md",
            "Instrumental resolution" => "api/instrumental_resolution.md",
            "Kinematics" => "api/kinematics.md",
            "Model components" => "api/model_components.md",
            "Fitting" => "api/fitting.md",
            "Photometry" => "api/photometry.md",
            "Bayesian inference" => "api/bayesian_inference.md",
        ],
    ],
)
