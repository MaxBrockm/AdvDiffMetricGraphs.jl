using Documenter
using AdvDiffMetricGraphs 

# Build the documentation
makedocs(
    sitename = "AdvDiffMetricGraphs.jl",
    format = Documenter.HTML(
        # Use pretty URLs only in the CI environment to avoid broken local links
        prettyurls = get(ENV, "CI", "false") == "true"
    ),
    modules = [AdvDiffMetricGraphs],
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Diffusion-Reaction" => [
            "Diffusion-Reaction Equations on Metric Graphs" => "diffusion_reaction/diffusion_reaction.md",
            "Multigrid Examples" => "diffusion_reaction/multigrid_examples.md",
            "Multigrid: Do it yourself Setup" => "diffusion_reaction/multigrid_diy.md",
            "PCG Methode" => "diffusion_reaction/pcg_method.md",
        ],
        "Diffusion Advection" => [
            "Advection-Diffusion Equations on Metric Graphs" => "diffusion_advection/diffusion_advection_landing.md",
            "Case Setup" => "diffusion_advection/case_setup.md",
            "Shishkin & Bakhvalov Meshes" => "diffusion_advection/shishkin.md",
            "SUPG" => "diffusion_advection/supg.md",
            "Discontinuous Galerkin" => "diffusion_advection/dg.md",
            "Multigrid & Directed Smoothing" => "diffusion_advection/mg_directed.md",
            "AMG" => "diffusion_advection/amg.md",
            "Collocation" => "diffusion_advection/collocation.md",
        ],
        "Misc" => "misc.md",
        "Index" => "end.md"
    ]
)

# Deploy the documentation to the gh-pages branch of the repository
deploydocs(
    repo = "github.com/maxbrockm/AdvDiffMetricGraphs.jl.git",
    devbranch = "main"
)