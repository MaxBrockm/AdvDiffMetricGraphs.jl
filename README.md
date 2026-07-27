Welcome to AdvDiffMetricGraphs.jl, a Julia package designed for solving diffusion-reaction equations and advection-diffusion equations on metric graphs. This package provides a comprehensive suite of numerical solvers and discretization methods for partial differential equations (PDEs) on metric graphs, with a strong emphasis on robust geometric multigrid solvers for advection-dominated problems.
A comprehensive documentation of this package is available at https://maxbrockm.github.io/AdvDiffMetricGraphs.jl/dev/

I developed this package in connection with my Ph.D. thesis at the University of Cologne: Multigrid Methods for Diffusion-Advection Equations on Metric Graphs: Creating Archaeological Networks and Modelling Migration Dynamics (2026).

## Features 
* Robust Multigrid Solvers: Custom geometric multigrid hierarchy and solvers specifically adapted for the topological constraints of metric graphs.
* Advection-Dominated Regimes: Includes advanced stabilization techniques to prevent oscillations:
  - Streamline-Upwind Petrov-Galerkin (SUPG) methods.
  - Discontinuous Galerkin (dG) methods with upwind fluxes.
* Layer-Adapted Meshes: Built-in generators for boundary layer resolution using Shishkin and Bakhvalov meshes.
* Topological Auto-Fix: Automatic detection and topological correction (auto-fix) for full-inflow vertices.

## Installation
You can install the package using Julia's package manager. From the Julia REPL, type ] to enter the Pkg prompt and run:
```julia
pkg> add AdvDiffMetricGraphs
```

## Citation
If you use AdvDiffMetricGraphs.jl in your research, please cite the associated thesis:

Brockmann, M. (2026). Multigrid Methods for Diffusion-Advection Equations on Metric Graphs: Creating Archaeological Networks and Modelling Migration Dynamics. Ph.D. thesis, University of Cologne.
