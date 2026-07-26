# Multigrid for Advection-Diffusion

Standard multigrid smoothers often fail in advection-dominated regimes due to the strong directional coupling of the flow. To restore optimal multigrid efficiency, this package provides specialized smoothers and macroscopic topological sorting algorithms tailored for metric graphs.

## Core Hierarchy and Solvers
The following structures and functions handle the setup and execution of the multigrid cycles specifically tailored for directed flow graphs.

```@docs
AdvectionMGLevel
setup_advection_hierarchy
solve_advection_MG!
Multigrid_Graph_adv
```

## Matrix-Dependent Transfer Operators
For advection-dominated problems, standard geometric interpolation is often insufficient. These matrix-dependent prolongation and restriction operators are constructed directly from the system matrices to preserve upwind characteristics across grid levels.

```@docs
Prol_int_nodes_MatrixDep
Prol_HEV_MatrixDep
Rest_int_nodes_MatrixDep
Rest_HVE_MatrixDep
```

## Macroscopic Topological Sorting

For advection-dominated flows on metric graphs, information propagates predominantly along the flow direction. We exploit this by performing a topological sort of the graph vertices based on the advective field. This macroscopic ordering is then used to construct level-dependent permutation vectors for the smoothing steps.

```@docs
get_macroscopic_sort
build_level_permutation
```

## Specialized Smoothers
Using the topological permutations, we can apply Directional Gauss-Seidel (DGS) smoothing, which sweeps through the graph following the physical flow of information. Alternatively, robust Krylov-based smoothers (GMRES) or Incomplete LU (ILU) factorizations are available in case of cyclic graphs (for which a macroscopic sort fails).

```@docs
smoother_DGS
smoother_GMRES
smoother_ILU
```

## Examples & Parameter Sweeps
Pre-configured example routines demonstrating the directed multigrid method applied to advection-diffusion problems. These functions include single runs on simple topologies as well as extensive parameter sweeps on large complex networks (e.g., Barabási-Albert graphs) to analyze convergence rates and solver performance.

```@docs
MG_example_AdvectionDiffusion
MG_advection_varying_parameters
MG_advection_Barabasi_extended
run_MG_stargraph
run_MG_DiffAdv_example
run_barabasi_example
run_barabasi_example_with_cycle
run_MG_cycle
```