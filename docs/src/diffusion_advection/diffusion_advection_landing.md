# Advection Diffusion Equation

The codes of this section are for solving the advection diffusion equation on metric graphs. Special interest is on the advection dominant regime, where stabilization methods are required. For solvers we consider a multigrid method for advection-diffusion as well as a collocation method.

## [Case Setup](case_setup.md)
Setup of advection-diffusion test cases on various metric graph topologies, including pre-defined graph structures and analytical exact solutions.

## [Shishkin & Bakhvalov meshes](shishkin.md)
Generation of boundary layer-adapted meshes to resolve exponential boundary layers without artificial diffusion.

## [SUPG method](supg.md)
Implementation of the Streamline Upwind Petrov-Galerkin (SUPG) stabilization method for advection-dominated problems on uniform grids.

## [Discontinuous Galerkin](dg.md)
Discontinuous Galerkin formulations for the advection-diffusion equation.

## [Multigrid Method](mg_directed.md)
Multigrid solvers featuring directed smoothing techniques tailored for advection-dominated flows on graphs.

## [Collocation](collocation.md)
B-spline collocation methods relying on polynomial basis functions to satisfy the governing differential equations.

## Full Inflow vertices
On metric graphs, the advection dominant regime gives rise to full-inflow vertices with poor numerical properties. As a detection mechanism, we detect and fix full-inflow vertices using the following functions:
```@docs
check_inflow_vertices
fix_inflow_nodes!
```