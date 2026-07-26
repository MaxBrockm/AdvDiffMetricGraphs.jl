# Streamline Upwind Petrov-Galerkin (SUPG)

This section details the Streamline Upwind Petrov-Galerkin (SUPG) stabilization method. When solving advection-dominated problems on uniform grids, standard Galerkin methods often produce spurious oscillations. SUPG introduces consistent artificial diffusion along the streamlines (edges) to stabilize the numerical solution.

## Stabilization Parameters

Functions to compute the optimal SUPG stabilization parameter $\tau$ dynamically based on the local element size and Péclet number.

```@docs
tau_supg
coth_safe
```

## System Assembly
The system matrix for the SUPG method can be created in two ways: Via local stiffness and load vector assembly routines incorporating the SUPG test functions; or via global assembly using the incidence matrix of the extended graph. We include the stabilization term for non-conforming SUPG stabilization at vertices. Based on comparative results, standard SUPG is recommended above non-conforming SUPG.

```@docs
local_stiff
local_load
createH_AdvectionDiffusion
apply_dirichlet_blocks!
apply_vertex_stabilization_new!
```

## Examples & Validation
The following cases are designed to show the effects of the SUPG method.
```@docs
exact_solution_star_MMS
testcase_star_MMS_smooth
```

Functions demonstrating the SUPG method on star graphs, comparing it to the standard Galerkin approach, and computing the respective convergence rates in the energy norm ($||\cdot||_D$).
```@docs
plot_convergence_rate_SUPG
plot_convergence_rate_SUPG_comparison
```