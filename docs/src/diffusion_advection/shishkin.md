# Shishkin & Bakhvalov Meshes

This section covers boundary layer-adapted meshes, specifically Shishkin and Bakhvalov grids. These meshes are essential for correctly resolving exponential boundary layers that appear in singularly perturbed (advection-dominated) diffusion-advection equations on metric graphs, without relying on artificial diffusion. The Shishkin mesh is adapted for the finite element method from (Kumar, V., Leugering, G., *Convection dominated singularly perturbed problems on a metric graph*). 

## Mesh Generation

The following functions generate 1D boundary layer meshes on individual edges. They automatically fall back to uniform grids if the local Péclet number is sufficiently small (diffusion-dominated regime).

```@docs
shishkin_grid
bakhvalov_nodes
```

## System Assembly

To assemble the system matrices on these non-uniform meshes without applying additional stabilization techniques (like SUPG), the standard Galerkin approach is used locally. The functions are designed to work with non-uniform meshes.

```@docs
local_stiff_std
local_load_std
assemble_system_blocks_shishkin
```

## Examples & Validation
The following includes pre-configured examples to solve via direct solvers on boundary layer-adapted meshes, alongside functions to visualize their solutions and convergence rates.
```@docs
run_star_shishkin
run_star_bakhvalov
run_star_shishkin_block_backslash
plot_convergence_rate_shishkin
```
The examples are based on the following cases.
```@docs
exact_solution_star_Kumar_Leugering
testcase_star_exact_Kumar_Leugering
run_star_KumarLeugering
```

