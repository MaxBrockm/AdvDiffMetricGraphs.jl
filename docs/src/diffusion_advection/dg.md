# Discontinuous Galerkin

This section covers the Discontinuous Galerkin (dG) methods for the advection-diffusion equation on metric graphs. 

## System Assembly

The following function assembles the global block matrices using the Discontinuous Galerkin approach. It incorporates Symmetric Interior Penalty Galerkin (SIPG) terms for diffusion and upwind fluxes for the advective transport across element interfaces and graph vertices.

```@docs
createH_DG_AdvectionDiffusion
```

## Error Evaluation
Tools to compute numerical errors specifically in the dG energy norm. These functions account for both the bulk element contributions (broken $H^1$ semi-norm) and the inter-element/vertex jump penalties.

```@docs
calculate_case_dG_energy_error
```

## Examples & Validation

Pre-configured examples to run the dG solvers on specific topologies (e.g., star graphs), compare their convergence rates against standard continuous Galerkin methods, and visualize the discontinuous solutions.

```@docs
run_star_KumarLeugering_dG
run_star_KumarLeugering_pure_dG
plot_convergence_rate_dG_comparison
plot_convergence_rate_pure_dG_comparison
```