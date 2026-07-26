# Misc & Helper Functions

General helper functions, plotting utilities, and basic system setup routines.
The following includes Helper functions and Miscellaneous. The functions support the core multigrid and collocation solvers.

## General Utilities & Helpers

```@docs
righthandside
normal_sign
run_MG_from_case
check_vertex_flux_condition_MG
build_mesh
estimate_dofs
choose_sparse_index_type
evalBSpline
```

## Plotting & Error Evaluation
```@docs
plot_graph_3d
plot_case_3d
plot_shishkin_3d
plot_case_edge_difference
plot_case_num_vs_exact
plot_edges_2d_simple
calculate_case_l2_error
calculate_case_D_error
```