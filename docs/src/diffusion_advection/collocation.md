# Collocation

This section covers the B-spline collocation methods for metric graphs. Functions include the setup of B-Splines of order `p`. The efficient implementation of V-Splines is done in reference to (Piegel, L., Tiller, W., **The NURBS book**, 1997).   

The collocation method is exemplary applied in examples. 
```@docs
solve_star_poly
testcase_star_poly
```

The following functions can be used to set up the system of equations.
```@docs
get_edge_coeff
eval_f
get_g
generate_knot_vector_from_breaks
greville_points
find_span
ders_basis_funs
assemble_bspline_collocation
solve_bspline_collocation
eval_edge_u_ders
eval_edge_u
get_exact_solution_ders
```

The following functions serve the illustration of convergence properties and visualization. 
```@docs
calculate_errors_collocation
calculate_errors_infty
plot_convergence_rates_trig
```