# Multigrid: Setup & Solvers

This section provides detailed instructions and function references for setting up the multigrid solver manually. It covers the core assembly of the system matrices, the initialization of the multigrid hierarchy, and the underlying smoothers and grid transfer operators for diffusion-reaction equations on metric graphs.

The multigrid method for diffusion-reaction equations has been developed in the context of my master's thesis, see (Brockmann, M., *Solving Elliptic PDEs on Metric Graphs: Finite Element Discretization, Multigrid Method and PCG Solver*, 2023).

## System Assembly

The following functions are used to assemble the fundamental block matrices required to represent the metric graph topology and the differential operators. The matrix is created using the incidence matrix of the extended graph as described in (Arioli, M., Benzi, M., **A Finite Element Method for Quantuum Graphs**, 2018). 

```@docs
createH
E_bar(int_vert::Int, M)
E_bar(int_vert::AbstractVector{Int}; index_type::Type{<:Integer}=Int)
E_hat
```

## Hierarchy Setup & Solver

These functions are utilized to initialize the required multigrid hierarchy and to execute the multigrid cycles (V-cycle or W-cycle) to solve the system of equations.

```@docs
MGLevel
setup_MG_hierarchy
Multigrid_Graph
Multigrid_Graph_solve!
```

## Intergrid Operators

The following functions handle the restriction operations, transferring residuals and errors from fine grids to coarser grids efficiently.

```@docs
Rest_int_nodes
Rest_HEV_setup
Rest_HEV
```

## Smoothers
Standard smoothers applied during the pre- and post-smoothing steps within a multigrid cycle to dampen high-frequency error components.
```@docs
smoother_Jac
smoother_Jac!
```