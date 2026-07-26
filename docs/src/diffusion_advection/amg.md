# Algebraic Multigrid (AMG)

This section details the Algebraic Multigrid (AMG) components of the package. The AMG framework provides robust coarsening strategies and solvers tailored for the solving step of the (geometric) multigrid method. The aggregation strategy of the AMG method is based on the DRA-Clustering method and a GNN clustering method. The DRA Clustering method is based on (Napov, A and Notay, Y, *An Efficient Multigrid Method for Graph Laplacian Systems II: Robust Aggregation*, SIAM Journal on Scientific Computing, 2017) and the GNN-Clustering on (Moore et. al., *Graph Neural Networks and Applied Linear Algebra*, SIAM review, 2025).

This section of code was kindly provided by Lukas Schmitz from his master's thesis: Schmitz, L. (2026). *Multigrid Methoden für Graph-Laplace Matrizen*. Master's thesis, University of Cologne. His code was adapted for the inclusion in this package. 

## AMG Hierarchy and Solvers

The core structures and iterative solvers for the algebraic multigrid approach.

```@docs
AMGLevel
build_amg_hierarchy
amg_solve
fcg_solve
mg_preconditioner!
_solve_coarsest!
```

## Coarsening Algorithms

Implementation of the Directed Resolution Algorithm (DRA) and its variants, including Quality Control (QC) and Cholesky Elimination (CE).

```@docs
DRA_QC
DRA_QC_CE
DRA_QC_CE_detailed
DRA_mix
DRA_mix_alt
```

## Utilities & Internal Methods

Helper functions and internal routines used for graph coarsening, weight computation, and singular Laplacian handling.

```@docs
prolongation
_dra_qc_ce_internal
_compute_gamma
_dra_tentative
_ext_weight
_int_weight
_build_AG_XG
_criterion_16
_bad_vertices_removal
_cholesky_quality_test
_subgroup_extraction
_qc_extract
_solve_singular_laplacian
_is_singular_laplacian
_safe_lu
```