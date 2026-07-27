# Advection-Diffusion Metric Graph Package

Welcome to the documentation for the **AdvDiffMetricGraphs** Julia package. This package provides a comprehensive suite of numerical solvers and discretization methods for partial differential equations (PDEs) on metric graphs, with a strong emphasis on robust multigrid solvers for advection-dominated problems.

I have developed the AdvDiffMetricGraphs.jl package in connection with my Ph.D thesis at the University of Cologne (*Multigrid Methods for Diffusion-Advection Equations on Metric Graphs: Creating Archaeological Networks and Modelling Migration Dynamics*, 2026).

## Package Overview

The numerical framework is designed to handle complex graph topologies and is divided into two main domains:

*   **Diffusion-Reaction:** Standard Continuous Galerkin formulations, Preconditioned Conjugate Gradient (PCG) solvers, and geometric multigrid setups.
*   **Diffusion-Advection:** Specialized methods for advection-dominated flows. This includes Streamline-Upwind Petrov-Galerkin (SUPG) stabilization, Discontinuous Galerkin (dG) methods, boundary layer resolving meshes (Shishkin and Bakhvalov), and macroscopically directed multigrid smoothers tailored to graph topologies. 
*   **Topological Edge Cases:** Automated identification and stabilization of "Full Inflow" vertices.

## Table of Contents

```@contents
Depth = 2
```

Copyright (c) 2026 Max Brockmann (University of Cologne)