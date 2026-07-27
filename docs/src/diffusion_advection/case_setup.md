# Case Setup

This section covers the setup of advection-diffusion test cases on various metric graph topologies. It includes pre-defined graph structures, boundary condition configurations, and analytical exact solutions tailored for rigorous numerical validation.

## Creating Custom Test Cases

To evaluate custom graph topologies or specific advection-diffusion configurations, you can easily set up your own test case environment. The solvers in this package (such as run_MG_from_case) expect the problem definition to be structured as a standard `NamedTuple`.

### Required Fields
A valid custom case requires the following fields to successfully assemble the system matrices and execute the solvers:
- name: A String identifier for your case.
- nv: An Int representing the total number of vertices in the graph.
- edges: A Vector{Edge} defining the directed edges of the graph.
- n_e: A Dict{Edge, Int} mapping each edge to the number of discrete elements (often configured as $2^J$).
- edge_x: A Dict{Edge, Vector{Float64}} mapping each edge to a vector of its spatial grid nodes (e.g., from 0.0 to L). This allows for custom mesh grading (like Shishkin layers).
- eps_edge: A Dict{Edge, Float64} defining the diffusion coefficient $D$ for each individual edge.
- a_edge: A Dict{Edge, Float64} defining the advection coefficient $A$ for each individual edge.
- f_edge: A Dict{Edge, Function} mapping each edge to its continuous right-hand side forcing function $f(x)$.
- dirichlet: A Dict{Int, Float64} specifying the Dirichlet boundary conditions, mapping vertex indices to their fixed boundary values.

### Optional Fields
For advanced analysis, such as calculating numerical errors, convergence rates, or extracting specific derivatives, you can include the following optional fields:
- exakte_Loesung: A function returning the exact analytical solution evaluated at a spatial coordinate $x$. It must return a vector containing the evaluated values for all edges. Set this to nothing if the exact solution is unknown.
- exact_derivative: A function returning the first spatial derivative of the exact solution. Required for energy norm error evaluations.
- exact_k_th_derivative: A function (x, k) returning the $k$-th spatial derivative of the exact solution.
- layer_mesh: A Symbol (e.g., :uniform, :shishkin, :bakhvalov) indicating the type of boundary layer mesh applied to the graph.

The following are predefined cases based on the above layout. 

## Basic Graph Topologies

The following functions are used to initialize standard geometries, such as star graphs and Barabási-Albert networks. They return a standardized `case` tuple required by the solvers.

```@docs
testcase_stargraph
testcase_barabasi
```

## Cyclic Topologies & Directed Flow

These test cases introduce cycles into the metric graphs. They are particularly relevant for evaluating directed smoothers of the multigrid method.

```@docs
testcase_barabasi_with_cycle
testcase_cycle_advection
exact_solution_cycle_Lollipop
```

## Triangle Graph Configurations

The following functions initialize triangle graph problems with different analytical exact solutions. They generate the appropriate right-hand side vectors and exact solution references necessary for convergence rate calculations.

```@docs
testcase_triangle_exact_WOx
exact_solution_triangle_WOx
testcase_triangle_exact_any
exact_solution_triangle_any
```

