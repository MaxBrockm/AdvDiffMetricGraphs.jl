"""
    testcase_stargraph(A::Real, D::Real, J::Integer; dirichlet_nodes::Vector{Int}=[1], dirichlet_values::Vector{Float64}=[0.0])

Generates the star graph test case in case format, allowing multiple Dirichlet nodes via arrays.

# Arguments
- `A::Real`: Advection coefficient.
- `D::Real`: Diffusion coefficient.
- `J::Integer`: Refinement level determining the number of elements per edge (2^J).
- `dirichlet_nodes::Vector{Int}`: Vector of node indices with Dirichlet boundary conditions (default: `[1]`).
- `dirichlet_values::Vector{Float64}`: Vector of Dirichlet values corresponding to the nodes (default: `[0.0]`).

# Output
- A named tuple containing the test case configuration parameters (name, nv, edges, n_e, edge_x, eps_edge, a_edge, f_edge, dirichlet, exakte_Loesung, use_shishkin).
"""
function testcase_stargraph(A::Real, D::Real, J::Integer;
                            dirichlet_nodes::Vector{Int}=[1],
                            dirichlet_values::Vector{Float64}=[0.0])
    
    @assert length(dirichlet_nodes) == length(dirichlet_values) "Number of Dirichlet nodes must match the number of Dirichlet values!"

    Γ = star_graph(4)
    nv_graph = nv(Γ)
    edges_ordered = collect(edges(Γ))
    L = 1.0

    n_e = Dict(e => Int(2^J) for e in edges_ordered)
    eps_edge = Dict(e => float(D) for e in edges_ordered)
    a_edge   = Dict(e => float(A) for e in edges_ordered)
    
    f_edge = Dict{Graphs.SimpleGraphs.SimpleEdge{Int}, Function}()
    for (i, e) in enumerate(edges_ordered)
        if i == 1
            f_edge[e] = x -> sin(pi * x)
        else
            f_edge[e] = x -> 1.0
        end
    end

    edge_x = Dict{Graphs.SimpleGraphs.SimpleEdge{Int}, Vector{Float64}}()
    for e in edges_ordered
        edge_x[e] = collect(range(0.0, L; length=n_e[e]+1))
    end

    dirichlet = Dict{Int, Float64}(n => float(v) for (n, v) in zip(dirichlet_nodes, dirichlet_values))

    return (name="testcase_stargraph",
            nv=nv_graph, edges=edges_ordered, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=nothing,
            use_shishkin=false)
end


"""
    testcase_barabasi(nv_target::Int, A::Real, D::Real, J::Integer; dirichlet_nodes::Vector{Int}=[1], dirichlet_values::Vector{Float64}=[0.0])

Generates the Barabási-Albert graph test case in case format with support for multiple Dirichlet nodes. No exact solution can be given. Discretization along edges is uniform. The RHS function is defined edgewise as f(x) = cos(2πx/L * (i mod 4)) for edge i with length L.

# Arguments
- `nv_target::Int`: Target number of vertices for the Barabási-Albert graph.
- `A::Real`: Advection coefficient.
- `D::Real`: Diffusion coefficient.
- `J::Integer`: Refinement level determining the number of elements per edge (2^J).
- `dirichlet_nodes::Vector{Int}`: Vector of node indices with Dirichlet boundary conditions (default: `[1]`).
- `dirichlet_values::Vector{Float64}`: Vector of Dirichlet values corresponding to the nodes (default: `[0.0]`).

# Output
- A named tuple containing the test case configuration parameters.
"""
function testcase_barabasi(nv_target::Int, A::Real, D::Real, J::Integer;
                           dirichlet_nodes::Vector{Int}=[1],
                           dirichlet_values::Vector{Float64}=[0.0])
    
    @assert length(dirichlet_nodes) == length(dirichlet_values) "Number of Dirichlet nodes must match the number of Dirichlet values!"

    Γ = barabasi_albert(nv_target, 13, complete=true, seed=1)
    nv_graph = nv(Γ)
    edges_ordered = collect(edges(Γ))
    
    L_vals = repeat([1.0, 2.0], ceil(Int, length(edges_ordered) / 2))[1:length(edges_ordered)]
    
    n_e = Dict(e => Int(2^J) for e in edges_ordered)
    eps_edge = Dict(e => float(D) for e in edges_ordered)
    a_edge   = Dict(e => float(A) for e in edges_ordered)
    
    f_edge = Dict{Graphs.SimpleGraphs.SimpleEdge{Int}, Function}()
    edge_x = Dict{Graphs.SimpleGraphs.SimpleEdge{Int}, Vector{Float64}}()
    
    for (i, e) in enumerate(edges_ordered)
        L = L_vals[i]
        f_edge[e] = x -> cos(2 * pi * x / L * (i % 4))
        edge_x[e] = collect(range(0.0, L; length=n_e[e]+1))
    end

    dirichlet = Dict{Int, Float64}(n => float(v) for (n, v) in zip(dirichlet_nodes, dirichlet_values))

    return (name="testcase_barabasi_$(nv_target)",
            nv=nv_graph, edges=edges_ordered, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=nothing,
            use_shishkin=false)
end


"""
    testcase_barabasi_with_cycle(nv_target::Int, A::Real, D::Real, J::Integer; dirichlet_nodes::Vector{Int}=[1], dirichlet_values::Vector{Float64}=[0.0])

Generates a Barabási-Albert graph test case extended with an additional cycle structure.

# Arguments
- `nv_target::Int`: Target number of vertices for the base Barabási-Albert graph.
- `A::Real`: Advection coefficient.
- `D::Real`: Diffusion coefficient.
- `J::Integer`: Refinement level determining the number of elements per edge (2^J).
- `dirichlet_nodes::Vector{Int}`: Vector of node indices with Dirichlet boundary conditions (default: `[1]`).
- `dirichlet_values::Vector{Float64}`: Vector of Dirichlet values corresponding to the nodes (default: `[0.0]`).

# Output
- A named tuple containing the test case configuration parameters.
"""
function testcase_barabasi_with_cycle(nv_target::Int, A::Real, D::Real, J::Integer;
                                      dirichlet_nodes::Vector{Int}=[1],
                                      dirichlet_values::Vector{Float64}=[0.0])
    
    @assert length(dirichlet_nodes) == length(dirichlet_values) "Number of Dirichlet nodes must match the number of Dirichlet values!"

    Γ_undir = barabasi_albert(nv_target, 13, complete=true, seed=1)
    
    nv_graph = nv_target + 4
    Γ = SimpleDiGraph(nv_graph) 
    
    for e in edges(Γ_undir)
        add_edge!(Γ, src(e), dst(e))
    end
    
    attach_node = nv_target
    node_c1 = nv_target + 1
    node_c2 = nv_target + 2
    node_c3 = nv_target + 3
    node_c4 = nv_target + 4
    
    add_edge!(Γ, attach_node, node_c1) 
    add_edge!(Γ, node_c1, node_c2)     
    add_edge!(Γ, node_c2, node_c3)     
    add_edge!(Γ, node_c3, attach_node) 
    add_edge!(Γ, node_c2, node_c4)     
    
    edges_ordered = collect(edges(Γ))
    push!(dirichlet_nodes, node_c4)
    push!(dirichlet_values, 0.0) 

    L_vals = ones(Float64, length(edges_ordered))
    
    n_e = Dict(e => Int(2^J) for e in edges_ordered)
    eps_edge = Dict(e => float(D) for e in edges_ordered)
    a_edge   = Dict(e => float(A) for e in edges_ordered)
    
    
    f_edge = Dict{Graphs.SimpleGraphs.SimpleEdge{Int}, Function}()
    edge_x = Dict{Graphs.SimpleGraphs.SimpleEdge{Int}, Vector{Float64}}()
    
    for (i, e) in enumerate(edges_ordered)
        L = L_vals[i]
        f_edge[e] = x -> cos(2 * pi * x / L * (i % 4))
        edge_x[e] = collect(range(0.0, L; length=n_e[e]+1))
    end

    dirichlet = Dict{Int, Float64}(n => float(v) for (n, v) in zip(dirichlet_nodes, dirichlet_values))

    return (name="testcase_barabasi_with_cycle_$(nv_graph)",
            nv=nv_graph, edges=edges_ordered, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=nothing,
            use_shishkin=false)
end


"""
    exact_solution_triangle_WOx(A, D, dirichlet_value; show_plots::Bool=false)

Computes the exact solution and right-hand side vector for a triangle graph problem using specific ansatz functions. Triangle is extended with one Dirichlet vertex. Problem contains full inflow vertex at vertex 3. The ansatz functions are of the form c[1] + c[2]*exp(λ*x) + g(x), where λ = -A/D and g(x) is a polynomial function. 

# Arguments
- `A`: Advection coefficient.
- `D`: Diffusion coefficient.
- `dirichlet_value`: Value for the Dirichlet boundary condition.
- `show_plots::Bool`: Flag to display plots of the solution and right-hand side (default: `false`).

# Output
- A tuple `(exact_sol, Rhs)` containing the exact solution function and the vector of right-hand side functions per edge.
"""
function exact_solution_triangle_WOx(A,D, dirichlet_value; show_plots::Bool=false)
    λ = -A / D
    exp_λ = exp(λ)
    
    g_1(x) = x^2+x+1
    g_1_p(x) = 2x+1
    g_1_pp(x) = 2
    g_2(x) = x^2+x+1
    g_2_p(x) = 2x+1
    g_2_pp(x) = 2
    g_3(x) = x^2+x+1
    g_3_p(x) = 2x+1
    g_3_pp(x) = 2
    g_4(x) = x^2+x+1
    g_4_p(x) = 2x+1
    g_4_pp(x) = 2

    M = zeros(8, 8)
    b = zeros(8)
    
    # 1. Continuity vertex 1
    M[1, [1, 2, 3, 4]] = [1, 1, -1, -1]
    b[1] = g_2(0) - g_1(0)
    
    # 2. Continuity vertex 1
    M[2, [1, 2, 5, 6]] = [1, 1, -1, -1]
    b[2] = g_3(0) - g_1(0)
    
    # 3. Continuity vertex 2
    M[3, [1, 2, 7, 8]] = [1, exp_λ, -1, -1]
    b[3] = g_4(0) - g_1(1)
    
    # 4. Continuity vertex 3
    M[4, [3, 4, 7, 8]] = [1, exp_λ, -1, -exp_λ]
    b[4] = g_4(1) - g_2(1)

    # 5. Dirichlet vertex 4 
    M[5, 5:6] = [1, exp_λ]
    b[5] = -g_3(1) + dirichlet_value

    flux_g(g_val, gp_val) = -D*gp_val + A*g_val

    # 6. Flow vertex 1
    M[6, [1, 2, 3, 4, 5, 6]] = [-A, -2*A, -A, -2*A, -A, -2*A]
    b[6] = flux_g(g_1(0), g_1_p(0)) + flux_g(g_2(0), g_2_p(0)) + flux_g(g_3(0), g_3_p(0))
    
    # 7. Flow vertex 2
    M[7, [1, 2, 7, 8]] = [A, 2*A*exp_λ, -A, -2*A]
    b[7] = flux_g(g_4(0), g_4_p(0)) - flux_g(g_1(1), g_1_p(1))
    
    # 8. Flow vertex 3
    M[8, [3, 4, 7, 8]] = [A, 2*A*exp_λ, A, 2*A*exp_λ]
    b[8] = -flux_g(g_2(1), g_2_p(1)) - flux_g(g_4(1), g_4_p(1))
    
    c = M \ b

    u_e(x,c,g) = c[1] + c[2]*exp(λ*x) + g(x)
    ∂u_e(x,c,g_p) = λ*c[2]*exp(λ*x) + g_p(x)
    ∂∂u_e(x,c,g_pp) = λ^2*c[2]*exp(λ*x) + g_pp(x)
    f_e(x,c,g_p,g_pp) = -D*∂∂u_e(x,c,g_pp)+A*∂u_e(x,c,g_p)

    u(x, c) = [
        u_e(x, [c[1],c[2]], g_1)
        u_e(x, [c[3],c[4]], g_2)
        u_e(x, [c[5],c[6]], g_3)
        u_e(x, [c[7],c[8]], g_4)
    ]

    f(x,c) = [
        f_e(x, [c[1],c[2]], g_1_p, g_1_pp)
        f_e(x, [c[3],c[4]], g_2_p, g_2_pp)
        f_e(x, [c[5],c[6]], g_3_p, g_3_pp)
        f_e(x, [c[7],c[8]], g_4_p, g_4_pp)
    ]

    if show_plots
        xx = range(0, 1, length=100)
        p1 = plot(xx,u_e.(xx, Ref([c[3],c[4]]), g_2), label="u_13")
        plot!(p1, xx, g_2.(xx), label="g_2")
        plot!(p1, xx, c[3].+c[4]*exp.(λ*xx), label="exp-part")
        display(p1)
        plot_solution(c, f, "RHS f; D=$D, A=$A");
        plot_solution(c, u, "Exact Sol.; D=$D, A=$A")
    end

    exact_sol(x) = u(x,c)
    Rhs = [
        x -> f_e(x, [c[1],c[2]], g_1_p, g_1_pp)
        x -> f_e(x, [c[3],c[4]], g_2_p, g_2_pp)
        x -> f_e(x, [c[5],c[6]], g_3_p, g_3_pp)
        x -> f_e(x, [c[7],c[8]], g_4_p, g_4_pp)
    ]
    
    return exact_sol, Rhs
end

"""
    testcase_triangle_exact_WOx(A::Real, D::Real, J::Integer; dirichlet_node::Integer=4, dirichlet_value::Real=0.0, use_shishkin::Bool=false, shishkin_mode::Symbol=:symmetric, ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0, show_plots_exact::Bool=false)

Generates the triangle problem with a known exact solution in case format. The triangle is extended with one Dirichlet vertex. The ansatz functions are of the form c[1] + c[2]*exp(λ*x) + g(x), where λ = -A/D and g(x) is a polynomial function. The right-hand side vector is computed based on the exact solution. At vertex 3, a full inflow condition is created. 

# Arguments
- `A::Real`: Advection coefficient.
- `D::Real`: Diffusion coefficient.
- `J::Integer`: Refinement level determining the number of elements per edge (2^J).
- `dirichlet_node::Integer`: Node index for Dirichlet boundary condition (default: `4`).
- `dirichlet_value::Real`: Value of the Dirichlet boundary condition (default: `0.0`).
- `use_shishkin::Bool`: Flag whether to use a Shishkin layer mesh (default: `false`).
- `shishkin_mode::Symbol`: Mode for the Shishkin grid (default: `:symmetric`).
- `ρ::Real`: Scaling parameter for the Shishkin mesh (default: `2.0`).
- `cap::Real`: Cap parameter for the Shishkin mesh (default: `0.25`).
- `pe_switch::Real`: Péclet switch parameter (default: `1.0`).
- `show_plots_exact::Bool`: Flag to show plots during exact solution computation (default: `false`).

# Output
- A named tuple containing the triangle test case configuration.
"""
function testcase_triangle_exact_WOx(A::Real, D::Real, J::Integer;
                                 dirichlet_node::Integer=4,
                                 dirichlet_value::Real=0.0,
                                 use_shishkin::Bool=false,
                                 shishkin_mode::Symbol=:symmetric,
                                 ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0,
                                 show_plots_exact::Bool=false)
                                 
    u_exakt, rhs_vec_any = exact_solution_triangle_WOx(float(A), float(D), float(dirichlet_value); 
                                                       show_plots=show_plots_exact)
    rhs_vec = Function[f for f in rhs_vec_any]

    edges = [Edge(1,2), Edge(1,3), Edge(1,4), Edge(2,3)]
    nv = 4
    L = 1.0

    N_elements = 2^J
    n_e = Dict(e => Int(N_elements) for e in edges)
    
    eps_edge = Dict(e => float(D) for e in edges)
    a_edge   = Dict(e => float(A) for e in edges)
    f_edge   = Dict(e => rhs_vec[i] for (i,e) in enumerate(edges))

    edge_x = Dict{Graphs.SimpleGraphs.SimpleEdge{Int},Vector{Float64}}()
    for e in edges
        N = n_e[e]
        if use_shishkin
            edge_x[e] = shishkin_grid(L, N, float(A), float(D);
                                      ρ=ρ, mode=shishkin_mode, cap=cap, pe_switch=pe_switch)
        else
            edge_x[e] = collect(range(0.0, L; length=N+1))
        end
    end

    dirichlet = Dict{Int, Float64}(dirichlet_node => float(dirichlet_value))

    return (name="testcase_triangle_exact",
            nv=nv, edges=edges, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=u_exakt,
            use_shishkin=use_shishkin)
end


"""
    exact_solution_triangle_any(A, D, dirichlet_value; show_plots::Bool=false)

Computes an alternative exact solution and right-hand side for the triangle graph problem.
Teh function g is chosen as trigonometric functions. The ansatz functions are of the form c[1] + c[2]*exp(λ*x) + g(x), where λ = -A/D and g(x) is a trigonometric function.
# Arguments
- `A`: Advection coefficient.
- `D`: Diffusion coefficient.
- `dirichlet_value`: Value for the Dirichlet boundary condition.
- `show_plots::Bool`: Flag to display plots (default: `false`).

# Output
- A tuple `(exact_sol, Rhs)` containing the exact solution function and the vector of right-hand side functions per edge.
"""
function exact_solution_triangle_any(A,D,dirichlet_value; show_plots::Bool=false)
    M = zeros(8,8)
    b = zeros(8)
    
    g_1(x) = sin(pi*x)
    g_2(x) = sin(pi*x)
    g_3(x) = sin(pi*x)
    g_4(x) = sin(pi*x)
    g_1_p(x) = pi*cos(pi*x)
    g_2_p(x) = pi*cos(pi*x)
    g_3_p(x) = pi*cos(pi*x)
    g_4_p(x) = pi*cos(pi*x)
    g_1_pp(x) = -pi^2*sin(pi*x)
    g_2_pp(x) = -pi^2*sin(pi*x)
    g_3_pp(x) = -pi^2*sin(pi*x)
    g_4_pp(x) = -pi^2*sin(pi*x)

    M[1,1] = 1; M[1,3] = -1; b[1] = g_2(0)-g_1(0)
    M[2,1] = 1; M[2,5] = -1; b[2] = g_3(0)-g_1(0)
    M[3,1] = 1; M[3,7] = -1; b[3] = g_4(0)-g_1(1)
    M[4,4] = 1; M[4,8] = -1; b[4] = g_4(1)-g_1(1)

    M[5,6] = 1; b[5] = -g_3(1) + dirichlet_value

    M[6,2] = D*exp(-A/D); M[6,4] = D*exp(-A/D); M[6,6] = D*exp(-A/D); M[6,1] = -(D+A); M[6,3] = -(D+A); M[6,5] = -(D+A); b[6] = -D*g_1_p(0) + A*g_1(0) - D*g_2_p(0) + A*g_2(0) - D*g_3_p(0) + A*g_3(0)

    M[7,1] = D; M[7,2] = -D; M[7,7] = -(D+A); M[7,8] = D*exp(-A/D); b[7] = D*g_1_p(1) - D*g_4_p(0) - A*g_1(1) + A*g_4(0)

    M[8,3] = D; M[8,4] = -D; M[8,7] = D; M[8,8] = -D; b[8] = D*g_2_p(1) + D*g_4_p(1) - A*g_2(1) - A*g_4(1)

    c = M\b

    u_e(x,c,g) = c[1] + (c[2]*exp(-A/D*(1-x))-c[1])*x + g(x)
    ∂u_e(x,c,gp) = (c[2]*exp(-A/D*(1-x))-c[1]) + c[2]*A/D*exp(-A/D*(1-x))*x + gp(x)
    ∂∂u_e(x,c,gpp) = 2*A/D*c[2]*exp(-A/D*(1-x)) + (A/D)^2*c[2]*exp(-A/D*(1-x))*x + gpp(x)

    f_e(x,c,g,gp,gpp) = -D*∂∂u_e(x,c,gpp) + A*∂u_e(x,c,gp)

    u(x,c) = [
        u_e(x, [c[1],c[2]], g_1)
        u_e(x, [c[3],c[4]], g_2)
        u_e(x, [c[5],c[6]], g_3)
        u_e(x, [c[7],c[8]], g_4)
    ]
    ∂u(x,c) = [
        ∂u_e(x, [c[1],c[2]], g_1_p)
        ∂u_e(x, [c[3],c[4]], g_2_p)
        ∂u_e(x, [c[5],c[6]], g_3_p)
        ∂u_e(x, [c[7],c[8]], g_4_p)
    ]
    ∂∂u(x,c) = [
        ∂∂u_e(x, [c[1],c[2]], g_1_pp)
        ∂∂u_e(x, [c[3],c[4]], g_2_pp)
        ∂∂u_e(x, [c[5],c[6]], g_3_pp)
        ∂∂u_e(x, [c[7],c[8]], g_4_pp)
    ]
    f(x,c) = [
        f_e(x, [c[1],c[2]], g_1, g_1_p, g_1_pp)
        f_e(x, [c[3],c[4]], g_2, g_2_p, g_2_pp)
        f_e(x, [c[5],c[6]], g_3, g_3_p, g_3_pp)
        f_e(x, [c[7],c[8]], g_4, g_4_p, g_4_pp)
    ]

    flux_e(x, c_edge, g, gp) = -D * ∂u_e(x, c_edge, gp) + A * u_e(x, c_edge, g)

    q_v1 = -flux_e(0.0, [c[1], c[2]], g_1, g_1_p) - flux_e(0.0, [c[3], c[4]], g_2, g_2_p)- flux_e(0.0, [c[5], c[6]], g_3, g_3_p)
    q_v2 = flux_e(1.0, [c[1], c[2]], g_1, g_1_p) - flux_e(0.0, [c[7], c[8]], g_4, g_4_p)
    q_v3 = flux_e(1.0, [c[3], c[4]], g_2, g_2_p) + flux_e(1.0, [c[7], c[8]], g_4, g_4_p)

    if show_plots
        plot_solution(c, f, "RHS f; D=$D, A=$A");
        plot_solution(c, u, "Exakte Lsg.; D=$D, A=$A");
    end

    exact_sol(x) = u(x,c)
    Rhs = [
        x -> f_e(x, [c[1],c[2]], g_1, g_1_p, g_1_pp)
        x -> f_e(x, [c[3],c[4]], g_2, g_2_p, g_2_pp)
        x -> f_e(x, [c[5],c[6]], g_3, g_3_p, g_3_pp)
        x -> f_e(x, [c[7],c[8]], g_4, g_4_p, g_4_pp)
    ]
    return exact_sol, Rhs
end


"""
    testcase_triangle_exact_any(A::Real, D::Real, J::Integer; dirichlet_node::Integer=4, dirichlet_value::Real=0.0, use_shishkin::Bool=false, shishkin_mode::Symbol=:symmetric, ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0, show_plots_exact::Bool=false)

Generates the triangle problem with an alternative known exact solution in case format. The triangle is extended with one Dirichlet vertex. The ansatz functions are of the form c[1] + c[2]*exp(λ*x) + g(x), where λ = -A/D and g(x) is a trigonometric function. At vertex 3 a full inflow condition is created. 

# Arguments
- `A::Real`: Advection coefficient.
- `D::Real`: Diffusion coefficient.
- `J::Integer`: Refinement level determining the number of elements per edge (2^J).
- `dirichlet_node::Integer`: Node index for Dirichlet boundary condition (default: `4`).
- `dirichlet_value::Real`: Value of the Dirichlet boundary condition (default: `0.0`).
- `use_shishkin::Bool`: Flag whether to use a Shishkin layer mesh (default: `false`).
- `shishkin_mode::Symbol`: Mode for the Shishkin grid (default: `:symmetric`).
- `ρ::Real`: Scaling parameter for the Shishkin mesh (default: `2.0`).
- `cap::Real`: Cap parameter for the Shishkin mesh (default: `0.25`).
- `pe_switch::Real`: Péclet switch parameter (default: `1.0`).
- `show_plots_exact::Bool`: Flag to show plots during exact solution computation (default: `false`).

# Output
- A named tuple containing the test case configuration.
"""
function testcase_triangle_exact_any(A::Real, D::Real, J::Integer;
                                 dirichlet_node::Integer=4,
                                 dirichlet_value::Real=0.0,
                                 use_shishkin::Bool=false,
                                 shishkin_mode::Symbol=:symmetric,
                                 ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0,
                                 show_plots_exact::Bool=false)
                                 
    # Exakte Lösung aufrufen (Plots werden nur gezeigt, wenn gefordert)
    u_exakt, rhs_vec_any = exact_solution_triangle_any(float(A), float(D), float(dirichlet_value); show_plots=show_plots_exact)
    rhs_vec = Function[f for f in rhs_vec_any]

    edges = [Edge(1,2), Edge(1,3), Edge(1,4), Edge(2,3)]
    nv = 4
    L = 1.0

    N_elements = 2^J
    n_e = Dict(e => Int(N_elements) for e in edges)
    
    eps_edge = Dict(e => float(D) for e in edges)
    a_edge   = Dict(e => float(A) for e in edges)
    f_edge   = Dict(e => rhs_vec[i] for (i,e) in enumerate(edges))

    edge_x = Dict{Graphs.SimpleGraphs.SimpleEdge{Int},Vector{Float64}}()
    for e in edges
        N = n_e[e]
        if use_shishkin
            edge_x[e] = shishkin_grid(L, N, float(A), float(D);
                                      ρ=ρ, mode=shishkin_mode, cap=cap, pe_switch=pe_switch)
        else
            edge_x[e] = collect(range(0.0, L; length=N+1))
        end
    end

    dirichlet = Dict{Int, Float64}(dirichlet_node => float(dirichlet_value))

    return (name="testcase_triangle_exact_any",
            nv=nv, edges=edges, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=u_exakt,
            use_shishkin=use_shishkin)
end


"""
    exact_solution_star_Kumar_Leugering(D1, D2, D3; L=1.0, A1=-2.0, A2=-1.0, A3=-1.0, dirichlet_v1=1.0, dirichlet_v2=0.0, dirichlet_v3=0.0)

Computes the exact solution for a star graph problem following Kumar & Leugering.
The ansatz functions are of the form:
u1(x) = c1 + c2*exp(-2x/D1)
u2(x) = d1 + d2*exp(- (L-x)/D2) + (L-x)
u3(x) = e1 + e2*exp(- (L-x)/D3) + (L-x)^2
with a Dirichlet vertex at v_1.

# Arguments
- `D1`, `D2`, `D3`: Diffusion coefficients for the three edges.
- `L`: Edge length (default: `1.0`).
- `A1`, `A2`, `A3`: Advection coefficients for the three edges (defaults: `-2.0`, `-1.0`, `-1.0`).
- `dirichlet_v1`, `dirichlet_v2`, `dirichlet_v3`: Dirichlet boundary values at the outer nodes (defaults: `1.0`, `0.0`, `0.0`).

# Output
- A tuple `(exakte_Loesung, exakte_Loesung_prime, exakte_Loesung_k, Rhs)` containing the exact solution function, its first derivative, its k-th derivative function, and the right-hand side vector.
"""
function exact_solution_star_Kumar_Leugering(D1, D2, D3;
                             L=1.0, 
                             A1=-2.0, A2=-1.0, A3=-1.0,
                             dirichlet_v1=1.0, dirichlet_v2=0.0, dirichlet_v3=0.0)

    D1 = float(D1); D2 = float(D2); D3 = float(D3)
    L  = float(L)
    A1 = float(A1); A2 = float(A2); A3 = float(A3)

    (D1>0 && D2>0 && D3>0) || error("All D_i must be > 0.")
    (L>0) || error("L must be > 0.")

    # Calculate exponential terms at x = L
    E1 = exp(-2L / D1)
    E2 = exp(-L / D2)
    E3 = exp(-L / D3)

    M = zeros(6,6)
    b = zeros(6)

    # (1) Dirichlet u1(0) = dirichlet_v1
    M[1,1] = 1; M[1,2] = 1; b[1] = dirichlet_v1

    # (2) Dirichlet u2(0) = dirichlet_v2
    M[2,3] = 1; M[2,4] = 1; b[2] = dirichlet_v2

    # (3) Dirichlet u3(0) = dirichlet_v3
    M[3,5] = 1; M[3,6] = 1; b[3] = dirichlet_v3

    # (4) Continuity in v4: u1(L) = u2(L)
    M[4,1] = 1; M[4,2] = E1
    M[4,3] = -1; M[4,4] = -E2
    b[4] = -L

    # (5) Continuity in v4: u1(L) = u3(L)
    M[5,1] = 1; M[5,2] = E1
    M[5,5] = -1; M[5,6] = -E3
    b[5] = 2*D3*L - L^2

    # (6) Flux condition in v4
    M[6,1] = A1; M[6,2] = (2 + A1) * E1
    M[6,3] = A2; M[6,4] = (1 + A2) * E2
    M[6,5] = A3; M[6,6] = (1 + A3) * E3
    
    const_term = D2 - A2*L - 2*D3^2 + 2*D3*L + 2*A3*D3*L - A3*L^2
    b[6] = -const_term

    p = M \ b
    c1, c2, d1, d2, e1, e2 = p

    # General function for the k-th derivative
    function exakte_Loesung_k(x, k::Int)
        # Edge 1: u1(x) = c1 + c2*exp(-2x/D1)
        u1_val = c2 * (-2/D1)^k * exp(-2x/D1)
        if k == 0
            u1_val += c1
        end

        # Edge 2: u2(x) = d1 + d2*exp(-x/D2) - x
        u2_val = d2 * (-1/D2)^k * exp(-x/D2)
        if k == 0
            u2_val += (d1 - x)
        elseif k == 1
            u2_val -= 1.0
        end

        # Edge 3: u3(x) = e1 + e2*exp(-x/D3) + 2*D3*x - x^2
        u3_val = e2 * (-1/D3)^k * exp(-x/D3)
        if k == 0
            u3_val += (e1 + 2*D3*x - x^2)
        elseif k == 1
            u3_val += (2*D3 - 2*x)
        elseif k == 2
            u3_val -= 2.0
        end

        return [u1_val, u2_val, u3_val]
    end

    exakte_Loesung(x) = exakte_Loesung_k(x, 0)
    exakte_Loesung_prime(x) = exakte_Loesung_k(x, 1)

    f1(x) = -D1 * exakte_Loesung_k(x, 2)[1] + A1 * exakte_Loesung_k(x, 1)[1]
    f2(x) = -D2 * exakte_Loesung_k(x, 2)[2] + A2 * exakte_Loesung_k(x, 1)[2]
    f3(x) = -D3 * exakte_Loesung_k(x, 2)[3] + A3 * exakte_Loesung_k(x, 1)[3]

    Rhs = [x -> f1(x), x -> f2(x), x -> f3(x)]

    return exakte_Loesung, exakte_Loesung_prime, exakte_Loesung_k, Rhs
end


"""
    testcase_star_exact_Kumar_Leugering(D1::Real, D2::Real, D3::Real, J::Integer; A1::Real=-2.0, A2::Real=-1.0, A3::Real=-1.0, dirichlet_v1::Real=1.0, dirichlet_v2::Real=0.0, dirichlet_v3::Real=0.0, layer_mesh::Symbol=:uniform, shishkin_mode::Symbol=:outflow, ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0)

Generates the Kumar-Leugering star graph test case with known exact solution in case format. Uses the reference solution from `exact_solution_star_Kumar_Leugering` to compute the right-hand side vector. The mesh can be uniform, Shishkin, or Bakhvalov.

# Arguments
- `D1::Real`, `D2::Real`, `D3::Real`: Diffusion coefficients for the three edges.
- `J::Integer`: Refinement level determining the number of elements per edge (2^J).
- `A1::Real`, `A2::Real`, `A3::Real`: Advection coefficients (defaults: `-2.0`, `-1.0`, `-1.0`).
- `dirichlet_v1::Real`, `dirichlet_v2::Real`, `dirichlet_v3::Real`: Dirichlet values at the boundary nodes.
- `layer_mesh::Symbol`: Type of boundary layer mesh (`:uniform`, `:shishkin`, or `:bakhvalov`, default: `:uniform`).
- `shishkin_mode::Symbol`: Shishkin/Bakhvalov mode (default: `:outflow`).
- `ρ::Real`: Scaling parameter (default: `2.0`).
- `cap::Real`: Cap parameter (default: `0.25`).
- `pe_switch::Real`: Péclet switch parameter (default: `1.0`).

# Output
- A named tuple containing the test case configuration.
"""
function testcase_star_exact_Kumar_Leugering(D1::Real, D2::Real, D3::Real, J::Integer;
                                 A1::Real=-2.0, A2::Real=-1.0, A3::Real=-1.0,
                                 dirichlet_v1::Real=1.0, dirichlet_v2::Real=0.0, dirichlet_v3::Real=0.0,
                                 layer_mesh::Symbol=:uniform, # :shishkin, :bakhvalov oder :uniform
                                 shishkin_mode::Symbol=:outflow,
                                 ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0)
                                 
    u_exakt, u_exakt_prime, u_exakt_k, rhs_vec_any = exact_solution_star_Kumar_Leugering(float(D1), float(D2), float(D3); 
                                        A1=A1, A2=A2, A3=A3, 
                                        dirichlet_v1=dirichlet_v1, dirichlet_v2=dirichlet_v2, dirichlet_v3=dirichlet_v3)
    rhs_vec = Function[f for f in rhs_vec_any]

    edges = [Edge(1,4), Edge(2,4), Edge(3,4)]
    nv = 4
    L = 1.0

    N_elements = 2^J
    n_e = Dict(e => Int(N_elements) for e in edges)
    
    eps_edge = Dict(edges[1] => float(D1), edges[2] => float(D2), edges[3] => float(D3))
    a_edge   = Dict(edges[1] => float(A1), edges[2] => float(A2), edges[3] => float(A3))
    f_edge   = Dict(e => rhs_vec[i] for (i,e) in enumerate(edges))

    edge_x = Dict{typeof(edges[1]),Vector{Float64}}()
    for (i, e) in enumerate(edges)
        N = n_e[e]
        D_e = eps_edge[e]
        A_e = a_edge[e]
        
        if layer_mesh == :bakhvalov
            # Bakhvalov needs explicit :right or :left for outflow
            bakhvalov_side = if shishkin_mode == :symmetric
                :both
            else
                (A_e >= 0) ? :right : :left
            end
            
            edge_x[e] = bakhvalov_nodes(L, N, D_e, A_e; side=bakhvalov_side, sigma_factor=ρ, pe_switch=pe_switch)
            
        elseif layer_mesh == :shishkin
            edge_x[e] = shishkin_grid(L, N, A_e, D_e; ρ=ρ, mode=shishkin_mode, cap=cap, pe_switch=pe_switch)          
        else 
            edge_x[e] = collect(range(0.0, L; length=N+1))
        end
    end

    dirichlet = Dict{Int, Float64}(1 => float(dirichlet_v1), 
                                   2 => float(dirichlet_v2), 
                                   3 => float(dirichlet_v3))

    return (name="testcase_star_KumarLeugering",
            nv=nv, edges=edges, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=u_exakt,
            layer_mesh=layer_mesh,
            exact_derivative=u_exakt_prime,
            exact_k_th_derivative=u_exakt_k) 
end


"""
    exact_solution_star_MMS(D1, D2, D3; L=1.0, A1=-2.0, A2=-1.0, A3=-1.0, dirichlet_v1=1.0, dirichlet_v2=2.0, dirichlet_v3=-1.0)

Computes an exact solution using the Method of Manufactured Solutions (MMS) for a star graph configuration.

# Arguments
- `D1`, `D2`, `D3`: Diffusion coefficients.
- `L`: Edge length (default: `1.0`).
- `A1`, `A2`, `A3`: Advection coefficients.
- `dirichlet_v1`, `dirichlet_v2`, `dirichlet_v3`: Boundary parameters defining the smooth parabolic manufactured solutions.

# Output
- A tuple `(exakte_Loesung, exakte_Loesung_prime, Rhs)` containing the exact solution function, its derivative function, and the right-hand side vector.
"""
function exact_solution_star_MMS(D1, D2, D3;
                                 L=1.0, 
                                 A1=-2.0, A2=-1.0, A3=-1.0,
                                 dirichlet_v1=1.0, dirichlet_v2=2.0, dirichlet_v3=-1.0)

    u1(x)  = dirichlet_v1 * (x - L)^2 / L^2
    du1(x) = dirichlet_v1 * 2 * (x - L) / L^2
    d2u1(x)= dirichlet_v1 * 2 / L^2

    u2(x)  = dirichlet_v2 * (x - L)^2 / L^2
    du2(x) = dirichlet_v2 * 2 * (x - L) / L^2
    d2u2(x)= dirichlet_v2 * 2 / L^2

    u3(x)  = dirichlet_v3 * (x - L)^2 / L^2
    du3(x) = dirichlet_v3 * 2 * (x - L) / L^2
    d2u3(x)= dirichlet_v3 * 2 / L^2

    f1(x) = -D1*d2u1(x) + A1*du1(x)
    f2(x) = -D2*d2u2(x) + A2*du2(x)
    f3(x) = -D3*d2u3(x) + A3*du3(x)

    exakte_Loesung(x) = [u1(x), u2(x), u3(x)]
    Rhs = [x -> f1(x), x -> f2(x), x -> f3(x)]
    exakte_Loesung_prime(x) = [du1(x), du2(x), du3(x)]
    
    return exakte_Loesung, exakte_Loesung_prime, Rhs
end


"""
    testcase_star_MMS_smooth(D1::Real, D2::Real, D3::Real, J::Integer; A1::Real=-2.0, A2::Real=-1.0, A3::Real=-1.0, dirichlet_v1::Real=1.0, dirichlet_v2::Real=2.0, dirichlet_v3::Real=-1.0, layer_mesh::Symbol=:uniform, shishkin_mode::Symbol=:outflow, ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0)

Generates the smooth MMS star graph test case using `exact_solution_star_MMS` in case format.

# Arguments
- `D1::Real`, `D2::Real`, `D3::Real`: Diffusion coefficients.
- `J::Integer`: Refinement level.
- `A1::Real`, `A2::Real`, `A3::Real`: Advection coefficients.
- `dirichlet_v1::Real`, `dirichlet_v2::Real`, `dirichlet_v3::Real`: Boundary values.
- `layer_mesh::Symbol`: Mesh type (`:uniform`, `:shishkin`, or `:bakhvalov`).
- `shishkin_mode::Symbol`: Grid generation mode.
- `ρ::Real`, `cap::Real`, `pe_switch::Real`: Mesh control parameters.

# Output
- A named tuple containing the test case configuration.
"""
function testcase_star_MMS_smooth(D1::Real, D2::Real, D3::Real, J::Integer;
                                 A1::Real=-2.0, A2::Real=-1.0, A3::Real=-1.0,
                                 dirichlet_v1::Real=1.0, dirichlet_v2::Real=2.0, dirichlet_v3::Real=-1.0, layer_mesh::Symbol=:uniform, shishkin_mode::Symbol=:outflow, ρ::Real=2.0, cap::Real=0.25, pe_switch::Real=1.0)
                                 
    u_exakt, u_exakt_prime, rhs_vec_any = exact_solution_star_MMS(float(D1), float(D2), float(D3); 
                                        A1=A1, A2=A2, A3=A3, 
                                        dirichlet_v1=dirichlet_v1, dirichlet_v2=dirichlet_v2, dirichlet_v3=dirichlet_v3)
    rhs_vec = Function[f for f in rhs_vec_any]

    edges = [Edge(1,4), Edge(2,4), Edge(3,4)]
    nv = 4
    L = 1.0

    N_elements = 2^J
    n_e = Dict(e => Int(N_elements) for e in edges)
    
    eps_edge = Dict(edges[1] => float(D1), edges[2] => float(D2), edges[3] => float(D3))
    a_edge   = Dict(edges[1] => float(A1), edges[2] => float(A2), edges[3] => float(A3))
    f_edge   = Dict(e => rhs_vec[i] for (i,e) in enumerate(edges))

    edge_x = Dict{typeof(edges[1]),Vector{Float64}}()
    for (i, e) in enumerate(edges)
        N = n_e[e]
        D_e = eps_edge[e]
        A_e = a_edge[e]
        
        if layer_mesh == :bakhvalov
            # Bakhvalov requires explicit :right or :left for outflow
            bakhvalov_side = if shishkin_mode == :symmetric
                :both
            else
                (A_e >= 0) ? :right : :left
            end
            
            edge_x[e] = bakhvalov_nodes(L, N, D_e, A_e; side=bakhvalov_side, sigma_factor=ρ, pe_switch=pe_switch)
            
        elseif layer_mesh == :shishkin
            edge_x[e] = shishkin_grid(L, N, A_e, D_e; ρ=ρ, mode=shishkin_mode, cap=cap, pe_switch=pe_switch)
            
        else # Fallback to uniform grid
            edge_x[e] = collect(range(0.0, L; length=N+1))
        end
    end

    dirichlet = Dict{Int, Float64}(1 => float(dirichlet_v1), 
                                   2 => float(dirichlet_v2), 
                                   3 => float(dirichlet_v3))

    return (name="testcase_star_MMS",
            nv=nv, edges=edges, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=u_exakt,
            layer_mesh=layer_mesh,
            exact_derivative=u_exakt_prime) 
end


"""
    exact_solution_cycle_Lollipop(D_val::Real, A_val::Real; L=1.0, dirichlet_v1=1.0)

Computes the exact solution for a lollipop graph configuration. The effective orientation of the loop is chosen to contain a cycle. The ansatz functions are of the form c[1] + c[2]*exp(λ*x) + g(x), where λ = A/D and g(x) is a trigonometric function. The Dirichlet boundary condition is applied at vertex 1.

# Arguments
- `D_val::Real`: Diffusion coefficient for all edges.
- `A_val::Real`: Advection coefficient for all edges.
- `L`: Edge length (default: `1.0`).
- `dirichlet_v1`: Dirichlet boundary value (default: `1.0`).

# Output
- A tuple `(exact_sol, Rhs)` containing the exact solution function and the right-hand side functions vector.
"""
function exact_solution_cycle_Lollipop(D_val::Real, A_val::Real; L=1.0, dirichlet_v1=1.0)
    D_vec = fill(D_val, 5)
    A_vec = fill(A_val, 5)
    @assert length(D_vec) == 5 && length(A_vec) == 5 "We need 5 D and 5 A values"

    M = zeros(10, 10)
    b = zeros(10)
    
    λ(i) = A_vec[i] / D_vec[i]
    
    # Dirichlet vertex at v_1
    M[1, 2] = 1.0; b[1] = 0.0
    
    #  Continuity at v_2
    M[2, 1] = 1.0; M[2, 3] = -1.0; b[2] = 0.0
    # u5(1) = u2(0) => c52 = c21
    M[3, 10] = 1.0; M[3, 3] = -1.0; b[3] = 0.0
    
    # Continuity at v_3
    M[4, 4] = 1.0; M[4, 5] = -1.0; b[4] = 0.0
    # Continuity at v_4
    M[5, 6] = 1.0; M[5, 7] = -1.0; b[5] = 0.0
    # Continuity at v_5
    M[6, 8] = 1.0; M[6, 9] = -1.0; b[6] = 0.0
    
    # Flow at v_2
    M[7, 9] = D_vec[5]; M[7, 10] = -D_vec[5]                            #
    M[7, 1] = -(D_vec[1] + A_vec[1]); M[7, 2] = D_vec[1]*exp(-λ(1))    
    M[7, 3] = -(D_vec[2] + A_vec[2]); M[7, 4] = D_vec[2]*exp(-λ(2))    
    b[7] = -(D_vec[5] + D_vec[1] + D_vec[2]) * pi
    
    # Flow at v_3
    M[8, 3] = D_vec[2]; M[8, 4] = -D_vec[2]                             
    M[8, 5] = -(D_vec[3] + A_vec[3]); M[8, 6] = D_vec[3]*exp(-λ(3))     
    b[8] = -(D_vec[2] + D_vec[3]) * pi
    
    # Flow at v_4
    M[9, 5] = D_vec[3]; M[9, 6] = -D_vec[3]                             
    M[9, 7] = -(D_vec[4] + A_vec[4]); M[9, 8] = D_vec[4]*exp(-λ(4))     
    b[9] = -(D_vec[3] + D_vec[4]) * pi
    
    # Flow at v_5
    M[10, 7] = D_vec[4]; M[10, 8] = -D_vec[4]                           
    M[10, 9] = -(D_vec[5] + A_vec[5]); M[10, 10] = D_vec[5]*exp(-λ(5))  
    b[10] = -(D_vec[4] + D_vec[5]) * pi

    p = M \ b
    
    c11, c12 = p[1], p[2]
    c21, c22 = p[3], p[4]
    c31, c32 = p[5], p[6]
    c41, c42 = p[7], p[8]
    c51, c52 = p[9], p[10]

    lam1 = λ(1); lam2 = λ(2); lam3 = λ(3); lam4 = λ(4); lam5 = λ(5)
    
    u1(x)   = c11 + (c12 * exp(-lam1 * (1 - x)) - c11) * x + sin(pi * x)
    du1(x)  = c12 * exp(-lam1 * (1 - x)) * (lam1 * x + 1) - c11 + pi * cos(pi * x)
    d2u1(x) = c12 * lam1 * exp(-lam1 * (1 - x)) * (lam1 * x + 2) - pi^2 * sin(pi * x)

    u2(x)   = c21 + (c22 * exp(-lam2 * (1 - x)) - c21) * x + sin(pi * x)
    du2(x)  = c22 * exp(-lam2 * (1 - x)) * (lam2 * x + 1) - c21 + pi * cos(pi * x)
    d2u2(x) = c22 * lam2 * exp(-lam2 * (1 - x)) * (lam2 * x + 2) - pi^2 * sin(pi * x)

    u3(x)   = c31 + (c32 * exp(-lam3 * (1 - x)) - c31) * x + sin(pi * x)
    du3(x)  = c32 * exp(-lam3 * (1 - x)) * (lam3 * x + 1) - c31 + pi * cos(pi * x)
    d2u3(x) = c32 * lam3 * exp(-lam3 * (1 - x)) * (lam3 * x + 2) - pi^2 * sin(pi * x)

    u4(x)   = c41 + (c42 * exp(-lam4 * (1 - x)) - c41) * x + sin(pi * x)
    du4(x)  = c42 * exp(-lam4 * (1 - x)) * (lam4 * x + 1) - c41 + pi * cos(pi * x)
    d2u4(x) = c42 * lam4 * exp(-lam4 * (1 - x)) * (lam4 * x + 2) - pi^2 * sin(pi * x)

    u5(x)   = c51 + (c52 * exp(-lam5 * (1 - x)) - c51) * x + sin(pi * x)
    du5(x)  = c52 * exp(-lam5 * (1 - x)) * (lam5 * x + 1) - c51 + pi * cos(pi * x)
    d2u5(x) = c52 * lam5 * exp(-lam5 * (1 - x)) * (lam5 * x + 2) - pi^2 * sin(pi * x)

    f1(x) = -D_vec[1] * d2u1(x) + A_vec[1] * du1(x)
    f2(x) = -D_vec[2] * d2u2(x) + A_vec[2] * du2(x)
    f3(x) = -D_vec[3] * d2u3(x) + A_vec[3] * du3(x)
    f4(x) = -D_vec[4] * d2u4(x) + A_vec[4] * du4(x)
    f5(x) = -D_vec[5] * d2u5(x) + A_vec[5] * du5(x)

    exact_sol(x) = [u1(x), u2(x), u3(x), u4(x), u5(x)]
    exact_sol_prime(x) = [du1(x), du2(x), du3(x), du4(x), du5(x)]
    
    Rhs(x) = [f1(x), f2(x), f3(x), f4(x), f5(x)]
    
    return exact_sol, Rhs
end


"""
    testcase_cycle_advection(D_val::Real, A_val::Real, J::Integer; dirichlet_v1::Real=0.0, layer_mesh::Symbol=:uniform)

Generates a lollipop advection-diffusion test case in case format. Effetive orientation is chosen to contain a cycle. The exact solution is computed using `exact_solution_cycle_Lollipop`. 

# Arguments
- `D_val::Real`: Diffusion coefficient.
- `A_val::Real`: Advection coefficient.
- `J::Integer`: Refinement level.
- `dirichlet_v1::Real`: Dirichlet boundary value (default: `0.0`).
- `layer_mesh::Symbol`: Type of boundary layer mesh (default: `:uniform`).

# Output
- A named tuple containing the test case configuration.
"""
function testcase_cycle_advection(D_val::Real, A_val::Real, J::Integer;
                                  dirichlet_v1::Real=0.0,
                                  layer_mesh::Symbol=:uniform)
    
    L = 1.0
    u_ex_vec, f_ex_vec = exact_solution_cycle_Lollipop(D_val, A_val; L=L, dirichlet_v1=dirichlet_v1)
    
    edges = [Edge(2,1), Edge(2,3), Edge(3,4), Edge(4,5), Edge(5,2)]
    nv = 5
    
    N_elements = 2^J
    n_e = Dict(e => Int(N_elements) for e in edges)
    
    eps_edge = Dict(e => float(D_val) for e in edges)
    a_edge   = Dict(e => float(A_val) for e in edges)
    
    # RHS: f=0 für Kante 1, f=1 für Kanten im Loop
    f_edge = Dict(edges[1] => x -> f_ex_vec(x)[1], 
                  edges[2] => x -> f_ex_vec(x)[2], 
                  edges[3] => x -> f_ex_vec(x)[3], 
                  edges[4] => x -> f_ex_vec(x)[4],
                  edges[5] => x -> f_ex_vec(x)[5])

    edge_x = Dict{typeof(edges[1]),Vector{Float64}}()
    for e in edges
        edge_x[e] = collect(range(0.0, L; length=N_elements+1))
    end

    dirichlet = Dict{Int, Float64}(1 => float(dirichlet_v1))

    return (name="testcase_cycle_advection",
            nv=nv, edges=edges, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=u_ex_vec,
            layer_mesh=layer_mesh)
end


"""
    solve_star_poly(D::Real, A::Real, p::Int)

Computes polynomial exact solutions and their derivatives for a star graph configuration.

# Arguments
- `D::Real`: Diffusion coefficient.
- `A::Real`: Advection coefficient.
- `p::Int`: Polynomial degree (p >= 1).

# Output
- A tuple `(exakte_Loesung, exakte_Loesung_prime, exakte_Loesung_k, Rhs)` containing the exact solution, its derivative, k-th derivative, and right-hand side functions.
"""
function solve_star_poly(D::Real, A::Real, p::Int)
    if p < 1
        error("The polynomial degree p must be >= 1")
    end

    M = zeros(Float64, 6, 6)
    b = zeros(Float64, 6)

    # Condition at node v1 
    M[1, 2] = 1
    b[1] = 0

    # Condition at node v2 
    M[2, 4] = 1
    b[2] = 0

    #Continuity at node v3 
    M[3, 1] = 1.0  
    M[3, 2] = 1.0
    M[3, 6] = -1.0
    b[3] = -1.0

    # Continuity at node v3 
    M[4, 3] = 1.0  
    M[4, 4] = 1.0
    M[4, 6] = -1.0
    b[4] = -1.0

    # Kirchhoff condition at v3
    M[5, 1] = -D * p + A 
    M[5, 2] = A
    M[5, 3] = -D * p + A 
    M[5, 4] = A
    M[5, 6] = -A
    b[5] = -2A + 2D*(p-1)  

    # Dirichlet condition at v4
    M[6, 5] = 1.0  
    M[6, 6] = 1.0
    b[6] = -1.0

    c = M \ b

    u1(x)   = c[1] * x^p + x^(p-1) + c[2]
    du1(x)  = c[1] * p * x^(p-1) + (p-1) * x^(p-2)
    d2u1(x) = c[1] * p * (p-1) * x^(p-2) + (p-1) * (p-2) * x^(p-3)

    u2(x)   = c[3] * x^p + x^(p-1) + c[4]
    du2(x)  = c[3] * p * x^(p-1) + (p-1) * x^(p-2)
    d2u2(x) = c[3] * p * (p-1) * x^(p-2) + (p-1) * (p-2) * x^(p-3)

    u3(x)   = c[5] * x^p + x^(p-1) + c[6]
    du3(x)  = c[5] * p * x^(p-1) + (p-1) * x^(p-2)
    d2u3(x) = c[5] * p * (p-1) * x^(p-2) + (p-1) * (p-2) * x^(p-3)

    f1(x) = -D * d2u1(x) + A * du1(x)
    f2(x) = -D * d2u2(x) + A * du2(x)
    f3(x) = -D * d2u3(x) + A * du3(x)
    
    exact_sol(x) = [u1(x), u2(x), u3(x)]
    exact_sol_prime(x) = [du1(x), du2(x), du3(x)]
    
    Rhs(x) = [f1(x), f2(x), f3(x)]

    function exact_sol_k(x, k::Int)

        function deriv_term(c, m, dk)
            if dk == 0
                return c * x^m
            elseif dk <= m
                fac = prod((m - dk + 1):m)
                return c * fac * x^(m - dk)
            else
                return 0.0
            end
        end

        u1_val = deriv_term(c[1], p, k) + deriv_term(1.0, p - 1, k)
        if k == 0
            u1_val += c[2]
        end

        u2_val = deriv_term(c[3], p, k) + deriv_term(1.0, p - 1, k)
        if k == 0
            u2_val += c[4]
        end

        u3_val = deriv_term(c[5], p, k) + deriv_term(1.0, p - 1, k)
        if k == 0
            u3_val += c[6]
        end
        
        return [u1_val, u2_val, u3_val]
    end
   
    return exact_sol, exact_sol_prime, exact_sol_k, Rhs
end


"""
    testcase_star_poly(D_val::Real, A_val::Real, J::Integer, p::Int)

Generates the polynomial star graph test case in case format.

# Arguments
- `D_val::Real`: Diffusion coefficient.
- `A_val::Real`: Advection coefficient.
- `J::Integer`: Refinement level.
- `p::Int`: Polynomial degree.

# Output
- A named tuple containing the test case configuration.
"""
function testcase_star_poly(D_val::Real, A_val::Real, J::Integer, p::Int)
    
    L = 1.0
    u_ex_vec, _, u_exakt_k, f_ex_vec = solve_star_poly(D_val, A_val, p)
    
    edges = [Edge(1,3), Edge(2,3), Edge(3,4)]
    nv = 4
    
    N_elements = 2^J
    n_e = Dict(e => Int(N_elements) for e in edges)
    
    eps_edge = Dict(e => float(D_val) for e in edges)
    a_edge   = Dict(e => float(A_val) for e in edges)
    
    f_edge = Dict(edges[1] => x -> f_ex_vec(x)[1], 
                  edges[2] => x -> f_ex_vec(x)[2], 
                  edges[3] => x -> f_ex_vec(x)[3])

    edge_x = Dict{typeof(edges[1]),Vector{Float64}}()
    for e in edges
        edge_x[e] = collect(range(0.0, L; length=N_elements+1))
    end

    dirichlet = Dict{Int, Float64}(4 => 0.0)

    return (name="testcase_star_poly",
            nv=nv, edges=edges, n_e=n_e, edge_x=edge_x,
            eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
            dirichlet=dirichlet,
            exakte_Loesung=u_ex_vec,
            exact_k_th_derivative=u_exakt_k
            )
end


"""
    solve_star_trig(D::Real, A::Real)

Computes trigonometric exact solutions and their derivatives for a star graph configuration.

# Arguments
- `D::Real`: Diffusion coefficient.
- `A::Real`: Advection coefficient.

# Output
- A tuple `(exakte_Loesung, exakte_Loesung_prime, exakte_Loesung_k, Rhs)` containing the exact solution, its derivative, k-th derivative, and right-hand side functions.
"""
function solve_star_trig(D::Real, A::Real)

    function exact_sol_k(x, k::Int)
        u1_val =  (pi/2)^k * cos(pi*x/2 + k*pi/2)
        u2_val = -(pi/4)^k * sin(pi*x/4 + k*pi/2)
        u3_val = -(pi/4)^k * sin(pi*x/4 + k*pi/2)
        return [u1_val, u2_val, u3_val]
    end

    exact_sol(x) = exact_sol_k(x, 0)
    exact_sol_prime(x) = exact_sol_k(x, 1)

    function Rhs(x)
        u_prime = exact_sol_prime(x)
        u_bis   = exact_sol_k(x, 2)
        
        f1 = -D * u_bis[1] + A * u_prime[1]
        f2 = -D * u_bis[2] + A * u_prime[2]
        f3 = -D * u_bis[3] + A * u_prime[3]
        return [f1, f2, f3]
    end

    return exact_sol, exact_sol_prime, exact_sol_k, Rhs
end


"""
    testcase_star_trig(D_val::Real, A_val::Real, J::Integer)

Generates the trigonometric star graph test case in case format.

# Arguments
- `D_val::Real`: Diffusion coefficient.
- `A_val::Real`: Advection coefficient.
- `J::Integer`: Refinement level.

# Output
- A named tuple containing the test case configuration.
"""
function testcase_star_trig(D_val::Real, A_val::Real, J::Integer)
    L = 1.0
    u_ex_vec, _, u_exakt_k, f_ex_vec = solve_star_trig(D_val, A_val)

    edges = [Edge(1,4), Edge(4,2), Edge(4,3)]
    nv = 4
    
    N_elements = 2^J
    n_e = Dict(e => Int(N_elements) for e in edges)
    
    eps_edge = Dict(e => float(D_val) for e in edges)
    a_edge   = Dict(e => float(A_val) for e in edges)
    
    f_edge = Dict(
        edges[1] => x -> f_ex_vec(x)[1], 
        edges[2] => x -> f_ex_vec(x)[2], 
        edges[3] => x -> f_ex_vec(x)[3]
    )

    edge_x = Dict{typeof(edges[1]), Vector{Float64}}()
    for e in edges
        edge_x[e] = collect(range(0.0, L; length=N_elements+1))
    end

    dirichlet = Dict{Int, Float64}(
        1 => u_ex_vec(0.0)[1], 
        2 => u_ex_vec(1.0)[2], 
        3 => u_ex_vec(1.0)[3]
    )

    return (
        name="testcase_star_trig",
        nv=nv, edges=edges, n_e=n_e, edge_x=edge_x,
        eps_edge=eps_edge, a_edge=a_edge, f_edge=f_edge,
        dirichlet=dirichlet,
        exakte_Loesung=u_ex_vec,
        exact_k_th_derivative=u_exakt_k
    )
end