    """
    normal_sign(e, v::Int)

Determines the orientation sign (outward normal in 1D parameterization) 
for a given edge `e` and vertex `v`.

# Arguments
- `e`: The edge object.
- `v::Int`: The vertex index.

# Output:
- `Float64`: `-1.0` at the source vertex, `+1.0` at the destination vertex.
"""
@inline function normal_sign(e, v::Int)
    v == src(e) ? -1.0 :
    v == dst(e) ? +1.0 :
    error("Vertex $v not endpoint of edge $e")
end
   

"""
    fix_inflow_nodes!(G::DiGraph, problem_nodes::Vector{Int},
                      Levels::Vector{Int}, edge_length::Vector{Float64},
                      D_edge::Vector{Float64}, A_edge::Vector{Float64},
                      dirichlet_nodes::Vector{Int}, dirichlet_values::Vector{Float64};
                      L_new::Real = 1.0, D_new::Real = 0.01, A_new::Real = 1.0,
                      Level_new::Int = 3, val_new::Real = 0.0, verbose::Bool = false)

Adds a new vertex and an outgoing edge for each specified inflow problem node to resolve flow congestion. 
The new vertex is automatically assigned a Dirichlet boundary condition.

### Arguments
- `G::DiGraph`: The directed graph representing the network topology.
- `problem_nodes::Vector{Int}`: Indices of vertices flagged as inflow problem nodes.
- `Levels::Vector{Int}`: Vector tracking the refinement levels of each edge.
- `edge_length::Vector{Float64}`: Vector of edge lengths.
- `D_edge::Vector{Float64}`: Diffusion coefficients per edge.
- `A_edge::Vector{Float64}`: Advection coefficients per edge.
- `dirichlet_nodes::Vector{Int}`: Global indices where Dirichlet boundary conditions are applied.
- `dirichlet_values::Vector{Float64}`: Corresponding Dirichlet boundary values.

### Keyword Arguments
- `L_new::Real = 1.0`: Length of the newly added edges.
- `D_new::Real = 0.01`: Diffusion coefficient for the new edges.
- `A_new::Real = 1.0`: Advection coefficient for the new edges.
- `Level_new::Int = 3`: Refinement level for the new edges.
- `val_new::Real = 0.0`: Dirichlet value assigned to the new boundary vertex.
- `verbose::Bool = false`: Flag to enable detailed logging during the fix process.

### Output
- `G::DiGraph`: The updated directed graph including the newly added nodes and edges. (Note: Input vectors are updated in place).
"""
function fix_inflow_nodes!(G::DiGraph, problem_nodes::Vector{Int},
                        Levels::Vector{Int}, edge_length::Vector{Float64},
                        D_edge::Vector{Float64}, A_edge::Vector{Float64},
                        dirichlet_nodes::Vector{Int}, dirichlet_values::Vector{Float64};
                        L_new::Real = 1.0, D_new::Real = 0.01, A_new::Real = 1.0,
                        Level_new::Int = 3, val_new::Real = 0.0, verbose::Bool = false)
    
    for p_node in problem_nodes
        p_node in dirichlet_nodes && continue  
        add_vertex!(G)
        new_v = nv(G)
        
        add_edge!(G, p_node, new_v)
        
        push!(Levels, Level_new)
        push!(edge_length, L_new)
        push!(D_edge, D_new)
        
        push!(A_edge, A_new) 
        
        push!(dirichlet_nodes, new_v)
        push!(dirichlet_values, val_new)
        
        @debug "Auto-Fix: Edge from vertex $p_node to new vertex $new_v added (Dirichlet value: $val_new)."
    end
    
    return nothing
end


"""
    run_MG_from_case(case; v1=3, v2=3, use_SUPG=true, use_additional_SUPG=false,
                     L_fix::Real=1.0, D_fix::Real=0.01, 
                     A_fix::Real=1.0, Level_fix::Int=3, 
                     val_fix::Real=0.0, autofix::Bool=true)

Takes a case definition (`NamedTuple`), unpacks the data, checks for inflow nodes (auto-fix), 
assembles the block matrices, applies Dirichlet boundary conditions, sets up the multigrid hierarchy, 
and executes the multigrid solver loop.
The matrix structure is automatically determined based on the number of edges and vertices in the graph. For too large systems, we change to `Int32` for sparse matrix indices to avoid overflow.
Saves the convergence rate plot as a PDF in the current working directory.
### Arguments
- `case::NamedTuple`: A structured collection containing network parameters, exact solutions, and domain configuration.

### Keyword Arguments
- `v1::Int = 3`: Number of pre-smoothing iterations.
- `v2::Int = 3`: Number of post-smoothing iterations.
- `use_SUPG::Bool = true`: Flag to enable Streamline Upwind Petrov-Galerkin (SUPG) stabilization.
- `use_additional_SUPG::Bool = false`: Flag to enable extra vertex stabilization terms.
- `L_fix::Real = 1.0`: Default length for fixed edges.
- `D_fix::Real = 0.01`: Default diffusion parameter.
- `A_fix::Real = 1.0`: Default advection parameter.
- `Level_fix::Int = 3`: Default refinement level.
- `val_fix::Real = 0.0`: Default boundary value.
- `autofix::Bool = true`: Automatically handle inflow congestion nodes.

### Output
- `sol::Vector{Float64}`: The computed numerical solution vector over the graph degrees of freedom.
"""
function run_MG_from_case(case; v1=3, v2=3, use_SUPG=true,use_additional_SUPG=false,
                        L_fix::Real=1.0, D_fix::Real=0.01, 
                        A_fix::Real=1.0, Level_fix::Int=3, 
                        val_fix::Real=0.0, autofix::Bool=true)
                        
    @info "=== Starting solver for case: $(case.name) ==="

    M_orig = length(case.edges)
    NV_orig = case.nv
    
    Levels = zeros(Int, M_orig)
    edge_length = zeros(Float64, M_orig)
    D_vec = zeros(Float64, M_orig)
    A_vec = zeros(Float64, M_orig)

    for (i, e) in enumerate(case.edges)
        Levels[i] = Int(log2(case.n_e[e])) 
        edge_length[i] = case.edge_x[e][end] - case.edge_x[e][1]
        D_vec[i] = case.eps_edge[e]
        A_vec[i] = case.a_edge[e]
    end

    G = SimpleDiGraph(NV_orig)
    for e in case.edges
        add_edge!(G, e)
    end

    d_nodes = collect(keys(case.dirichlet))
    d_vals = collect(values(case.dirichlet))

    problem_nodes = check_inflow_vertices(G, A_vec)
    if !isempty(problem_nodes) && autofix
        
        fix_inflow_nodes!(G, problem_nodes, Levels, edge_length, D_vec, A_vec, 
                        d_nodes, d_vals; 
                        L_new=L_fix, D_new=D_fix, A_new=A_fix, 
                        Level_new=Level_fix, val_new=val_fix)
    end
    
    M_current = ne(G)
    @info "After Auto-Fix: Graph has $M_current edges and $(nv(G)) vertices."

    estimated_dofs = estimate_dofs(G, Levels)
    sparse_index_type = choose_sparse_index_type(estimated_dofs)
    @info "Estimated DoFs: $estimated_dofs | sparse index type: $sparse_index_type"

    function f_exakt_wrapper(x, i)
        if i <= M_orig
            return case.f_edge[case.edges[i]](x)
        else
            return 0.0 # No source term on newly added edges
        end
    end

    HEE, HEV, HVE, HVV, fe, fv = createH_AdvectionDiffusion(G, Levels, edge_length, 
            D_vec, A_vec, f_exakt_wrapper; 
        use_SUPG=use_SUPG, index_type=sparse_index_type)

    if use_SUPG && use_additional_SUPG
        apply_vertex_stabilization_new!(HEE, HEV, HVE, HVV, fe, fv, G, Levels, edge_length, D_vec, A_vec, f_exakt_wrapper; dirichlet_nodes = d_nodes)
    end

    HEE, HEV, HVE, HVV, fe, fv = apply_dirichlet_blocks!(HEE, HEV, HVE, HVV, fe, fv, d_nodes, d_vals)
        
    Scale_HEV = Rest_HEV_setup(HEV; index_type=sparse_index_type) 

    ue = ones(size(fe))
    uv = ones(size(fv))
    for (n, val) in zip(d_nodes, d_vals)
        uv[n] = val
    end

    rate_e_vec = Float64[]
    rate_v_vec = Float64[]
    rate_vec   = Float64[]
    kmax = 0

    J_max = maximum(Levels)
    macroscopic_order = get_macroscopic_sort(G, A_vec)
    
    mg_hierarchy = setup_advection_hierarchy(HEE, HEV, HVE, HVV, Levels, J_max, G, A_vec, macroscopic_order; index_type=sparse_index_type)
    
    rate_e_vec = Float64[]
    rate_v_vec = Float64[]
    rate_vec   = Float64[]
    kmax = 0

    while ((kmax < 250) && (sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - HVE * ue - HVV * uv)^2)) > 1e-8)
        
        solve_advection_MG!(mg_hierarchy, J_max, fv, fe, uv, ue, v1, v2, 1)

        res_e = norm(fe - HEE * ue - HEV * uv)
        res_v = norm(fv - HVE * ue - HVV * uv)
        res_total = sqrt(res_e^2 + res_v^2)
        
        push!(rate_e_vec, res_e)
        push!(rate_v_vec, res_v)
        push!(rate_vec, res_total)
        
        kmax += 1
        print("\rZyklus $kmax | Current residual: $(round(res_total, sigdigits=4))\033[K")
    end
    
    println()
    @info "Solver finished after $kmax cycles."
    
    if kmax > 1
        gr() 
        Plot_Res = plot(2:1:kmax, 
            [rate_vec[2:end] ./ rate_vec[1:end-1], 
            rate_e_vec[2:end] ./ rate_e_vec[1:end-1], 
            rate_v_vec[2:end] ./ rate_v_vec[1:end-1]], 
            label=["Total conv. rate" "Rate on inner vertices" "Rate on vertices"], 
            xlabel="Cycle", ylabel="Convergence Rate", 
            legend=:bottomright, lw=3, tickfontsize=8, 
            linestyle=[:solid :dash :dash], dpi=300) 
        
        if !isdir(joinpath("Figures", "run_MG_from_case"))
            mkdir(joinpath("Figures", "run_MG_from_case"))
        end
        savefig(Plot_Res, joinpath("Figures", "run_MG_from_case", "MG_Convergence_$(case.name).pdf"))
    end

    return ue, uv, rate_vec, nv(G), ne(G)
end


"""
    check_vertex_flux_condition_MG(ue, uv, case; tol=1e-4, verbose=true)

Checks the Kirchhoff flux conservation condition at all physical vertices for the computed multigrid solution.

### Arguments
- `ue::Vector{Float64}`: Solution components corresponding to edge degrees of freedom.
- `uv::Vector{Float64}`: Solution components corresponding to vertex degrees of freedom.
- `case::NamedTuple`: The case definition providing graph topology and physical parameters.

### Keyword Arguments
- `tol::Float64 = 1e-4`: Tolerance threshold for flux balance validation.
- `verbose::Bool = true`: Enables detailed reporting of flux discrepancies per vertex.

### Output
- `max_residual::Float64`: The maximum residual found across all vertex flux conditions.
"""
function check_vertex_flux_condition_MG(ue, uv, case; tol=1e-4, verbose=true)
    vertices = 1:case.nv
    
    R = Dict{Int,Float64}(v => 0.0 for v in vertices)
    contrib = Dict{Int,Vector{NamedTuple}}(v => NamedTuple[] for v in vertices)
    
    dirichlet_nodes = collect(keys(case.dirichlet))
    
    ue_idx = 1
    for e in case.edges
        x_nodes = case.edge_x[e]
        
        h_src = x_nodes[2] - x_nodes[1]          
        h_dst = x_nodes[end] - x_nodes[end-1]   
        
        n_inner = length(x_nodes) - 2 

        u_src = uv[src(e)]
        u_dst = uv[dst(e)]
        
        if n_inner > 0
            u_int_first = ue[ue_idx]
            u_int_last  = ue[ue_idx + n_inner - 1]
            ue_idx += n_inner
        else
            u_int_first = u_dst
            u_int_last  = u_src
        end
        
        A = case.a_edge[e]
        D = case.eps_edge[e]
        
        v_src = src(e)
        du_src = (u_int_first - u_src) / h_src   
        nsgn_src = normal_sign(e, v_src)
        term_src = (-D * du_src + A * u_src) * nsgn_src
        R[v_src] += term_src
        push!(contrib[v_src], (edge=e, at=:src, n=nsgn_src, u=u_src, du=du_src, term=term_src))
        
        v_dst = dst(e)
        du_dst = (u_dst - u_int_last) / h_dst    
        nsgn_dst = normal_sign(e, v_dst)
        term_dst = (-D * du_dst + A * u_dst) * nsgn_dst
        R[v_dst] += term_dst
        push!(contrib[v_dst], (edge=e, at=:dst, n=nsgn_dst, u=u_dst, du=du_dst, term=term_dst))
    end
    
    check_vertices = [v for v in vertices if !(v in dirichlet_nodes)]
    maxabs = isempty(check_vertices) ? 0.0 : maximum(abs(R[v]) for v in check_vertices)
    passed = maxabs ≤ tol
    
    if verbose
        println("\n--- Vertex-Flux-Check (Multigrid) ---")
        println("Tolerance = $(tol)")
        println("Dirichlet nodes (ignored for pass/fail): ", dirichlet_nodes)
        
        for v in vertices
            flag = (v in dirichlet_nodes) ? " (Dirichlet)" : ""
            println("Vertex $v: R(v) = $(round(R[v], sigdigits=6))" * flag)
            for c in contrib[v]
                println("    Edge=$(c.edge) @$(c.at): n=$(c.n), u=$(round(c.u, sigdigits=5)), du=$(round(c.du, sigdigits=5)) -> flux=$(round(c.term, sigdigits=5))")
            end
        end
        println("Maximum |residual| (without Dirichlet) = $maxabs")
        println(passed ? "Kirchhoff condition satisfied!" : "Kirchhoff condition NOT satisfied (greater than tolerance)!")
        println("----------------------------------------\n")
    end
    
    return (R=R, maxabs=maxabs, passed=passed)
end


"""
    run_star_shishkin_block_backslash(case)

Assembles the block system (`HEE`, `HEV`, `HVE`, `HVV`) using Shishkin meshes, combines them into a global system, 
enforces Dirichlet boundary conditions, and solves the system directly using the Julia backslash operator (`\\`).

### Arguments
- `case::NamedTuple`: The problem configuration instance containing geometry, mesh type, and coefficients.

### Output
- `sol::Vector{Float64}`: The direct solution vector obtained via sparse LU/backslash factorization.
"""
function run_star_shishkin_block_backslash(case)

    NV = case.nv
    edges_ordered = case.edges
    
    n_e_vec = [case.n_e[e] for e in edges_ordered]
    
    edge_dofs, Ntot = build_mesh(NV; n_e_per_edge=n_e_vec, edges_ordered=edges_ordered)

    HEE, HEV, HVE, HVV, fe, fv = assemble_system_blocks_shishkin(
        NV, 
        edge_dofs, 
        case.edge_x, 
        case.eps_edge, 
        case.a_edge, 
        case.f_edge; 
        edges_ordered=edges_ordered
    )
    
    NE = size(HEE, 1)
    
    H_global = [HEE HEV; HVE HVV]
    f_global = [fe; fv]
    
    NE = size(H_global, 1) - case.nv
    NV = case.nv

    dir_nodes = sort(collect(keys(case.dirichlet)))
    dir_values = [case.dirichlet[n] for n in dir_nodes]
    
    dir_nodes_global = [NE + n for n in dir_nodes]
    
    for (k, val) in zip(dir_nodes_global, dir_values)
        col = copy(H_global[:, k])
        f_global .-= col .* float(val)
        H_global[k, :] .= 0.0
        H_global[:, k] .= 0.0
        H_global[k, k] = 1.0
        f_global[k] = float(val)
    end
    dropzeros!(H_global)

    u_global = H_global \ f_global
    
    ue = u_global[1:NE]
    uv = u_global[NE+1:end]
    
    println("Residual ||H*u-f|| = ", norm(H_global * u_global - f_global))
    return ue, uv
end


"""
    build_mesh(nv::Int; n_e_per_edge::AbstractVector{<:Integer}, edges_ordered=EDGES_ORDERED)

Constructs the mesh topology by mapping each edge to its corresponding global degree of freedom indices.

### Arguments
- `nv::Int`: Total number of vertices in the graph network.

### Keyword Arguments
- `n_e_per_edge::AbstractVector{<:Integer}`: Number of sub-intervals or elements per edge.
- `edges_ordered`: Ordered collection of edge tuples defining the network structure.

### Output
- `mesh_data::NamedTuple`: Indices mapping local edge elements to global degrees of freedom.
"""
function build_mesh(nv::Int; n_e_per_edge::AbstractVector{<:Integer}, edges_ordered=EDGES_ORDERED)
    node_count = nv
    edge_dofs = Dict{Graphs.SimpleGraphs.SimpleEdge{Int},Vector{Int}}()
    for (idx,e) in enumerate(edges_ordered)
        i, j = src(e), dst(e)
        nloc = Int(n_e_per_edge[idx])
        nloc ≥ 1 || error("n_e_per_edge must be ≥1 per edge")
        nodes = Int[i]
        for _ in 1:(nloc-1)
            node_count += 1
            push!(nodes, node_count)
        end
        push!(nodes, j)
        edge_dofs[e] = nodes
    end
    return edge_dofs, node_count
end


"""
    apply_vertex_stabilization_new!(HEE, HEV, HVE, HVV, f_E, f_V,
                                    G::DiGraph, Levels::AbstractVector{Int},
                                    edge_length::AbstractVector{<:Float64},
                                    D_edge::AbstractVector{<:Float64},
                                    A_edge::AbstractVector{<:Float64},
                                    f_exakt::Function;
                                    dirichlet_nodes = Int[])

Applies advanced vertex stabilization terms (SUPG-related) to the block matrices and right-hand side vectors.

### Arguments
- `HEE`: Edge-edge block matrix of the discrete system.
- `HEV`: Edge-vertex block matrix.
- `HVE`: Vertex-edge block matrix.
- `HVV`: Vertex-vertex block matrix.
- `f_E::Vector{Float64}`: Right-hand side vector component for edges.
- `f_V::Vector{Float64}`: Right-hand side vector component for vertices.
- `G::DiGraph`: Directed graph representation of the network.
- `Levels::AbstractVector{Int}`: Refinement levels associated with each edge.
- `edge_length::AbstractVector{<:Float64}`: Lengths of the individual edges.
- `D_edge::AbstractVector{<:Float64}`: Diffusion coefficients per edge.
- `A_edge::AbstractVector{<:Float64}`: Advection coefficients per edge.
- `f_exakt::Function`: The exact analytical solution function used for source/stabilization evaluations.

### Keyword Arguments
- `dirichlet_nodes::Vector{Int} = Int[]`: List of nodes with enforced Dirichlet conditions.

### Returns
- `::Nothing`: The input block matrices (`HEE`, `HEV`, `HVE`, `HVV`) and right-hand side vectors (`f_E`, `f_V`) are updated and modified in place.
"""
function apply_vertex_stabilization_new!(HEE, HEV, HVE, HVV, f_E, f_V,
                                    G::DiGraph, Levels::AbstractVector{Int},
                                    edge_length::AbstractVector{<:Float64},
                                    D_edge::AbstractVector{<:Float64},
                                    A_edge::AbstractVector{<:Float64},
                                    f_exakt::Function;
                                    dirichlet_nodes = Int[])
    NV = nv(G)
    M = ne(G)
    int_nodes = max.(2 .^ Levels .- 1, 0)
    n_e = int_nodes .+ 1 

    edge_offsets = zeros(Int, M)
    curr_offset = 0
    for e in 1:M
        edge_offsets[e] = curr_offset
        curr_offset += int_nodes[e]
    end

    index_type = HEE isa SparseMatrixCSC ? eltype(rowvals(HEE)) : Int

    I_EE = index_type[]; J_EE = index_type[]; V_EE = Float64[]
    I_EV = index_type[]; J_EV = index_type[]; V_EV = Float64[]
    I_VE = index_type[]; J_VE = index_type[]; V_VE = Float64[]
    I_VV = index_type[]; J_VV = index_type[]; V_VV = Float64[]

    @inline function push_update!(t1::Symbol, i1::Integer, t2::Symbol, i2::Integer, val::Float64)
        val == 0.0 && return nothing
        if t1 == :E && t2 == :E;     push!(I_EE, i1); push!(J_EE, i2); push!(V_EE, val)
        elseif t1 == :E && t2 == :V; push!(I_EV, i1); push!(J_EV, i2); push!(V_EV, val)
        elseif t1 == :V && t2 == :E; push!(I_VE, i1); push!(J_VE, i2); push!(V_VE, val)
        elseif t1 == :V && t2 == :V; push!(I_VV, i1); push!(J_VV, i2); push!(V_VV, val)
        end
        return nothing
    end

    function coth_minus_inv(pe::Float64)
        abs_pe = abs(pe)
        if abs_pe < 1e-7
            return pe / 3.0 - (pe^3) / 45.0
        elseif abs_pe > 350.0
            return sign(pe) * 1.0 - 1.0 / pe
        else
            return coth(pe) - 1.0 / pe
        end
    end

    function A_coth_Pe(A::Float64, pe::Float64, D::Float64, h::Float64)
        abs_pe = abs(pe)
        if abs_pe < 1e-7
            return 2.0 * D / h + A * pe / 3.0
        elseif abs_pe > 350.0
            return A * sign(pe)
        else
            return A * coth(pe)
        end
    end

    all_edges = collect(edges(G))

    for v in 1:NV
        if v in dirichlet_nodes
            continue
        end

        neighbours = Tuple{Symbol, Int}[]
        U_coeffs  = Float64[] 
        W_coeffs  = Float64[] 
        
        U_v_sum = 0.0
        W_v_sum = 0.0
        gamma_v = 0.0
        F_v     = 0.0

        for (m, e) in enumerate(all_edges)
            is_src = (src(e) == v)
            is_dst = (dst(e) == v)
            
            if !is_src && !is_dst
                continue
            end

            nOut = is_src ? -1.0 : 1.0
            
            if int_nodes[m] > 0
                n_idx = is_src ? (:E, edge_offsets[m] + 1) : (:E, edge_offsets[m] + int_nodes[m])
            else
                n_idx = is_src ? (:V, dst(e)) : (:V, src(e))
            end

            A = A_edge[m]; D = D_edge[m]; L = edge_length[m]; h = L / n_e[m]
            
            pe_e = A * h / (2.0 * D)

            beta_e  = (h / 2.0) * coth_minus_inv(pe_e) * nOut + (h / 2.0)
            gamma_e = -0.5 * A_coth_Pe(A, pe_e, D, h) + 0.5 * A * nOut
            alpha_e =  0.5 * A + 0.5 * A * coth_minus_inv(pe_e) * nOut

            C_un = -D/h - A * nOut * beta_e / h
            C_uv =  D/h + A * nOut + A * nOut * beta_e / h

            C_wn = -alpha_e + D/h
            C_wv =  alpha_e - D/h

            gamma_v += gamma_e
            
            x_v = is_src ? 0.0 : L
            F_v += f_exakt(x_v, m) * beta_e

            push!(neighbours, n_idx)
            push!(U_coeffs, C_un)
            push!(W_coeffs, C_wn)

            U_v_sum += C_uv
            W_v_sum += C_wv
        end

        push!(neighbours, (:V, v))
        push!(U_coeffs, U_v_sum)
        push!(W_coeffs, W_v_sum)
        
        if abs(gamma_v) > 1e-14
            for i in 1:length(neighbours)
                test_node = neighbours[i]
                W_i = W_coeffs[i]

                if abs(W_i) < 1e-12
                    W_i = 0.0
                end

                val_f = W_i * F_v / gamma_v
                if test_node[1] == :E
                    f_E[test_node[2]] += val_f
                else
                    f_V[test_node[2]] += val_f
                end

                for j in 1:length(neighbours)
                    trial_node = neighbours[j]
                    U_j = U_coeffs[j]

                    if abs(U_j) < 1e-12
                        U_j = 0.0
                    end

                    val_K = W_i * (-1.0 / gamma_v) * U_j
                    
                    push_update!(test_node[1], test_node[2], trial_node[1], trial_node[2], val_K)
                end
            end
        end
    end

    HEE .+= sparse(I_EE, J_EE, V_EE, size(HEE)...)
    HEV .+= sparse(I_EV, J_EV, V_EV, size(HEV)...)
    HVE .+= sparse(I_VE, J_VE, V_VE, size(HVE)...)
    HVV .+= sparse(I_VV, J_VV, V_VV, size(HVV)...)

    return HEE, HEV, HVE, HVV, f_E, f_V
end