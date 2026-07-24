"""
    get_macroscopic_sort(G::DiGraph, A_edge::AbstractVector{<:Float64})

Performs the macroscopic topological sorting of the graph vertices based on the 
advection flow direction. This only needs to be computed ONCE per graph and advection field,
regardless of the multigrid discretization level.

# Arguments
- `G::DiGraph`: The directed graph representing the topology.
- `A_edge::AbstractVector{<:Float64}`: A vector containing the advection coefficients for each edge.

# Output
- `Vector{Int}`: A topologically sorted array of vertex indices corresponding to the macroscopic flow direction. Returns `nothing` if the physical flow contains cycles.
"""
function get_macroscopic_sort(G::DiGraph, A_edge::AbstractVector{<:Float64})
    NV = nv(G)
    M = ne(G)

    flow_graph = DiGraph(NV)
    edge_list = collect(edges(G)) 

    for e in 1:M
        edge = edge_list[e]
        u = src(edge)
        v = dst(edge)
        
        if A_edge[e] >= 0 
            add_edge!(flow_graph, u, v)
        else 
            add_edge!(flow_graph, v, u)
        end
    end
    vertex_sweep_order = Int[]
    try
        vertex_sweep_order = topological_sort_by_dfs(flow_graph)
    catch
        @warn "Physical flow contains cycles! Perfect upwind sorting is impossible. Falling back to default ordering."
        return 
    end

    return vertex_sweep_order
end


"""
    build_level_permutation(G::DiGraph, vertex_sweep_order::Vector{Int}, int_nodes::AbstractVector{Int}, A_edge::AbstractVector{<:Float64})

Builds the global permutation vector `p` for a specific multigrid level. It uses the precomputed macroscopic vertex sorting and inserts the internal edge DOFs according to the current level's discretization.

# Arguments
- `G::DiGraph`: The directed graph.
- `vertex_sweep_order::Vector{Int}`: The precomputed topological sorting of the vertices.
- `int_nodes::AbstractVector{Int}`: A vector specifying the number of internal nodes on each edge.
- `A_edge::AbstractVector{<:Float64}`: The advection coefficients for each edge.

# Output
- `Vector{Int}`: The global permutation vector `p` used to reorder the system matrix for directional smoothing.
"""
function build_level_permutation(G::DiGraph, vertex_sweep_order::Vector{Int}, 
                                 int_nodes::AbstractVector{Int}, A_edge::AbstractVector{<:Float64})
    NV = nv(G)
    M = ne(G)
    NE = sum(int_nodes)

    edge_offsets = zeros(Int, M)
    curr_offset = 0
    for e in 1:M
        edge_offsets[e] = curr_offset
        curr_offset += int_nodes[e]
    end

    p = Int[]
    sizehint!(p, NE + NV)
    visited_edges = falses(M)
    
    edge_list = collect(edges(G)) 

    for v in vertex_sweep_order
        push!(p, NE + v)

        for e in 1:M
            if visited_edges[e]
                continue
            end
            
            edge = edge_list[e]
            
            if A_edge[e] >= 0 && src(edge) == v
                for k in 1:int_nodes[e]
                    push!(p, edge_offsets[e] + k)
                end
                visited_edges[e] = true
            elseif A_edge[e] < 0 && dst(edge) == v
                for k in int_nodes[e]:-1:1
                    push!(p, edge_offsets[e] + k)
                end
                visited_edges[e] = true
            end
        end
    end
    
    return p
end

"""
    smoother_DGS(HEE, HEV, HVE, HVV, f_E, f_V, u_E_old, u_V_old, p; omega=1.0, nu=1)

Applies `nu` steps of Directional Gauss-Seidel smoothing.

# Arguments
- `HEE`, `HEV`, `HVE`, `HVV`: The submatrices representing the edge-edge, edge-vertex, vertex-edge, and vertex-vertex couplings of the system.
- `f_E`, `f_V`: The right-hand side vectors for the edges and vertices.
- `u_E_old`, `u_V_old`: The current approximate solution vectors for the edges and vertices.
- `p::Vector{Int}`: The permutation vector indicating the directional sweeping order.
- `omega::Float64`: Relaxation factor (default is 1.0 for standard Gauss-Seidel).
- `nu::Int`: The number of smoothing steps to apply (default is 1).

# Output
- `Tuple{Vector, Vector}`: Returns a tuple `(u_E_new, u_V_new)` containing the smoothed solution approximations.
"""
function smoother_DGS(HEE, HEV, HVE, HVV, f_E, f_V, u_E_old, u_V_old, p; omega=1.0, nu=1)
    NE = size(HEE, 1)
    
    H = [HEE HEV; HVE HVV]
    f = [f_E; f_V]
    u = [u_E_old; u_V_old]

    H_p = H[p, p]
    f_p = f[p]
    u_p = u[p]

    D = Diagonal(H_p)
    L = -tril(H_p, -1)
    U = -triu(H_p, 1)

    L_matrix = LowerTriangular(D - L)

    for iter in 1:nu
        u_GS = L_matrix \ (U * u_p + f_p)
        u_p = (1.0 - omega) .* u_p .+ omega .* u_GS
    end

    inv_p = invperm(p)
    u_new = u_p[inv_p]

    return u_new[1:NE], u_new[NE+1:end]
end


"""
    smoother_GMRES(HEE, HEV, HVE, HVV, ue, uv, fe, fv, nu)

GMRES smoother for the asymmetric advection-diffusion problem.

# Arguments
- `HEE`, `HEV`, `HVE`, `HVV::AbstractArray{<:Real}`: The block submatrices of the discrete operator.
- `ue`, `uv::AbstractVector{<:Real}`: The current approximate solution vectors for the edges and vertices.
- `fe`, `fv::AbstractVector{<:Real}`: The right-hand side vectors.
- `nu::Int`: The number of GMRES iterations to execute.

# Output
- `Tuple{Vector, Vector}`: Returns a tuple `(u_E_new, u_V_new)` containing the updated solution vectors.
"""
function smoother_GMRES(HEE::AbstractArray{<:Real}, HEV::AbstractArray{<:Real},
                        HVE::AbstractArray{<:Real}, HVV::AbstractArray{<:Real},
                        ue::AbstractVector{<:Real}, uv::AbstractVector{<:Real},
                        fe::AbstractVector{<:Real}, fv::AbstractVector{<:Real},
                        nu::Int)
    
    nu == 0 && return ue, uv

    Ne = length(ue)
    Nv = length(uv)
    N = Ne + Nv

    u = vcat(ue, uv)
    f = vcat(fe, fv)

    r0_e = fe - (HEE * ue + HEV * uv)
    r0_v = fv - (HVE * ue + HVV * uv)
    r0 = vcat(r0_e, r0_v)
    
    beta = norm(r0)
    if beta < 1e-14
        return ue, uv
    end

    V = zeros(N, nu + 1)
    H_mat = zeros(nu + 1, nu)
    
    V[:, 1] = r0 ./ beta

    actual_nu = nu
    for j = 1:nu
        vj_e = @view V[1:Ne, j]
        vj_v = @view V[Ne+1:end, j]
        w = vcat(HEE * vj_e + HEV * vj_v, 
                 HVE * vj_e + HVV * vj_v)

        for i = 1:j
            H_mat[i, j] = dot(@view(V[:, i]), w)
            w .-= H_mat[i, j] .* @view(V[:, i])
        end

        H_mat[j+1, j] = norm(w)
        
        if H_mat[j+1, j] < 1e-14
            actual_nu = j
            break
        end
        
        V[:, j+1] = w ./ H_mat[j+1, j]
    end

    e1 = zeros(actual_nu + 1)
    e1[1] = beta
    
    y = H_mat[1:actual_nu+1, 1:actual_nu] \ e1 
    
    u_new = u + V[:, 1:actual_nu] * y
    
    return u_new[1:Ne], u_new[Ne+1:end]
end


"""
    smoother_ILU(fe, fv, ue, uv, K_global, ilu_fact; nu=1)

Performs `nu` steps of ILU smoothing. Utilizes the cached global 
system matrix and factorization to avoid memory allocations.

# Arguments
- `fe`, `fv::AbstractVector{<:Real}`: The right-hand side vectors for the edges and vertices.
- `ue`, `uv::AbstractVector{<:Real}`: The current approximate solution vectors.
- `K_global::SparseMatrixCSC`: The assembled global stiffness matrix.
- `ilu_fact`: The precomputed ILU factorization object.
- `nu::Int`: The number of smoothing iterations (default is 1).

# Output
- `Tuple{Vector, Vector}`: Returns a tuple `(u_E_new, u_V_new)` containing the updated solution vectors.
"""
function smoother_ILU(fe::AbstractVector{<:Real}, fv::AbstractVector{<:Real}, 
                      ue::AbstractVector{<:Real}, uv::AbstractVector{<:Real}, 
                      K_global::SparseMatrixCSC, ilu_fact; nu::Int=1)
    
    NE = length(fe)
    
    u = vcat(ue, uv)
    f = vcat(fe, fv)
    
    r = similar(u)
    
    for _ in 1:nu
        mul!(r, K_global, u)
        r .= f .- r
        u .+= ilu_fact \ r
    end
    
    return u[1:NE], u[NE+1:end]
end