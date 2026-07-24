const INT32_DOF_THRESHOLD_DEFAULT = 2_000_000

"""
    estimate_dofs(G::AbstractGraph, Levels::AbstractVector{Int})

Estimates the total number of degrees of freedom (DoFs) by summing the internal edge nodes and graph vertices.

# Arguments
- `G::AbstractGraph`: The graph representing the problem domain.
- `Levels::AbstractVector{Int}`: A vector containing the level of discretization for each edge.

# Output
- Returns the estimated total number of degrees of freedom as an integer.
"""
function estimate_dofs(G::AbstractGraph, Levels::AbstractVector{Int})
    return nv(G) + sum(2 .^ Levels .- 1)
end


"""
    choose_sparse_index_type(estimated_dofs::Integer; int32_threshold::Integer=INT32_DOF_THRESHOLD_DEFAULT)

Selects the optimal sparse index storage type based on the estimated number of DoFs.
If the number of DoFs exceeds the `int32_threshold` but is still representable by `Int32`, it returns `Int32` to reduce memory usage for sparse matrices.

# Arguments
- `estimated_dofs::Integer`: The estimated total number of degrees of freedom.
- `int32_threshold::Integer`: The threshold above which `Int32` is considered (default is `INT32_DOF_THRESHOLD_DEFAULT`).

# Output
- Returns the appropriate integer type (`Int32` or `Int`) for sparse matrix indexing.
"""
function choose_sparse_index_type(estimated_dofs::Integer;
    int32_threshold::Integer=INT32_DOF_THRESHOLD_DEFAULT)

    if estimated_dofs > int32_threshold && estimated_dofs <= typemax(Int32)
        return Int32
    end
    return Int
end


"""
    createH(G::Graph, Levels::AbstractVector{Int}, edge_length::AbstractVector{<:Float64}, potential::Float64; index_type::Type{<:Integer}=Int)

Assembles the block matrices diffusion problem using a topology-based approach with incidence matrices.

# Arguments
- `G::Graph`: The graph representing the domain.
- `Levels::AbstractVector{Int}`: A vector containing the discretization level on each edge.
- `edge_length::AbstractVector{<:Float64}`: A vector specifying the physical length of each edge.
- `potential::Float64`: The potential value used in the problem formulation.
- `index_type::Type{<:Integer}`: The integer type for sparse indices (default is `Int`).

# Output
- `HEE`: The submatrix corresponding to edge-edge interactions.
- `HEV`: The submatrix corresponding to edge-vertex interactions.
- `HVV`: The submatrix corresponding to vertex-vertex interactions.
"""
function createH(G::Graph, Levels::AbstractVector{Int},
    edge_length::AbstractVector{<:Float64}, potential::Float64;
    index_type::Type{<:Integer}=Int)

    int_vert = 2 .^ Levels .- 1

    Ebar = E_bar(int_vert; index_type=index_type)
    Ehat = E_hat(G, int_vert; index_type=index_type)

    W = ones(sum(int_vert .+ 1))
    lgt = 1
    for j = 1:ne(G)
        W[lgt:lgt+int_vert[j]] *= edge_length[j] / (int_vert[j] + 1)
        lgt += int_vert[j] + 1
    end
    WM = Diagonal(W)
    WS = Diagonal(1 ./ W)

    SEE = Ebar * WS * sparse(transpose(Ebar))
    
    MEEoD = abs.(Ebar) * WM * sparse(transpose(abs.(Ebar)))
    
    MEE = MEEoD + spdiagm(diag(MEEoD))
    HEE = SEE + potential / 6 * MEE
    SEV = Ebar * WS * sparse(transpose(Ehat))
    
    MEV = abs.(Ebar) * WM * sparse(transpose(abs.(Ehat)))
    HEV = SEV + potential / 6 * MEV    
    SVV = Ehat * WS * sparse(transpose(Ehat))
    
    MVVoD = abs.(Ehat) * WM * sparse(transpose(abs.(Ehat)))
    MVV = MVVoD + spdiagm(diag(MVVoD))
    HVV = SVV + potential / 6 * MVV

    return HEE, HEV, HVV
end


"""
    E_bar(int_vert::AbstractVector{Int}; index_type::Type{<:Integer}=Int)

Creates the matrix E_bar for the incidence structure of internal vertices. The necessary non-zero values are assigned directly using the COO format.

# Arguments
- `int_vert::AbstractVector{Int}`: A vector containing the number of internal vertices on each edge.
- `index_type::Type{<:Integer}`: The integer type for sparse indices (default is `Int`).

# Output
- Returns the assembled matrix E_bar as a sparse matrix.
"""
function E_bar(int_vert::AbstractVector{Int}; index_type::Type{<:Integer}=Int)
    M = length(int_vert)

    row_vec = index_type.(kron(collect(1:sum(int_vert)), [1, 1]))
    val_vec = kron(ones(sum(int_vert)), [-1, 1])

    col_vec = zeros(index_type, length(row_vec))
    lgt = 1
    plusend = zero(index_type)
    for j = 1:M
        if lgt - 1 == 0
            plusend = zero(index_type)
        else
            plusend = col_vec[lgt-1]
        end
        col_vec[lgt:lgt+int_vert[j]*2-1] = (
            kron(collect(1:int_vert[j]+1), [1, 1])[2:end-1] .+ plusend)
        lgt += int_vert[j] * 2
    end

    Ebar = sparse(row_vec, col_vec, val_vec)
    return Ebar
end


"""
    E_bar(int_vert::Int, M::Int)

Creates the matrix `E_bar` when every edge is discretized with the exact same number of internal vertices.
This function optimizes the creation by generating one diagonal block and duplicating it for every edge in a block-diagonal arrangement.

# Arguments
- `int_vert::Int`: The number of internal vertices on every edge.
- `M::Int`: The total amount of edges in the graph.

# Output
- Returns the assembled matrix E_bar as a sparse matrix.
"""
function E_bar(int_vert::Int, M)
    row_vec = kron(collect(1:int_vert), [1, 1])
    val_vec = kron(ones(int_vert), [-1, 1])

    col_vec = kron(collect(1:int_vert+1), [1, 1])[2:end-1]
    Ebar = kron(I(M), sparse(row_vec, col_vec, val_vec))
    return Ebar
end


"""
    E_hat(G::AbstractGraph, int_vert::AbstractVector{Int}; index_type::Type{<:Integer}=Int)

Creates the matrix E_hat for incidence of vertices. The function assigns an orientation to all edges to meet the requirements of metric graphs, following the default orientation structure provided by the Graphs package.

# Arguments
- `G::AbstractGraph`: The graph representing the structure.
- `int_vert::AbstractVector{Int}`: A vector containing the number of internal vertices on each edge.
- `index_type::Type{<:Integer}`: The integer type for sparse indices (default is `Int`).

# Output
- Returns the assembled matrix E_hat as a sparse matrix.
"""
function E_hat(G::AbstractGraph, int_vert::AbstractVector{Int}; index_type::Type{<:Integer}=Int)
    NV = nv(G)
    M = ne(G)

    nnz_total = 2 * M
    
    I_vec = Vector{index_type}(undef, nnz_total)
    J_vec = Vector{index_type}(undef, nnz_total)
    V_vec = Vector{Float64}(undef, nnz_total)
    
    col_offset = 0
    idx = 1
    
    @inbounds for (m, edge) in enumerate(edges(G))
        u = src(edge)
        v = dst(edge)
        N_loc = int_vert[m]
        
        col_start = col_offset + 1
        col_end   = col_offset + N_loc + 1
        
        I_vec[idx]   = u
        J_vec[idx]   = col_start
        V_vec[idx]   = 1.0
        
        I_vec[idx+1] = v
        J_vec[idx+1] = col_end
        V_vec[idx+1] = -1.0
        
        col_offset += N_loc + 1
        idx += 2
    end
    
    return sparse(I_vec, J_vec, V_vec, NV, col_offset)
end


