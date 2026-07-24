"""
    righthandside(f_exakt::Function, G::AbstractGraph, Levels::AbstractVector{Int}, edge_length::AbstractVector{<:Float64}, Potential::Float64; index_type::Type{<:Integer}=Int)

Creates the right-hand-side vectors for the system of equations.

The setup of the vectors follows the theoretical description for creating the right-hand side 
from a continuous formulation. For arbitrary continuous functions f: R -> R, the required 
integrals are approximated using Simpson's rule. As quadrature rules can only be applied 
to continuous functions, we require f to be continuous.

This function is designed to work in the general case of a non-equilateral graph and 
varying discretizations on every edge (i.e., the case in which the element size h_e is 
assigned a different value for all edges e in E).

# Arguments
- `f_exakt::Function`: Edgewise definition of the exact right-hand side function. Requires 3 arguments:
    1. `x`: Point at which the function needs to be evaluated on one edge.
    2. `i`: Index of the current edge.
    3. `Potential`: Value of the potential (for when the right-hand side depends on it).
- `G::AbstractGraph`: The graph topology.
- `Levels::AbstractVector{Int}`: Vector of discretization levels for each edge.
- `edge_length::AbstractVector{<:Float64}`: Vector of lengths for each edge.
- `Potential::Float64`: Potential value from the problem formulation.
- `index_type::Type{<:Integer}`: Integer type for sparse matrix indices (default: `Int`).

# Output:
- `Tuple{Vector{Float64}, Vector{Float64}}`: The right-hand-side vectors `(f_hat_edge, f_hat_vertex)`.
"""
function righthandside(f_exakt::Function, G::AbstractGraph,
    Levels::AbstractVector{Int}, edge_length::AbstractVector{<:Float64},
    Potential::Float64; index_type::Type{<:Integer}=Int)

    int_nodes = 2 .^ Levels .- 1
    N = nv(G)
    M = ne(G)

    W = transpose(edge_length ./ (int_nodes .+ 1))

    # Find values of the exact right hand side f(x) at all points of evaluation 
    f_tilde = ones(2 * sum(int_nodes) + 3 * M)
    lgt = 1
    for i = 1:M
        f_tilde[lgt:lgt+2*int_nodes[i]+2] = (
            f_exakt.(0:0.5*W[i]:edge_length[i], i, Potential))
        lgt += 2 * int_nodes[i] + 3
    end

    # create weight matrix
    W = ones(2 * sum(int_nodes) + 3 * M)
    lgt = 1
    for j = 1:ne(G)
        W[lgt:lgt+2*int_nodes[j]+2] *= edge_length[j] / (int_nodes[j] + 1)
        lgt += 2 * int_nodes[j] + 3
    end
    W = Diagonal(W)

    # create the matrix bar{F} from section 2.4

    row_vec = index_type.(kron(collect(1:sum(int_nodes)), [1, 1, 1]))
    val_vec = ones(length(row_vec))

    col_vec = zeros(index_type, length(val_vec))
    lgt = 1
    plusend = zero(index_type)
    for j = 1:M
        if lgt == 1
            plusend = zero(index_type)
        else
            plusend = col_vec[Int(lgt - 1)] + one(index_type)
        end
        col_vec[Int(lgt):Int(lgt + (2^(Levels[j] + 2) - 4) * 0.75 - 1)] = (
            index_type.(deleteat!(kron(collect(2:2^Levels[j]*2), [1, 1])[2:end-1],
                2:4:2^Levels[j]*4-4)) .+ plusend)
        lgt += (2^(Levels[j] + 2) - 4) * 0.75
    end

    F_bar = sparse(row_vec, col_vec, val_vec,
        sum(int_nodes), sum(2 .^ (Levels .+ 1) .+ 1))

    # create the matrix hat{F} from section 2.4
    E = -sparse(incidence_matrix(G, oriented=true))
    row_vec, col_vec, _ = findnz(E)

    row_vec = index_type.(kron(row_vec, [1, 1]))
    for j = 1:M
        col_vec[2j:end] .+= 2 * (int_nodes[j] + 1)
    end
    col_vec = index_type.(kron(col_vec, [1, 1]))
    col_vec .+= repeat(index_type[0, 1, -1, 0], M)
    val_vec = repeat([1.0, 2.0, 2.0, 1.0], M)
    F_hat = sparse(row_vec, col_vec, val_vec)


    f_hat_edge = F_bar * W / 3 * f_tilde
    f_hat_vertex = F_hat * W / 6 * f_tilde
    
    return f_hat_edge, f_hat_vertex
end