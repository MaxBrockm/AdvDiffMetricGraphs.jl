"""
    coth_safe(x)

Computes the hyperbolic cotangent of `x` safely to prevent overflow for large values.

# Arguments
- `x::Real`: The input value.

# Output
- Returns the hyperbolic cotangent of `x`, or `sign(x) * 1.0` if `abs(x) > 40.0`.
"""
@inline function coth_safe(x)
    if abs(x) > 40.0
        return sign(x) * 1.0
    else
        return coth(x)
    end
end


"""
    tau_supg(A::Float64, D::Float64, h::Float64)

Calculates the Streamline Upwind Petrov-Galerkin (SUPG) stabilization parameter `tau`.
Provides a stable diffusion-dominated limit.

# Arguments
- `A::Float64`: The advection coefficient.
- `D::Float64`: The diffusion coefficient.
- `h::Float64`: The local element size (mesh size).

# Output
- Returns the stabilization parameter `tau` as a `Float64`.
"""
@inline function tau_supg(A::Float64, D::Float64, h::Float64)
    a = abs(A)
    
    if a < 1e-14
        return 0.0
    end
    if D <= 0
        return h/(2a)
    end
    
    Pe = a*h/(2D)
    
    if Pe < 1e-6
        return h*h/(12D)
    elseif Pe > 40.0
        return (h/(2a)) * (1.0 - 1.0/Pe)
    else
        return (h/(2a)) * (coth_safe(Pe) - 1.0/Pe)
    end
end


"""
    local_stiff(D::Float64, A::Float64, h::Float64, tau::Float64)

Computes the local stiffness matrix for the advection-diffusion equation using the integration by parts form. The SUPG stabilization adds an artificial diffusion term: `tau * A^2`.

# Arguments
- `D::Float64`: The diffusion coefficient.
- `A::Float64`: The advection coefficient.
- `h::Float64`: The local element size.
- `tau::Float64`: The SUPG stabilization parameter.

# Output
- Returns a 2x2 local stiffness matrix of type `Matrix{Float64}`.
"""
function local_stiff(D::Float64, A::Float64, h::Float64, tau::Float64)
    D_eff = D + tau * (A^2)
    return (D_eff/h) * [1.0 -1.0; -1.0  1.0] + (A/2.0) * [ 1.0  1.0; -1.0 -1.0]
end


"""
    local_load(f::Function, x1::Float64, x2::Float64, A::Float64, tau::Float64)

Computes the right-hand side combining the standard Galerkin formulation and the SUPG stabilization part.

# Arguments
- `f::Function`: The right-hand side source function.
- `x1::Float64`: The left coordinate of the element.
- `x2::Float64`: The right coordinate of the element.
- `A::Float64`: The advection coefficient.
- `tau::Float64`: The SUPG stabilization parameter.

# Output
- Returns a tuple `(b1, b2)` of type `Tuple{Float64, Float64}` representing the RHS vector entries.
"""
function local_load(f::Function, x1::Float64, x2::Float64, A::Float64, tau::Float64)
    h = x2 - x1
    φ1(x) = (x2 - x) / h
    φ2(x) = (x - x1) / h
    
    # Standard Galerkin part
    b1, _ = quadgk(x -> f(x) * φ1(x), x1, x2; rtol=1e-8, atol=1e-10)
    b2, _ = quadgk(x -> f(x) * φ2(x), x1, x2; rtol=1e-8, atol=1e-10)
    
    # SUPG stabilization part
    if tau > 0.0
        b1_supg, _ = quadgk(x -> f(x) * (-1.0/h), x1, x2; rtol=1e-8, atol=1e-10)
        b2_supg, _ = quadgk(x -> f(x) * ( 1.0/h), x1, x2; rtol=1e-8, atol=1e-10)
        
        b1 += tau * A * b1_supg
        b2 += tau * A * b2_supg
    end
    return (float(b1), float(b2))
end


"""
    apply_dirichlet_blocks!(HEE, HEV, HVE, HVV, fe, fv, dirichlet_nodes, dirichlet_values)

Modifies the block matrices and right-hand side vectors in-place on the finest grid 
to enforce hard Dirichlet boundary conditions at the graph vertices.

# Arguments
- `HEE`: The edge-edge block sparse matrix.
- `HEV`: The edge-vertex block sparse matrix.
- `HVE`: The vertex-edge block sparse matrix.
- `HVV`: The vertex-vertex block sparse matrix.
- `fe`: The right-hand side vector for the edges.
- `fv`: The right-hand side vector for the vertices.
- `dirichlet_nodes::Vector{Int}`: Indices of the vertices where Dirichlet conditions are applied.
- `dirichlet_values::Vector{Float64}`: The corresponding prescribed Dirichlet values.

# Output
- Returns the modified tuple `(HEE, HEV, HVE, HVV, fe, fv)` after dropping structural zeros.
"""
function apply_dirichlet_blocks!(HEE, HEV, HVE, HVV, fe, fv, 
                                 dirichlet_nodes::Vector{Int}, 
                                 dirichlet_values::Vector{Float64})
    
    for (i, v) in enumerate(dirichlet_nodes)
        val = dirichlet_values[i]
        
        fe .-= HEV[:, v] .* val
        fv .-= HVV[:, v] .* val
        
        HEV[:, v] .= 0.0
        HVV[:, v] .= 0.0
        
        HVE[v, :] .= 0.0
        HVV[v, :] .= 0.0
        
        HVV[v, v] = 1.0
        fv[v] = val
    end

    dropzeros!(HEV)
    dropzeros!(HVE)
    dropzeros!(HVV)
    
    return HEE, HEV, HVE, HVV, fe, fv
end


"""
    check_inflow_vertices(G::DiGraph, A_edge::AbstractVector{<:Float64})

Checks all vertices of the graph for pure inflow behaviour based on the advection field.

# Arguments
- `G::DiGraph`: The directed graph representing the topology.
- `A_edge::AbstractVector{<:Float64}`: A vector containing the advection coefficients for each edge.

# Output
- Returns a `Vector{Int}` containing the indices of the vertices identified as pure inflow nodes.
"""
function check_inflow_vertices(G::DiGraph, A_edge::AbstractVector{<:Float64})
    NV = nv(G)
    edges_ordered = collect(edges(G))
    problem_nodes = Int[]
    
    for v in 1:NV
        sum_inflow = 0.0
        sum_abs = 0.0

        for (m, e) in enumerate(edges_ordered)
            if src(e) == v || dst(e) == v
                n_e_v = (v == dst(e)) ? 1.0 : -1.0
                
                sum_inflow += A_edge[m] * n_e_v
                sum_abs += abs(A_edge[m])
            end
        end

        if sum_abs > 1e-14
            ratio = sum_inflow / sum_abs
            
            if isapprox(ratio, 1.0; atol=1e-10)
                push!(problem_nodes, v)
            end
        end
    end
    
    if isempty(problem_nodes)
        @info "No pure inflow nodes detected."
    else
        @warn "$(length(problem_nodes)) pure inflow nodes detected! Without Dirichlet conditions at these nodes, the system may diverge."
    end
    
    return problem_nodes
end


"""
    createH_AdvectionDiffusion(G::DiGraph, Levels::AbstractVector{Int}, edge_length::AbstractVector{<:Float64}, D_edge::AbstractVector{<:Float64}, A_edge::AbstractVector{<:Float64}, f_exakt::Function; use_SUPG::Bool=true, index_type::Type{<:Integer}=Int)

Assembles the block matrices and right-hand side vectors for the advection-diffusion problem 
using a topology-based approach with incidence matrices. Includes the structure `E * W * abs(E)'` 
for the non-symmetric advection part.

# Arguments
- `G::DiGraph`: The directed graph.
- `Levels::AbstractVector{Int}`: Discretization levels for the edges.
- `edge_length::AbstractVector{<:Float64}`: Physical lengths of the edges.
- `D_edge::AbstractVector{<:Float64}`: Diffusion coefficients for the edges.
- `A_edge::AbstractVector{<:Float64}`: Advection coefficients for the edges.
- `f_exakt::Function`: The right-hand side source function.
- `use_SUPG::Bool`: Flag to enable or disable SUPG stabilization (default: `true`).
- `index_type::Type{<:Integer}`: The integer type for sparse matrix indices (default: `Int`).

# Output
- Returns a tuple `(HEE, HEV, HVE, HVV, f_E, f_V)` containing the four block matrices and the two right-hand side vectors for edges and vertices.
"""
function createH_AdvectionDiffusion(G::DiGraph, Levels::AbstractVector{Int},
                                    edge_length::AbstractVector{<:Float64},
                                    D_edge::AbstractVector{<:Float64},
                                    A_edge::AbstractVector{<:Float64},
                                    f_exakt::Function;
                                    use_SUPG::Bool=true,
                                    index_type::Type{<:Integer}=Int)
    int_vert = 2 .^ Levels .- 1
    M = ne(G)

    Ebar = E_bar(int_vert, index_type=index_type)
    Ehat = E_hat(G, int_vert, index_type=index_type)
    

    diff_coeffs = Float64[]
    adv_coeffs = Float64[]

    @inbounds for j = 1:M
        h = edge_length[j] / (int_vert[j] + 1)
        D = float(D_edge[j])
        A = float(A_edge[j])
        
        tau = use_SUPG ? tau_supg(A, D, h) : 0.0
        
        D_eff = D + tau * A^2
        
        for _ in 1:(int_vert[j] + 1)
            push!(diff_coeffs, D_eff / h)
            push!(adv_coeffs, A / 2.0)
        end
    end

    WD = Diagonal(diff_coeffs)
    WA = Diagonal(adv_coeffs)

    
    SEE = Ebar * WD * sparse(transpose(Ebar))
    AEE = -Ebar * WA * sparse(transpose(abs.(Ebar)))
    HEE = SEE - AEE

    SEV = Ebar * WD * sparse(transpose(Ehat))
    AEV = -Ebar * WA * sparse(transpose(abs.(Ehat)))
    HEV = SEV - AEV

    SVE = Ehat * WD * sparse(transpose(Ebar))
    AVE = -Ehat * WA * sparse(transpose(abs.(Ebar)))
    HVE = SVE - AVE

    SVV = Ehat * WD * sparse(transpose(Ehat))
    AVV = -Ehat * WA * sparse(transpose(abs.(Ehat)))
    HVV = SVV - AVV

    int_nodes = max.(2 .^ Levels .- 1, 0)
    NE = sum(int_nodes)
    NV = nv(G)

    edge_offsets = zeros(Int, M)
    curr_offset = 0
    @inbounds for e in 1:M
        edge_offsets[e] = curr_offset
        curr_offset += int_nodes[e]
    end

    f_E = zeros(NE)
    f_V = zeros(NV)

    @inbounds for (m, edge) in enumerate(edges(G))
        u_node = src(edge)
        v_node = dst(edge)

        D = float(D_edge[m])
        A = float(A_edge[m])
        L = float(edge_length[m])
        N_loc = int_nodes[m]
        h = L / (N_loc + 1)

        tau = use_SUPG ? tau_supg(A, D, h) : 0.0
        offset = edge_offsets[m]
        
        f_func(x) = f_exakt(x, m)
        
        if N_loc == 0
            b1, b2 = local_load(f_func, 0.0, h, A, tau)
            f_V[u_node] += b1
            f_V[v_node] += b2
        else
            b1, b2 = local_load(f_func, 0.0, h, A, tau)
            n_edge1 = offset + 1
            f_V[u_node] += b1
            f_E[n_edge1] += b2
            for k in 2:N_loc
                x1 = (k-1) * h
                x2 = k * h
                b1, b2 = local_load(f_func, x1, x2, A, tau)

                n1 = offset + k - 1
                n2 = offset + k

                f_E[n1] += b1
                f_E[n2] += b2
            end

            x1 = N_loc * h
            x2 = (N_loc + 1) * h
            b1, b2 = local_load(f_func, x1, x2, A, tau)
            n_edgeL = offset + N_loc
            f_E[n_edgeL] += b1
            f_V[v_node]  += b2
        end
    end
    return HEE, HEV, HVE, HVV, f_E, f_V
end