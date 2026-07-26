"""
    MGLevel{MHEE, MHEV, MHVV, MSCALE, MRE, V}

Structure to store the operators and allocation-free workspace for a specific 
multigrid level.

$FIELDS
"""
struct MGLevel{
    MHEE <: AbstractMatrix{Float64},
    MHEV <: AbstractMatrix{Float64},
    MHVV <: AbstractMatrix{Float64},
    MSCALE <: AbstractMatrix{Float64},
    MRE <: AbstractMatrix{Float64},
    V <: AbstractVector{Float64}
}
    "System matrix block for internal edge nodes"
    HEE::MHEE
    "System matrix block coupling edges to vertices"
    HEV::MHEV
    "System matrix block for graph vertices"
    HVV::MHVV
    "Scale matrix for HEV restriction"
    Scale_HEV::MSCALE
    "Restriction operator for internal edge nodes"
    R_E::MRE
    
    "Precomputed diagonals for the Jacobi smoother (edges)"
    diag_EE::V
    "Precomputed diagonals for the Jacobi smoother (vertices)"
    diag_VV::V
    
    "Preallocated workspace: residual for edges"
    res_e::V
    "Preallocated workspace: residual for vertices"
    res_v::V
    "Preallocated workspace: error correction for edges"
    v_tilde_e::V
    "Preallocated workspace: error correction for vertices"
    v_tilde_v::V
    
    "Temporary array for smoother operations (edges)"
    tmp_e::V
    "Temporary array for smoother operations (vertices)"
    tmp_v::V
    "Next iteration state for edges"
    ue_next::V
    "Next iteration state for vertices"
    uv_next::V

    "Algebraic Multigrid (AMG) setup for the coarsest geometric level"
    amg_data::Any
end

"""
    setup_MG_hierarchy(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real}, 
                       HVV::AbstractMatrix{<:Real}, Levels::AbstractVector{Int}, 
                       J::Int, Scale_HEV::AbstractMatrix{<:Real};
                       index_type::Type{<:Integer}=Int)

Calculates the Galerkin projections and pre-allocates the workspace for all multigrid levels.

# Arguments
- `HEE::AbstractMatrix{<:Real}`: Fine-level system matrix block for internal edge nodes.
- `HEV::AbstractMatrix{<:Real}`: Fine-level system matrix block coupling edges to vertices.
- `HVV::AbstractMatrix{<:Real}`: Fine-level system matrix block for graph vertices.
- `Levels::AbstractVector{Int}`: Vector specifying the discretization level on each edge.
- `J::Int`: Maximum hierarchy depth.
- `Scale_HEV::AbstractMatrix{<:Real}`: Initial scale matrix for HEV.
- `index_type::Type{<:Integer}`: Integer type for sparse matrix indices (default: `Int`).

# Output:
- `Vector{MGLevel}`: The fully constructed multigrid hierarchy ready for the solve phase.
"""
function setup_MG_hierarchy(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real}, 
                            HVV::AbstractMatrix{<:Real}, Levels::AbstractVector{Int}, 
                            J::Int, Scale_HEV::AbstractMatrix{<:Real};
                            index_type::Type{<:Integer}=Int)
    
    hierarchy = Vector{MGLevel}(undef, J + 1)
    
    curr_HEE = HEE
    curr_HEV = HEV
    curr_HVV = HVV
    curr_Levels = copy(Levels)
    curr_Scale_HEV = Scale_HEV
    
    for j in J:-1:1
        R_E = Rest_int_nodes(curr_Levels; index_type=index_type)
        R_V = Rest_HEV(curr_Levels; index_type=index_type)
        
        dim_e = size(curr_HEE, 1)
        dim_v = size(curr_HVV, 1)
        
        hierarchy[j+1] = MGLevel(
            curr_HEE, curr_HEV, curr_HVV, curr_Scale_HEV, R_E,
            Vector(diag(curr_HEE)), Vector(diag(curr_HVV)), 
            zeros(dim_e), zeros(dim_v),                     
            zeros(dim_e), zeros(dim_v),                     
            zeros(dim_e), zeros(dim_v),                     
            zeros(dim_e), zeros(dim_v),
            nothing # No AMG data on fine levels
        )
        
        HEE_m1 = R_E * curr_HEE * R_E'
        HEV_m1 = R_E * curr_HEE * curr_Scale_HEV + R_E * curr_HEV
        HVV_m1 = curr_HVV + 2 * curr_Scale_HEV' * curr_HEV + curr_Scale_HEV' * curr_HEE * curr_Scale_HEV
        
        curr_Scale_HEV = R_V * curr_Scale_HEV
        curr_Levels = max.(curr_Levels .- 1, 0)
        curr_HEE = HEE_m1
        curr_HEV = HEV_m1
        curr_HVV = HVV_m1
        @info "Finished setting up level $j of the multigrid hierarchy."
    end
    
    dim_e = size(curr_HEE, 1)
    dim_v = size(curr_HVV, 1)
    empty_R_E = sparse(Vector{index_type}(undef, 0), Vector{index_type}(undef, 0), Float64[], 0, 0)
    
    A_coarse = [curr_HEE curr_HEV; curr_HEV' curr_HVV]
    
    Adj_coarse = Diagonal(diag(A_coarse)) - A_coarse
    Adj_sym = sparse(0.5 .* (Adj_coarse + Adj_coarse'))
    dropzeros!(Adj_sym)
    Gamma_coarse = SimpleGraph(Adj_sym)
    
    @info "Building AMG hierarchy for the coarsest geometric level (J=0)..."
    amg_levels, amg_A_c, amg_cs, amg_cs_fact = build_amg_hierarchy(
        A_coarse, Gamma_coarse; 
        max_coarse_size=50, 
        max_levels=10
    )
    
    amg_pack = (levels=amg_levels, A_coarsest=amg_A_c, cs=amg_cs, cs_fact=amg_cs_fact)

    hierarchy[1] = MGLevel(
        curr_HEE, curr_HEV, curr_HVV, curr_Scale_HEV, empty_R_E,
        Vector(diag(curr_HEE)), Vector(diag(curr_HVV)),
        zeros(dim_e), zeros(dim_v), zeros(dim_e), zeros(dim_v),
        zeros(dim_e), zeros(dim_v), zeros(dim_e), zeros(dim_v),
        amg_pack
    )
    
    return hierarchy
end

"""
    Rest_int_nodes(Levels::AbstractVector{Int}; index_type::Type{<:Integer}=Int)

Creates the restriction operator R_EE.
This handles varying discretization levels across different edges to ensure minimal 
system size upon refinement (levels J_min < J < J_max).

# Arguments
- `Levels`: Vector of Level of discretization on each edge.
- `index_type`: Integer type for sparse matrix indices.

# Output:
- `SparseMatrixCSC`: Block-diagonal matrix containing the restriction operators for each edge.
"""
function Rest_int_nodes(Levels::AbstractVector{Int}; index_type::Type{<:Integer}=Int)
    total_coarse_nodes = 0
    total_fine_nodes = 0
    nnz = 0
    
    for L in Levels
        if L > 1
            n_c = 2^(L-1) - 1 
            total_coarse_nodes += n_c
            nnz += 3 * n_c
            total_fine_nodes += 2^L - 1
        elseif L == 1
            total_fine_nodes += 1
        end
    end
    
    row_vec = Vector{index_type}(undef, nnz)
    col_vec = Vector{index_type}(undef, nnz)
    val_vec = Vector{Float64}(undef, nnz)
    
    idx = 1
    current_coarse_row = 1
    current_fine_col_offset = 0
    
    for L in Levels
        if L == 0
            continue
        elseif L == 1
            current_fine_col_offset += 1
        else
            n_c = 2^(L - 1) - 1
            n_f = 2^L - 1
            
            for k in 1:n_c
                row = current_coarse_row + k - 1
                base_col = current_fine_col_offset + 2k - 1
                
                row_vec[idx]   = row
                col_vec[idx]   = base_col
                val_vec[idx]   = 0.5
                
                row_vec[idx+1] = row
                col_vec[idx+1] = base_col + 1
                val_vec[idx+1] = 1.0
                
                row_vec[idx+2] = row
                col_vec[idx+2] = base_col + 2
                val_vec[idx+2] = 0.5
                
                idx += 3
            end
            
            current_coarse_row += n_c
            current_fine_col_offset += n_f
        end
    end
    
    return sparse(row_vec, col_vec, val_vec, total_coarse_nodes, total_fine_nodes)
end


"""
    Rest_HEV_setup(HEV::AbstractMatrix{<:Real}; index_type::Type{<:Integer}=Int)

Initializes the restriction operator for the matrix H_EV based on its nonzero pattern.

# Arguments
- `HEV`: Matrix H_EV.
- `index_type`: Integer type for sparse matrix indices.

# Output:
- `SparseMatrixCSC`: Initial restriction operator matrix for H_EV. Needs reduction via `Rest_HEV`.
"""
function Rest_HEV_setup(HEV::AbstractMatrix{<:Real}; index_type::Type{<:Integer}=Int)
        Zeilen, Spalten, _ = findnz(HEV)
    return sparse(index_type.(Zeilen), index_type.(Spalten), 0.5 * ones(length(Zeilen)), size(HEV, 1), size(HEV, 2))
end


"""
    Rest_HEV(Levels::AbstractVector{Int}; index_type::Type{<:Integer}=Int)

Creates a reduction matrix for the H_EV restriction operator to match the current 
discretization level dimensions in the multigrid refinement process.

# Arguments
- `Levels`: Vector of level of discretization on each edge.
- `index_type`: Integer type for sparse matrix indices.

# Output:
- `SparseMatrixCSC`: Matrix which reduces the size of R_EV when multiplied to it.
"""
function Rest_HEV(Levels::AbstractVector{Int}; index_type::Type{<:Integer}=Int)
        Levels_vec = Levels[Levels.>1]
        Levels_vec0 = Levels[Levels.!=0]
        
        nnz = 2 * length(Levels_vec)
        val_vec = ones(nnz)
        row_vec = ones(index_type, nnz)
        col_vec = zeros(index_type, nnz)
        
        if length(row_vec) != 0
                row_vec[2] = 2^(Levels_vec[1] - 1) - 1
                for j in 2:length(Levels_vec)
                        row_vec[2j-1] = row_vec[2j-2] + 1
                        row_vec[2j] = row_vec[2j-2] + 2^(Levels_vec[j] - 1) - 1
                end
        end

        plus1 = 0
        idx = 1
        plusend = 0
        
        for j in eachindex(Levels_vec0)
                if Levels_vec0[j] == 1
                        plus1 += 1
                else
                        col_vec[idx] = 1 + plusend + plus1
                        col_vec[idx+1] = (1 << Levels_vec0[j]) - 1 + plusend + plus1
                        
                        plusend = col_vec[idx+1]
                        plus1 = 0
                        idx += 2
                end
        end
        
        return sparse(row_vec, col_vec, val_vec, sum(2 .^ (Levels_vec .- 1) .- 1), sum(2 .^ (Levels_vec0) .- 1))
end

"""
    Multigrid_Graph(HEE, HEV, HVV, Levels, J, fv, fe, uv, ue, M, Scale_HEV, v1, v2, w, my)

Function for one symmetric multigrid cycle. Assumes the appropriate setup of an extended graph 
where every edge is discretized with 2^(J) subintervals.

# Arguments
- `HEE`: Submatrix H_EE.
- `HEV`: Submatrix H_EV.
- `HVV`: Submatrix H_VV.
- `Levels`: Vector of level of discretization on each edge.
- `J`: Counter for how often the level of discretization is yet to be reduced.
- `fv`: Right-hand-side on all vertices (f_V).
- `fe`: Right-hand-side on all internal vertices (f_E).
- `uv`: Current approximation of the solution on vertices (u_V).
- `ue`: Current approximation of the solution on edges (u_E).
- `M`: Amount of edges.
- `Scale_HEV`: Matrix of the nonzero pattern of H_EV used for the restriction operator.
- `v1`, `v2`: Amount of pre- and post-smoothing steps.
- `w`: Relaxation parameter for the smoothing method.
- `my`: Amount of recursive calls; 1: V-cycle, 2: W-cycle.

# Output:
- `Tuple{Vector{Float64}, Vector{Float64}}`: Smoothed iteration vector approximations `(ue, uv)`.
"""
function Multigrid_Graph(HEE::AbstractMatrix{<:Real},
        HEV::AbstractMatrix{<:Real}, HVV::AbstractMatrix{<:Real},
        Levels::AbstractVector{Int}, J::Int,
        fv::AbstractVector{<:Real}, fe::AbstractVector{<:Real},
        uv::AbstractVector{<:Real}, ue::AbstractVector{<:Real},
        M::Int, Scale_HEV::AbstractMatrix{<:Real},
        v1::Int, v2::Int, w::Float64, my::Int)
        if J == 0
                u = Matrix([HEE HEV; HEV' HVV]) \ [fe; fv]
                ue = u[1:length(fe)]; uv = u[length(fe)+1:end]
        else
                ue, uv = smoother_Jac(HEE, HEV, HVV, ue, uv, fe, fv, v1, w)
                
                Rest_Kanten = Rest_int_nodes(Levels)
                HEE_m1 = Rest_Kanten * HEE * Rest_Kanten'
                HEV_m1 = Rest_Kanten * HEE * Scale_HEV + Rest_Kanten * HEV
                HVV_m1 = HVV + 2 * Scale_HEV' * HEV + Scale_HEV' * HEE * Scale_HEV

                RestHEV = Rest_HEV(Levels)
                Scale_HEV_m1 = RestHEV * Scale_HEV

                d_J_e = fe - HEE * ue - HEV * uv
                d_J_v = fv - HEV' * ue - HVV * uv
                fe_m1 = Rest_Kanten * d_J_e
                fv_m1 = d_J_v + Scale_HEV' * d_J_e

                v_tilde_e = zeros(size(fe_m1))
                v_tilde_v = zeros(size(fv_m1))

                if J == 1
                        v_tilde_e, v_tilde_v = Multigrid_Graph(HEE_m1, HEV_m1, HVV_m1,
                                max.(Levels .- 1, 0), J - 1, fv_m1, fe_m1, v_tilde_v, v_tilde_e,
                                M, Scale_HEV_m1, v1, v2, w, 1)
                else
                        for i in 1:my
                                v_tilde_e, v_tilde_v = Multigrid_Graph(HEE_m1, HEV_m1, HVV_m1,
                                        max.(Levels .- 1, 0), J - 1, fv_m1, fe_m1, v_tilde_v, v_tilde_e,
                                        M, Scale_HEV_m1, v1, v2, w, my)
                        end
                end
                
                ue = ue + Rest_Kanten' * v_tilde_e + Scale_HEV * v_tilde_v
                uv = uv + v_tilde_v
                ## Post-smoothing
                ue, uv = smoother_Jac(HEE, HEV, HVV, ue, uv, fe, fv, v2, w)
        end

        return ue, uv
end

"""
    Multigrid_Graph_solve!(hierarchy, J, fv, fe, uv, ue, v1, v2, w, my)

Recursive V- or W-cycle utilizing the pre-allocated workspace in the MGLevel hierarchy to prevent memory allocations.

# Arguments
- `hierarchy::Vector{<:MGLevel}`: The pre-constructed multigrid hierarchy.
- `J::Int`: Current hierarchy depth.
- `fv::AbstractVector{<:Real}`: Right-hand side for vertices.
- `fe::AbstractVector{<:Real}`: Right-hand side for internal edges.
- `uv::AbstractVector{<:Real}`: Current approximation on vertices.
- `ue::AbstractVector{<:Real}`: Current approximation on internal edges.
- `v1::Int`: Number of pre-smoothing steps.
- `v2::Int`: Number of post-smoothing steps.
- `w::Float64`: Relaxation parameter.
- `my::Int`: Cycle type.

# Output:
- `Tuple{Vector{Float64}, Vector{Float64}}`: Smoothed approximations `(ue, uv)`.
"""
function Multigrid_Graph_solve!(hierarchy::Vector{<:MGLevel}, J::Int,
                                fv::AbstractVector{<:Real}, fe::AbstractVector{<:Real},
                                uv::AbstractVector{<:Real}, ue::AbstractVector{<:Real},
                                v1::Int, v2::Int, w::Float64, my::Int)
    
    lvl = hierarchy[J+1]
    if J == 0
        b_coarse = [fe; fv]
        amg = lvl.amg_data
        
        u_amg, _ = amg_solve(amg.levels, amg.A_coarsest, amg.cs, amg.cs_fact, b_coarse; 
                             tol=1e-10, maxiter=500, inner_iter=2)
        
        n_e = length(fe)
        copyto!(ue, u_amg[1:n_e])
        copyto!(uv, u_amg[n_e+1:end])
    else
        smoother_Jac!(lvl, ue, uv, fe, fv, v1, w)

        copyto!(lvl.tmp_e, fe)
        mul!(lvl.tmp_e, lvl.HEE, ue, -1.0, 1.0)
        mul!(lvl.tmp_e, lvl.HEV, uv, -1.0, 1.0) 

        copyto!(lvl.tmp_v, fv)
        mul!(lvl.tmp_v, lvl.HEV', ue, -1.0, 1.0)
        mul!(lvl.tmp_v, lvl.HVV, uv, -1.0, 1.0) 

        lvl_m1 = hierarchy[J] 
        
        mul!(lvl_m1.res_e, lvl.R_E, lvl.tmp_e)
        
        copyto!(lvl_m1.res_v, lvl.tmp_v)
        mul!(lvl_m1.res_v, lvl.Scale_HEV', lvl.tmp_e, 1.0, 1.0)

        fill!(lvl_m1.v_tilde_e, 0.0)
        fill!(lvl_m1.v_tilde_v, 0.0)

        cycles = J == 1 ? 1 : my
        for i in 1:cycles
            Multigrid_Graph_solve!(hierarchy, J - 1, 
                                   lvl_m1.res_v, lvl_m1.res_e, 
                                   lvl_m1.v_tilde_v, lvl_m1.v_tilde_e, 
                                   v1, v2, w, my)
        end

        mul!(ue, lvl.R_E', lvl_m1.v_tilde_e, 1.0, 1.0)
        mul!(ue, lvl.Scale_HEV, lvl_m1.v_tilde_v, 1.0, 1.0)
        
        @. uv = uv + lvl_m1.v_tilde_v

        smoother_Jac!(lvl, ue, uv, fe, fv, v2, w)
    end

    return ue, uv
end


"""
    smoother_Jac(HEE, HEV, HVV, ue, uv, fe, fv, v1, omega)

Function for the smoother of the multigrid method. A damped Jacobi-smoother with `v1` 
smoothing steps is utilized for both pre- and post-smoothing.

# Arguments
- `HEE`, `HEV`, `HVV`: System submatrices.
- `ue`, `uv`: Current solution approximations.
- `fe`, `fv`: Right-hand side vectors.
- `v1::Int`: Number of smoothing steps.
- `omega::Float64`: Relaxation parameter.

# Output:
- `Tuple{Vector{Float64}, Vector{Float64}}`: Smoothed approximations `(ue, uv)`.
"""
function smoother_Jac(HEE::AbstractArray{<:Real},
        HEV::AbstractArray{<:Real}, HVV::AbstractArray{<:Real},
        ue::AbstractVector{<:Real}, uv::AbstractVector{<:Real},
        fe::AbstractVector{<:Real}, fv::AbstractVector{<:Real},
        v1::Int, omega::Float64)

        diag_EE = diag(HEE)
        diag_VV = diag(HVV)
        
        HEV_T = sparse(transpose(HEV))

        for i in 1:1:v1
                ue_next = ue .- omega .* ((HEE * ue .+ HEV * uv .- fe) ./ diag_EE)
                uv = uv .- omega .* ((HEV_T * ue .+ HVV * uv .- fv) ./ diag_VV)
                ue = ue_next
        end
        
        return Vector(ue), Vector(uv)
end

"""
    smoother_Jac!(lvl::MGLevel, ue, uv, fe, fv, v1, omega)

Allocation-free damped Jacobi-smoother using in-place operations (`mul!`) and broadcasting.

# Arguments
- `lvl::MGLevel`: The current multigrid level object.
- `ue`, `uv`: Current solution approximations.
- `fe`, `fv`: Right-hand side vectors.
- `v1::Int`: Number of smoothing steps.
- `omega::Float64`: Relaxation parameter.

# Output:
- Modifies `ue` and `uv` in place.
"""
function smoother_Jac!(lvl::MGLevel, ue::AbstractVector{<:Real}, uv::AbstractVector{<:Real}, 
                       fe::AbstractVector{<:Real}, fv::AbstractVector{<:Real}, 
                       v1::Int, omega::Float64)
    for i in 1:v1
        mul!(lvl.tmp_e, lvl.HEE, ue)
        mul!(lvl.tmp_e, lvl.HEV, uv, 1.0, 1.0)
        @. lvl.ue_next = ue - omega * (lvl.tmp_e - fe) / lvl.diag_EE

        mul!(lvl.tmp_v, lvl.HEV', ue)
        mul!(lvl.tmp_v, lvl.HVV, uv, 1.0, 1.0)
        @. lvl.uv_next = uv - omega * (lvl.tmp_v - fv) / lvl.diag_VV
        
        copyto!(ue, lvl.ue_next)
        copyto!(uv, lvl.uv_next)
    end
end