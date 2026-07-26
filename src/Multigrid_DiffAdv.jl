"""
    AdvectionMGLevel{MHEE, MHEV, MHVE, MHVV, MPE, MPV, MRE, MRV, MK, V}

Structure for the advection multigrid level containing system matrices, 
transfer operators, smoother utilities, and workspace.

$FIELDS
"""
struct AdvectionMGLevel{
    MHEE<:AbstractMatrix{Float64}, MHEV<:AbstractMatrix{Float64},
    MHVE<:AbstractMatrix{Float64}, MHVV<:AbstractMatrix{Float64},
    MPE<:AbstractMatrix{Float64}, MPV<:AbstractMatrix{Float64},
    MRE<:AbstractMatrix{Float64}, MRV<:AbstractMatrix{Float64},
    MK<:Union{Nothing, AbstractMatrix{Float64}},
    V<:AbstractVector{Float64}
}
    "System matrix block for internal edge nodes (H_EE)"
    HEE::MHEE
    "System matrix block coupling edges to vertices (H_EV)"
    HEV::MHEV
    "System matrix block coupling vertices to edges (H_VE - Asymmetric)"
    HVE::MHVE
    "System matrix block for graph vertices (H_VV)"
    HVV::MHVV
    
    "Prolongation operator for internal edge nodes"
    P_E::MPE
    "Prolongation operator for vertices"
    P_V::MPV
    "Restriction operator for internal edge nodes"
    R_E::MRE
    "Restriction operator for vertices"
    R_V::MRV
    
    "Permutation vector for the Downwind Gauss-Seidel (DGS) smoother"
    p_sweep::Union{Nothing, Vector{Int}}
    "Global block matrix used for ILU preconditioning"
    K_global::MK
    "Factorization object for the ILU smoother"
    ilu_fact::Any
    
    "Preallocated workspace: residual for edges"
    res_e::V
    "Preallocated workspace: residual for vertices"
    res_v::V
    "Preallocated workspace: error correction for edges"
    v_tilde_e::V
    "Preallocated workspace: error correction for vertices"
    v_tilde_v::V

    "Algebraic Multigrid (AMG) setup for the coarsest geometric level"
    amg_data::Any
end


"""
    setup_advection_hierarchy(HEE, HEV, HVE, HVV, Levels, J_max, G, A_vec, macroscopic_order; index_type=Int)

Builds the matrix-dependent Petrov-Galerkin hierarchy for advection-dominated problems.

# Arguments
- `HEE`, `HEV`, `HVE`, `HVV`: Fine-level block matrices. Note the asymmetry (H_VE != H_EV').
- `Levels::AbstractVector{Int}`: Discretization level on each edge.
- `J_max::Int`: Maximum hierarchy depth.
- `G::SimpleDiGraph`: The directed graph topology.
- `A_vec::Vector{Float64}`: Advection coefficients for each edge.
- `macroscopic_order::Union{Nothing, Vector{Int}}`: Topological sorting for the DGS smoother.
- `index_type::Type{<:Integer}`: Integer type for sparse matrix indices.

# Output:
- `Vector{AdvectionMGLevel}`: The fully constructed advection multigrid hierarchy.
"""
function setup_advection_hierarchy(HEE, HEV, HVE, HVV, Levels, J_max, G, A_vec, macroscopic_order;
                                   index_type::Type{<:Integer}=Int)
    
    hierarchy = Vector{AdvectionMGLevel}(undef, J_max + 1)
    
    curr_HEE = HEE
    curr_HEV = HEV
    curr_HVE = HVE
    curr_HVV = HVV
    curr_Levels = copy(Levels)
    
    @info "Build hierarchy from finest level down to coarsest level..."
    
    for J in J_max:-1:1
        int_nodes = 2 .^ curr_Levels .- 1

        use_DGS = !isnothing(macroscopic_order)
        if use_DGS
            @info "Build DGS smoother for Level J=$J with macroscopic order"
            p_sweep = build_level_permutation(G, macroscopic_order, int_nodes, A_vec)
            K_global = nothing
            ilu_fact = nothing
        else
            p_sweep = nothing
            K_global = sparse([curr_HEE curr_HEV; curr_HVE curr_HVV])
            ilu_fact = ilu0(K_global)
        end
        @info "Build intergrid operators for Level J=$J..."

        P_E = Prol_int_nodes_MatrixDep(curr_Levels, curr_HEE; index_type=index_type)
        P_V = Prol_HEV_MatrixDep(curr_HEV, curr_HEE; index_type=index_type) 
        R_E = Rest_int_nodes_MatrixDep(curr_Levels, curr_HEE; index_type=index_type)
        R_V = Rest_HVE_MatrixDep(curr_HVE, curr_HEE; index_type=index_type) 
        
        dim_e = size(curr_HEE, 1)
        dim_v = size(curr_HVV, 1)
        
        hierarchy[J+1] = AdvectionMGLevel(
            curr_HEE, curr_HEV, curr_HVE, curr_HVV,
            P_E, P_V, R_E, R_V,
            p_sweep, K_global, ilu_fact,
            zeros(dim_e), zeros(dim_v), zeros(dim_e), zeros(dim_v),
            nothing 
        )
        
        HEE_m1 = R_E * curr_HEE * P_E
        HEV_m1 = R_E * curr_HEE * P_V + R_E * curr_HEV
        HVE_m1 = R_V * curr_HEE * P_E + curr_HVE * P_E
        HVV_m1 = curr_HVV + curr_HVE * P_V + R_V * curr_HEV + R_V * curr_HEE * P_V
        
        curr_Levels = max.(curr_Levels .- 1, 0)
        curr_HEE = HEE_m1
        curr_HEV = HEV_m1
        curr_HVE = HVE_m1
        curr_HVV = HVV_m1
    end
    
    dim_e = size(curr_HEE, 1)
    dim_v = size(curr_HVV, 1)
    empty_mat = sparse(Vector{index_type}(undef, 0), Vector{index_type}(undef, 0), Float64[], 0, 0)
    
    @info "Build algebraic Multigrid (AMG) hierarchy for the coarsest geometric level..."

    A_coarse = [curr_HEE curr_HEV; curr_HVE curr_HVV]
    
    Adj_coarse = Diagonal(diag(A_coarse)) - A_coarse
    Adj_sym = sparse(0.5 .* (Adj_coarse + Adj_coarse'))
    dropzeros!(Adj_sym)
    Gamma_coarse = SimpleGraph(Adj_sym)
    
    amg_levels, amg_A_c, amg_cs, amg_cs_fact = build_amg_hierarchy(
        A_coarse, Gamma_coarse; 
        max_coarse_size=50, 
        max_levels=10
    )
    
    amg_pack = (levels=amg_levels, A_coarsest=amg_A_c, cs=amg_cs, cs_fact=amg_cs_fact)
    
    hierarchy[1] = AdvectionMGLevel(
        curr_HEE, curr_HEV, curr_HVE, curr_HVV,
        empty_mat, empty_mat, empty_mat, empty_mat,
        nothing, nothing, nothing,
        zeros(dim_e), zeros(dim_v), zeros(dim_e), zeros(dim_v),
        amg_pack 
    )
    
    return hierarchy
end


"""
    Multigrid_Graph_adv(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real}, HVE::AbstractMatrix{<:Real}, HVV::AbstractMatrix{<:Real}, Levels::AbstractVector{Int}, J::Int, fv::AbstractVector{<:Real}, fe::AbstractVector{<:Real}, uv::AbstractVector{<:Real}, ue::AbstractVector{<:Real}, M::Int, Scale_HEV::AbstractMatrix{<:Real}, v1::Int, v2::Int, w::Float64, my::Int, G::SimpleDiGraph, A_vec::Vector{Float64}, macroscopic_order::Union{Nothing, Vector{Int}}, ILU_cache::Union{Nothing, Dict{Int, Tuple{SparseMatrixCSC, Any}}})

Executes a single multigrid cycle for advection-diffusion problems.

# Arguments
- `HEE`, `HEV`, `HVE`, `HVV`: System block matrices.
- `Levels`: Discretization vectors.
- `J`: Current hierarchy depth.
- `fv`, `fe`: Right-hand side vectors.
- `uv`, `ue`: Current solution approximations.
- `M`: Number of edges.
- `Scale_HEV`: Scale matrix (maintained for legacy signatures).
- `v1`, `v2`: Pre- and post-smoothing steps.
- `w`: Relaxation parameter.
- `my`: Cycle type (e.g., 1 for V-cycle, 2 for W-cycle).
- `G`: Directed graph structure.
- `A_vec`: Advection coefficients.
- `macroscopic_order`: Topological sort for DGS.
- `ILU_cache`: Dictionary storing precomputed ILU factorizations.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Smoothed approximations `(ue, uv)`.
"""
function Multigrid_Graph_adv(HEE::AbstractMatrix{<:Real},
        HEV::AbstractMatrix{<:Real}, 
        HVE::AbstractMatrix{<:Real}, 
        HVV::AbstractMatrix{<:Real},
        Levels::AbstractVector{Int}, J::Int,
        fv::AbstractVector{<:Real}, fe::AbstractVector{<:Real},
        uv::AbstractVector{<:Real}, ue::AbstractVector{<:Real},
        M::Int, Scale_HEV::AbstractMatrix{<:Real},
        v1::Int, v2::Int, w::Float64, my::Int, G::SimpleDiGraph, 
        A_vec::Vector{Float64}, macroscopic_order::Union{Nothing, Vector{Int}},
        ILU_cache::Union{Nothing, Dict{Int, Tuple{SparseMatrixCSC, Any}}})

        if J == 0
                u = [HEE HEV; HVE HVV] \ [fe; fv]
                ue = u[1:length(fe)]; uv = u[length(fe)+1:end]
        else
                int_nodes = 2 .^ Levels .- 1
                p_sweep = !isnothing(macroscopic_order) ? build_level_permutation(G, macroscopic_order, int_nodes, A_vec) : nothing
                if !isnothing(p_sweep)
                    println("Using DGS Smoother")
                    ue, uv = smoother_DGS(HEE, HEV, HVE, HVV, fe, fv, ue, uv, p_sweep, nu=v1)
                else    
                    println("Using ILU Smoother")
                    if !haskey(ILU_cache, J)
                        K_global = sparse([HEE HEV; HVE HVV])
                        ILU_cache[J] = (K_global, ilu0(K_global))
                    end
                    K_global, ilu_fact = ILU_cache[J]
                    
                    ue, uv = smoother_ILU(fe, fv, ue, uv, K_global, ilu_fact, nu=v1)
                    # ue, uv = smoother_GMRES(HEE, HEV, HVE, HVV, ue, uv, fe, fv, v1)
                end

                Prol_Kanten = Prol_int_nodes_MatrixDep(Levels, HEE)
                Prol_Scale_HEV = Prol_HEV_MatrixDep(HEV, HEE)

                Rest_Kanten = Rest_int_nodes_MatrixDep(Levels, HEE)
                Rest_HVE = Rest_HVE_MatrixDep(HVE, HEE)
                HEE_m1 = Rest_Kanten * HEE * Prol_Kanten
                
                HEV_m1 = Rest_Kanten * HEE * Prol_Scale_HEV + Rest_Kanten * HEV
                
                HVE_m1 = Rest_HVE * HEE * Prol_Kanten + HVE * Prol_Kanten
                
                HVV_m1 = HVV + HVE * Prol_Scale_HEV + Rest_HVE * HEV + Rest_HVE * HEE * Prol_Scale_HEV

                d_J_e = fe - HEE * ue - HEV * uv
                d_J_v = fv - HVE * ue - HVV * uv
                
                fe_m1 = Rest_Kanten * d_J_e
                fv_m1 = d_J_v + Rest_HVE * d_J_e

                v_tilde_e = zeros(size(fe_m1))
                v_tilde_v = zeros(size(fv_m1))

                if J == 1
                        v_tilde_e, v_tilde_v = Multigrid_Graph_adv(HEE_m1, HEV_m1, HVE_m1, HVV_m1,
                                max.(Levels .- 1, 0), J - 1, fv_m1, fe_m1, v_tilde_v, v_tilde_e,
                                M, Scale_HEV, v1, v2, w, 1, G, A_vec, macroscopic_order, ILU_cache) 
                else
                        for i in 1:my
                                v_tilde_e, v_tilde_v = Multigrid_Graph_adv(HEE_m1, HEV_m1, HVE_m1, HVV_m1,
                                        max.(Levels .- 1, 0), J - 1, fv_m1, fe_m1, v_tilde_v, v_tilde_e,
                                        M, Scale_HEV, v1, v2, w, my, G, A_vec, macroscopic_order, ILU_cache)
                        end
                end

                ue = ue + Prol_Kanten * v_tilde_e + Prol_Scale_HEV * v_tilde_v
                uv = uv + v_tilde_v

                if !isnothing(p_sweep)
                    ue, uv = smoother_DGS(HEE, HEV, HVE, HVV, fe, fv, ue, uv, p_sweep, nu=v2)
                else    
                    K_global, ilu_fact = ILU_cache[J]
                    ue, uv = smoother_ILU(fe, fv, ue, uv, K_global, ilu_fact, nu=v2)
                    # ue, uv = smoother_GMRES(HEE, HEV, HVE, HVV, ue, uv, fe, fv, v2)
                end
        end

        return ue, uv
end


"""
    solve_advection_MG!(hierarchy::Vector{<:AdvectionMGLevel}, J::Int, fv, fe, uv, ue, v1, v2, my)

Recursive advection multigrid cycle utilizing the pre-allocated workspace.

# Arguments
- `hierarchy::Vector{<:AdvectionMGLevel}`: The constructed advection multigrid hierarchy.
- `J::Int`: Current hierarchy depth.
- `fv`, `fe`: Right-hand side vectors.
- `uv`, `ue`: Current solution approximations.
- `v1`, `v2`: Pre- and post-smoothing steps.
- `my::Int`: Cycle type.

# Output:
- `Tuple{Vector{Float64}, Vector{Float64}}`: Solved approximations `(ue, uv)`.
"""
function solve_advection_MG!(hierarchy::Vector{<:AdvectionMGLevel}, J::Int, 
                             fv::AbstractVector{<:Real}, fe::AbstractVector{<:Real},
                             uv::AbstractVector{<:Real}, ue::AbstractVector{<:Real},
                             v1::Int, v2::Int, my::Int)
    
    lvl = hierarchy[J+1]
    
    if J == 0
        b_coarse = [fe; fv]
        amg = lvl.amg_data
        if isempty(amg.levels)
            @info "Solving using Backlash due to initial vertex set being too small for AMG coarsening..."
            u = [lvl.HEE lvl.HEV; lvl.HVE lvl.HVV] \ [fe; fv]
            copyto!(ue, u[1:length(fe)])
            copyto!(uv, u[length(fe)+1:end])
            return ue, uv
        end

        u_amg, _ = amg_solve(amg.levels, amg.A_coarsest, amg.cs, amg.cs_fact, b_coarse; 
                             tol=1e-10, maxiter=500, inner_iter=2)

        n_e = length(fe)
        copyto!(ue, u_amg[1:n_e])
        copyto!(uv, u_amg[n_e+1:end])
        
        return ue, uv
    end

    if !isnothing(lvl.p_sweep)
        ue_new, uv_new = smoother_DGS(lvl.HEE, lvl.HEV, lvl.HVE, lvl.HVV, fe, fv, ue, uv, lvl.p_sweep, nu=v1)
        copyto!(ue, ue_new); copyto!(uv, uv_new)
    else
        ue_new, uv_new = smoother_ILU(fe, fv, ue, uv, lvl.K_global, lvl.ilu_fact, nu=v1)
        # ue_new, uv_new = smoother_GMRES(lvl.HEE, lvl.HEV, lvl.HVE, lvl.HVV, ue, uv, fe, fv, v1)
        copyto!(ue, ue_new); copyto!(uv, uv_new)
    end
    
    d_J_e = fe - lvl.HEE * ue - lvl.HEV * uv
    d_J_v = fv - lvl.HVE * ue - lvl.HVV * uv
    
    lvl_m1 = hierarchy[J]
    
    fe_m1 = lvl.R_E * d_J_e
    fv_m1 = d_J_v + lvl.R_V * d_J_e
    
    copyto!(lvl_m1.res_e, fe_m1)
    copyto!(lvl_m1.res_v, fv_m1)
    fill!(lvl_m1.v_tilde_e, 0.0)
    fill!(lvl_m1.v_tilde_v, 0.0)
    
    cycles = J == 1 ? 1 : my
    for i in 1:cycles
        solve_advection_MG!(hierarchy, J - 1, 
                            lvl_m1.res_v, lvl_m1.res_e, 
                            lvl_m1.v_tilde_v, lvl_m1.v_tilde_e, 
                            v1, v2, my)
    end
    
    ue .+= lvl.P_E * lvl_m1.v_tilde_e .+ lvl.P_V * lvl_m1.v_tilde_v
    uv .+= lvl_m1.v_tilde_v
    
    if !isnothing(lvl.p_sweep)
        ue_new, uv_new = smoother_DGS(lvl.HEE, lvl.HEV, lvl.HVE, lvl.HVV, fe, fv, ue, uv, lvl.p_sweep, nu=v2)
        copyto!(ue, ue_new); copyto!(uv, uv_new)
    else
        ue_new, uv_new = smoother_ILU(fe, fv, ue, uv, lvl.K_global, lvl.ilu_fact, nu=v2)
        # ue_new, uv_new = smoother_GMRES(lvl.HEE, lvl.HEV, lvl.HVE, lvl.HVV, ue, uv, fe, fv, v2)
        copyto!(ue, ue_new); copyto!(uv, uv_new)
    end
    
    return ue, uv
end

"""
    Prol_int_nodes_MatrixDep(Levels, HEE; index_type=Int)

Builds the matrix-dependent prolongation operator for internal edge nodes, 
incorporating advection asymmetry from H_EE.

# Arguments
- `Levels`: Discretization vectors.
- `HEE`: System submatrix H_EE.
- `index_type`: Integer type for sparse matrix indices.

# Output:
- `SparseMatrixCSC`: Prolongation operator.
"""
function Prol_int_nodes_MatrixDep(Levels::AbstractVector{Int}, HEE::AbstractMatrix{<:Real};index_type::Type{<:Integer}=Int)
    N_fine_total = 0
    N_coarse_total = 0
    
    for L in Levels
        if L > 0
            N_fine_total += 2^L - 1
            if L > 1
                N_coarse_total += 2^(L-1) - 1
            end
        end
    end

    # Each coarse node maps to exactly 3 fine nodes (left, center, right)
    nnz_total = 3 * N_coarse_total
    
    I_vec = Vector{index_type}(undef, nnz_total)     
    J_vec = Vector{index_type}(undef, nnz_total)    
    V_vec = Vector{Float64}(undef, nnz_total)

    d_EE = diag(HEE) 
    
    offset_c = 0
    offset_f = 0
    idx = 1

    @inbounds for L in Levels
        if L == 0
            continue
        elseif L == 1
            offset_f += 1
        else
            Nc = 2^(L-1) - 1
            Nf = 2^L - 1
            
            for c in 1:Nc
                global_c = offset_c + c
                # Coarse node c structurally corresponds to fine node 2c
                global_f_center = offset_f + 2 * c 
                
                idx_L = global_f_center - 1
                idx_C = global_f_center
                idx_R = global_f_center + 1
                
                # Assign left neighbour mapping
                I_vec[idx]   = idx_L
                J_vec[idx]   = global_c
                V_vec[idx]   = -HEE[idx_L, idx_C] / d_EE[idx_L]
                
                # Assign center mapping (Identity)
                I_vec[idx+1] = idx_C
                J_vec[idx+1] = global_c
                V_vec[idx+1] = 1.0
                
                # Assign right neighbour mapping
                I_vec[idx+2] = idx_R
                J_vec[idx+2] = global_c
                V_vec[idx+2] = -HEE[idx_R, idx_C] / d_EE[idx_R]
                
                idx += 3
            end
            
            # Shift global offsets for the next edge
            offset_c += Nc
            offset_f += Nf
        end
    end
    Prol_Kanten = sparse(I_vec, J_vec, V_vec, N_fine_total, N_coarse_total)
    return Prol_Kanten
end

function Prol_HEV_MatrixDep(HEV::SparseMatrixCSC{<:Real, Int}, HEE::AbstractMatrix{<:Real})
    return Prol_HEV_MatrixDep(HEV, HEE; index_type=Int)
end

"""
    Prol_HEV_MatrixDep(HEV, HEE; index_type=Ti)

Adapts the vertex-to-edge prolongation based on matrix dependencies.

# Arguments
- `HEV`: System matrix block coupling edges to vertices.
- `HEE`: System submatrix H_EE.
- `index_type`: Integer type for sparse matrix indices.

# Output:
- `SparseMatrixCSC`: Adjusted prolongation operator.
"""
function Prol_HEV_MatrixDep(HEV::SparseMatrixCSC{Tv, Ti}, HEE::AbstractMatrix{<:Real}; 
                            index_type::Type{<:Integer}=Ti) where {Tv, Ti <: Integer}
    Prol = copy(HEV)
    d_EE = diag(HEE)
    
    @inbounds for k in eachindex(Prol.nzval)
        row_idx = Prol.rowval[k]
        Prol.nzval[k] = -Prol.nzval[k] / d_EE[row_idx]
    end
    
    # Konvertiere nur, wenn der gewünschte Typ vom aktuellen Matrix-Typ abweicht
    if index_type != Ti
        return SparseMatrixCSC{Tv, index_type}(Prol.m, Prol.n, 
                    convert(Vector{index_type}, Prol.colptr), 
                    convert(Vector{index_type}, Prol.rowval), 
                    Prol.nzval)
    end
    
    return Prol
end

"""
    Rest_int_nodes_MatrixDep(Levels, HEE; index_type=Int)

Builds the restriction operator for internal edge nodes using matrix-dependent weights 
to properly collect upstream/downstream residuals.

# Arguments
- `Levels`: Discretization vectors.
- `HEE`: System submatrix H_EE.
- `index_type`: Integer type for sparse matrix indices.

# Output:
- `SparseMatrixCSC`: Matrix-dependent restriction operator.
"""
function Rest_int_nodes_MatrixDep(Levels::AbstractVector{Int}, HEE::AbstractMatrix{<:Real}; 
                                  index_type::Type{<:Integer}=Int)
    N_fine_total = 0
    N_coarse_total = 0
    
    for L in Levels
        if L > 0
            N_fine_total += 2^L - 1
            if L > 1
                N_coarse_total += 2^(L-1) - 1
            end
        end
    end

    nnz_total = 3 * N_coarse_total
    
    I_vec = Vector{index_type}(undef, nnz_total)     
    J_vec = Vector{index_type}(undef, nnz_total)     
    V_vec = Vector{Float64}(undef, nnz_total)
    
    d_EE = diag(HEE)
    
    offset_c = 0
    offset_f = 0
    idx = 1
    
    @inbounds for L in Levels
        if L == 0
            continue
        elseif L == 1
            offset_f += 1 
        else
            Nc = 2^(L-1) - 1
            Nf = 2^L - 1
            
            for c in 1:Nc
                global_c = offset_c + c
                
                global_f_center = offset_f + 2 * c
                
                idx_L = global_f_center - 1
                idx_C = global_f_center
                idx_R = global_f_center + 1
                
                I_vec[idx]   = global_c
                J_vec[idx]   = idx_L
                V_vec[idx]   = -HEE[idx_C, idx_L] / d_EE[idx_L]
                
                I_vec[idx+1] = global_c
                J_vec[idx+1] = idx_C
                V_vec[idx+1] = 1.0
                
                I_vec[idx+2] = global_c
                J_vec[idx+2] = idx_R
                V_vec[idx+2] = -HEE[idx_C, idx_R] / d_EE[idx_R]
                
                idx += 3
            end
            
            offset_c += Nc
            offset_f += Nf
        end
    end
    
    Rest_Kanten = sparse(I_vec, J_vec, V_vec, N_coarse_total, N_fine_total)
    return Rest_Kanten
end

function Rest_HVE_MatrixDep(HVE::SparseMatrixCSC{<:Real, Int}, HEE::AbstractMatrix{<:Real})
    return Rest_HVE_MatrixDep(HVE, HEE; index_type=Int)
end

"""
    Rest_HVE_MatrixDep(HVE, HEE; index_type=Ti)

Restriction operator mapping fine edge node residuals onto adjacent graph vertices.

# Arguments
- `HVE`: System matrix block coupling vertices to edges.
- `HEE`: System submatrix H_EE.
- `index_type`: Integer type for sparse matrix indices.

# Output:
- `SparseMatrixCSC`: Restriction operator onto the vertices.
"""
function Rest_HVE_MatrixDep(HVE::SparseMatrixCSC{Tv, Ti}, HEE::AbstractMatrix{<:Real}; 
                            index_type::Type{<:Integer}=Ti) where {Tv, Ti <: Integer}
    Rest_Kanten = copy(HVE)
    d_EE = diag(HEE)
    
    @inbounds for col in 1:Rest_Kanten.n
        nz_start = Rest_Kanten.colptr[col]
        nz_end   = Rest_Kanten.colptr[col+1] - 1
        inv_d_EE_col = 1.0 / d_EE[col]
        
        for k in nz_start:nz_end
            Rest_Kanten.nzval[k] = -Rest_Kanten.nzval[k] * inv_d_EE_col
        end
    end
    
    if index_type != Ti
        return SparseMatrixCSC{Tv, index_type}(Rest_Kanten.m, Rest_Kanten.n, 
                    convert(Vector{index_type}, Rest_Kanten.colptr), 
                    convert(Vector{index_type}, Rest_Kanten.rowval), 
                    Rest_Kanten.nzval)
    end
    
    return Rest_Kanten
end

function Prol_HEV_MatrixDep(HEV::AbstractMatrix{<:Real}, HEE::AbstractMatrix{<:Real}, Scale_HEV::AbstractMatrix{<:Real}; index_type::Type{<:Integer}=Int)
    Zeilen, Spalten, _ = findnz(Scale_HEV)
    Zeilen = convert(Vector{index_type}, Zeilen)
    Spalten = convert(Vector{index_type}, Spalten)
    val_vec = zeros(length(Zeilen))
    d_EE = diag(HEE)
    for i in eachindex(Zeilen)
        idx_E = Zeilen[i]
        val_vec[i] = -HEV[idx_E, Spalten[i]] / d_EE[idx_E]
    end
    return sparse(Zeilen, Spalten, val_vec, HEV.m, HEV.n)
end

function Rest_HVE_MatrixDep(HVE::AbstractMatrix{<:Real}, HEE::AbstractMatrix{<:Real}, Scale_HEV::AbstractMatrix{<:Real}; index_type::Type{<:Integer}=Int)
    Zeilen, Spalten, _ = findnz(sparse(transpose(Scale_HEV)))
    Zeilen = convert(Vector{index_type}, Zeilen)
    Spalten = convert(Vector{index_type}, Spalten)
    val_vec = zeros(length(Zeilen))
    d_EE = diag(HEE)
    for i in eachindex(Zeilen)
        idx_V = Zeilen[i]
        idx_E = Spalten[i]
        val_vec[i] = -HVE[idx_V, idx_E] / d_EE[idx_E]
    end
    return sparse(Zeilen, Spalten, val_vec, HVE.m, HVE.n)
end