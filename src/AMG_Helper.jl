# ---------------------------------------------------------------------------------
# AMG_Helper.jl
# Helper functions for the Algebraic Multigrid (AMG) framework, specifically 
# adapted for advection-diffusion problems on metric graphs.
# All baseline functions are taken from L. Schmitz, "Multigrid Methoden für 
# Graph-Laplace Matrizen", Master's Thesis, 2026.
# ---------------------------------------------------------------------------------


"""
    DRA_QC_CE(L::AbstractGraph; θ::Float64=10.0, η::Float64=2.0)

Degree-Aware Rooted Aggregation with Quality Control and Complexity Enhancement.
Direct replacement for `DRA_mix(L)`: same input, same output structure.

# Arguments
- `L::AbstractGraph`: The input graph (from Graphs.jl).
- `θ::Float64`: Quality threshold (default: 10.0).
- `η::Float64`: Safety factor for criterion (default: 2.0).

# Output
- `Vector{Vector{Int}}`: The final clustered vertex partition.
"""
function DRA_QC_CE(L; θ=10.0, η=2.0)
    Partition, _ = _dra_qc_ce_internal(L; θ=θ, η=η)
    return Partition
end


"""
    DRA_QC_CE_detailed(L::AbstractGraph; θ::Float64=10.0, η::Float64=2.0)

Extended version of `DRA_QC_CE` that returns QC flags for analysis and comparison.

# Arguments
- `L::AbstractGraph`: The input graph.
- `θ::Float64`: Quality threshold (default: 10.0).
- `η::Float64`: Safety factor (default: 2.0).

# Output
- `Tuple{Vector{Vector{Int}}, Vector{Bool}}`: The computed partition and the corresponding QC boolean flags.
"""
function DRA_QC_CE_detailed(L; θ=10.0, η=2.0)
    return _dra_qc_ce_internal(L; θ=θ, η=η)
end


"""
    DRA_QC(L::AbstractGraph; θ::Float64=10.0, η::Float64=2.0)

Degree-Aware Rooted Aggregation with Quality Control ONLY (no Complexity Enhancement).
Demonstrates the pure effect of the QC without the CE repair step. 
Aggregates with size |G| ≤ 3 remain as singletons.

# Arguments
- `L::AbstractGraph`: The input graph.
- `θ::Float64`: Quality threshold (default: 10.0).
- `η::Float64`: Safety factor (default: 2.0).

# Output
- `Vector{Vector{Int}}`: Global vertex indices per aggregate without CE post-processing.
"""
function DRA_QC(L; θ=10.0, η=2.0)
    W       = Float64.(adjacency_matrix(L))
    n       = nv(L)
    d_total = vec(sum(W, dims=2))
    γ       = _compute_gamma(W, d_total, n)

    deg    = degree(L)
    active = copy(deg)

    Partition = Vector{Vector{Int}}()

    while maximum(active) > -1
        r = argmax(active)
        G = _dra_tentative(L, r, active)
        G = _qc_extract(G, W, γ, r, θ, η)
        active[G] .= -1
        push!(Partition, G)
    end

    return Partition
end


"""
    _dra_qc_ce_internal(L::AbstractGraph; θ::Float64=10.0, η::Float64=2.0)

Internal function handling the core logic for the DRA_QC_CE algorithm.

# Arguments
- `L::AbstractGraph`: The input graph.
- `θ::Float64`: Quality threshold.
- `η::Float64`: Safety factor.

# Output
- `Tuple{Vector{Vector{Int}}, Vector{Bool}}`: Partition vectors and QC execution flags.
"""
function _dra_qc_ce_internal(L; θ=10.0, η=2.0)
    W       = Float64.(adjacency_matrix(L))
    n       = nv(L)
    d_total = vec(sum(W, dims=2))
    γ       = _compute_gamma(W, d_total, n)

    deg    = degree(L)
    active = copy(deg)       

    Partition = Vector{Vector{Int}}()
    qc_flags  = Vector{Bool}()  

    while maximum(active) > -1
        r = argmax(active)
        G = _dra_tentative(L, r, active)
        G = _qc_extract(G, W, γ, r, θ, η)
        active[G] .= -1
        push!(Partition, G)
        push!(qc_flags, true)
    end

    n_c = length(Partition)
    if n_c > n ÷ 4
        small_idx = findall(p -> length(p) <= 3, Partition)
        released  = vcat(Partition[small_idx]...)
        deleteat!(Partition, sort(small_idx))
        deleteat!(qc_flags,  sort(small_idx))

        for v in released
            active[v] = deg[v]
        end

        while maximum(active) > -1
            r = argmax(active)
            G = _dra_tentative(L, r, active)
            active[G] .= -1
            push!(Partition, G)
            push!(qc_flags, false)   
        end
    end

    return Partition, qc_flags
end


"""
    _compute_gamma(W::AbstractMatrix{<:Real}, d::AbstractVector{<:Real}, n::Int)

Computes the auxiliary values `γ` representing local connectivity potential for each node.

# Arguments
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `d::AbstractVector{<:Real}`: Total degree vector.
- `n::Int`: Number of nodes.

# Output
- `Vector{Float64}`: Gamma values for density estimation.
"""
function _compute_gamma(W, d, n)
    γ    = zeros(n)
    rows = rowvals(W)
    vals = nonzeros(W)
    for j in 1:n
        for idx in nzrange(W, j)
            k = rows[idx]
            w = vals[idx]
            k > j && (γ[j] += w^2 / d[k])
        end
    end
    return γ
end


"""
    _dra_tentative(L::AbstractGraph, r::Int, active::Vector{Int})

Generates a tentative aggregate centered at root node `r` based on 1-ring and 2-ring connectivity.

# Arguments
- `L::AbstractGraph`: The input graph.
- `r::Int`: The root node index.
- `active::Vector{Int}`: Indicator vector for active nodes in the pool.

# Output
- `Vector{Int}`: Tentative cluster members.
"""
function _dra_tentative(L, r, active)
    G     = Int[r]
    G_set = Set(G)
    for nb in neighbors(L, r)
        if active[nb] >= 0
            push!(G, nb)
            push!(G_set, nb)
        end
    end
    if length(G) <= 6
        for v in copy(G)
            for nb in neighbors(L, v)
                if active[nb] >= 0 && nb ∉ G_set
                    push!(G, nb)
                    push!(G_set, nb)
                end
            end
        end
    end
    return G
end


"""
    _ext_weight(W::AbstractMatrix{<:Real}, j::Int, G_set::Set{Int})

Calculates external edge weights of vertex `j` relative to cluster set `G_set`.

# Arguments
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `j::Int`: Node index.
- `G_set::Set{Int}`: Set of nodes in aggregate G.

# Output
- `Float64`: Computed external weight.
"""
function _ext_weight(W, j, G_set)
    ext  = 0.0
    rows = rowvals(W)
    vals = nonzeros(W)
    for idx in nzrange(W, j)
        k = rows[idx]
        k ∉ G_set && (ext += vals[idx])
    end
    return ext
end


"""
    _int_weight(W::AbstractMatrix{<:Real}, j::Int, G_set::Set{Int})

Calculates internal edge weights of vertex `j` relative to cluster set `G_set`.

# Arguments
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `j::Int`: Node index.
- `G_set::Set{Int}`: Set of nodes in aggregate G.

# Output
- `Float64`: Computed internal weight.
"""
function _int_weight(W, j, G_set)
    s    = 0.0
    rows = rowvals(W)
    vals = nonzeros(W)
    for idx in nzrange(W, j)
        k = rows[idx]
        (k ∈ G_set && k ≠ j) && (s += vals[idx])
    end
    return s
end


"""
    _build_AG_XG(G::AbstractVector{Int}, W::AbstractMatrix{<:Real}, γ::AbstractVector{<:Real})

Constructs the auxiliary Laplacian `AG` and the density matrix `XG` for spectral testing.

# Arguments
- `G::AbstractVector{Int}`: Aggregate node indices.
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `γ::AbstractVector{<:Real}`: Gamma vector.

# Output
- `Tuple{Matrix{Float64}, Matrix{Float64}}`: Tuple `(AG, XG)`.
"""
function _build_AG_XG(G, W, γ)
    m       = length(G)
    idx_map = Dict(G[i] => i for i in 1:m)
    AG      = zeros(m, m)
    ext_vec = zeros(m)
    G_set   = Set(G)
    rows    = rowvals(W)
    vals    = nonzeros(W)

    for (li, vi) in enumerate(G)
        for idx in nzrange(W, vi)
            k = rows[idx]
            w = vals[idx]
            if k ∈ G_set && k ≠ vi
                lk = idx_map[k]
                AG[li, lk]  = -w
                AG[li, li] +=  w
            else
                ext_vec[li] += w
            end
        end
    end

    Gamma_diag = [γ[G[i]] + 2.0 * ext_vec[i] for i in 1:m]
    XG         = AG + Diagonal(Gamma_diag)
    return AG, XG
end


"""
    _criterion_16(G::AbstractVector{Int}, W::AbstractMatrix{<:Real}, γ::AbstractVector{<:Real}, r::Int, θ::Float64)

Criterion (16) (from [NN]): Sufficient condition μ(G) < θ for fast evaluation.

# Arguments
- `G::AbstractVector{Int}`: Aggregate node indices.
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `γ::AbstractVector{<:Real}`: Gamma vector.
- `r::Int`: Root node.
- `θ::Float64`: Quality threshold.

# Output
- `Bool`: Returns `true` if criterion is met.
"""
function _criterion_16(G, W, γ, r, θ)
    G_set = Set(G)
    for j in G
        j == r && continue
        w_jr = W[j, r]
        w_jr == 0.0 && return false
        ext = _ext_weight(W, j, G_set)
        (2ext + γ[j]) / w_jr > θ - 1 && return false
    end
    return true
end


"""
    _bad_vertices_removal(G::AbstractVector{Int}, W::AbstractMatrix{<:Real}, γ::AbstractVector{<:Real}, r::Int, θ::Float64, η::Float64)

Iteratively removes vertices from aggregate `G` that violate the density/QC conditions.

# Arguments
- `G::AbstractVector{Int}`: Aggregate node indices.
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `γ::AbstractVector{<:Real}`: Gamma vector.
- `r::Int`: Root node.
- `θ::Float64`: Quality threshold.
- `η::Float64`: Safety factor.

# Output
- `Vector{Int}`: Cleaned cluster partition.
"""
function _bad_vertices_removal(G, W, γ, r, θ, η)
    G     = copy(G)
    large = length(G) > 1024

    cleaned = false
    while !cleaned
        cleaned = true
        i = 1
        while i <= length(G)
            j = G[i]
            if j == r; i += 1; continue; end

            G_set = Set(G)
            ext   = _ext_weight(W, j, G_set)
            w_jr  = W[j, r]

            # Criteria (18): upper bound
            c18 = (w_jr > 0.0) && ((2ext + γ[j]) / w_jr <= θ - 1)

            # Criteria (19): lower bound, only for |G| ≤ 1024
            c19 = false
            if !c18 && !large
                int_w = _int_weight(W, j, G_set)
                c19   = (int_w > 0.0) && ((2ext + γ[j]) / int_w <= (θ - 1) / η)
            end

            if c18 || c19
                i += 1
            else
                deleteat!(G, i)
                cleaned = false
            end
        end
    end
    return G
end


"""
    _cholesky_quality_test(G::AbstractVector{Int}, W::AbstractMatrix{<:Real}, γ::AbstractVector{<:Real}, θ::Float64)

Tests cluster `G` quality using a Cholesky decomposition of the aggregated density matrix.

# Arguments
- `G::AbstractVector{Int}`: Aggregate node indices.
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `γ::AbstractVector{<:Real}`: Gamma vector.
- `θ::Float64`: Quality threshold.

# Output
- `Tuple{Bool, Union{Vector{Float64}, Nothing}}`: Success status and Fiedler-like eigenvector if failed.
"""
function _cholesky_quality_test(G, W, γ, θ)
    m = length(G)
    m == 1 && return true, nothing

    AG, XG = _build_AG_XG(G, W, γ)

    xg1 = XG * ones(m)
    s   = dot(ones(m), xg1)
    ZG  = θ .* AG .- XG .+ (xg1 * xg1') ./ s

    ZG_sub = (ZG[1:m-1, 1:m-1] .+ ZG[1:m-1, 1:m-1]') ./ 2
    F      = cholesky(Symmetric(ZG_sub); check=false)
    issuccess(F) && return true, nothing

    try
        AG_s = (AG .+ AG') ./ 2
        XG_s = (XG .+ XG') ./ 2
        e    = eigen(Symmetric(AG_s), Symmetric(XG_s))
        ord  = sortperm(real.(e.values))
        v    = real.(e.vectors[:, ord[2]])
        return false, v
    catch
        return false, randn(m)
    end
end


"""
    _subgroup_extraction(G::AbstractVector{Int}, W::AbstractMatrix{<:Real}, γ::AbstractVector{<:Real}, r::Int, θ::Float64, η::Float64, v::AbstractVector{<:Real})

Refines cluster `G` by splitting it into subgroups based on spectral information `v`.

# Arguments
- `G::AbstractVector{Int}`: Aggregate node indices.
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `γ::AbstractVector{<:Real}`: Gamma vector.
- `r::Int`: Root node.
- `θ::Float64`: Quality threshold.
- `η::Float64`: Safety factor.
- `v::AbstractVector{<:Real}`: Fiedler vector.

# Output
- `Vector{Int}`: Extracted subgroup.
"""
function _subgroup_extraction(G, W, γ, r, θ, η, v)
    r_local = findfirst(==(r), G)
    vr      = v[r_local]

    Gp_set = if vr == 0.0
        Set(G[i] for i in 1:length(G) if v[i] >= 0.0)
    else
        Set(G[i] for i in 1:length(G) if v[i] * vr >= 0.0)
    end

    new_G = Int[r]
    for j in G
        j == r && continue
        ext   = _ext_weight(W, j, Gp_set)
        w_jr  = W[j, r]
        int_w = _int_weight(W, j, Gp_set)
        c_a   = (w_jr  > 0.0) && ((2ext + γ[j]) / w_jr  <= θ - 1)
        c_b   = (int_w > 0.0) && ((2ext + γ[j]) / int_w <= (θ - 1) / η)
        (c_a || c_b) && push!(new_G, j)
    end
    return new_G
end


"""
    _qc_extract(G::AbstractVector{Int}, W::AbstractMatrix{<:Real}, γ::AbstractVector{<:Real}, r::Int, θ::Float64, η::Float64)

Performs Quality Control (QC) and refinement on a tentative cluster aggregate `G`.

# Arguments
- `G::AbstractVector{Int}`: Tentative aggregate node indices.
- `W::AbstractMatrix{<:Real}`: Weighted adjacency matrix.
- `γ::AbstractVector{<:Real}`: Gamma vector.
- `r::Int`: Root node.
- `θ::Float64`: Quality threshold.
- `η::Float64`: Safety factor.

# Output
- `Vector{Int}`: Refined cluster partition.
"""
function _qc_extract(G, W, γ, r, θ, η)
    length(G) == 1 && return G

    current_η = η

    while length(G) > 1

        # 1. Bad Vertices Removal
        G = _bad_vertices_removal(G, W, γ, r, θ, current_η)
        length(G) == 1 && return G

        # 2. Billiges Kriterium (16)
        _criterion_16(G, W, γ, r, θ) && return G

        # 3. Große Aggregate: nur Kriterium (16) → akzeptieren
        length(G) > 1024 && return G

        # 4. Cholesky-Qualitätstest
        passed, v = _cholesky_quality_test(G, W, γ, θ)
        passed && return G

        # 5. Subgroup Extraction
        current_η += 0.5
        G_neu = _subgroup_extraction(G, W, γ, r, θ, current_η, v)

        if length(G_neu) < length(G)
            G = G_neu
        else
            # Stagnation: Notfallreduktion auf direkte Nachbarn + einmaliges
            # Bad-Vertices-Removal, dann akzeptieren (sehr selten)
            G_fb = [j for j in G if j == r || W[j, r] > 0.0]
            return _bad_vertices_removal(G_fb, W, γ, r, θ, current_η)
        end
    end

    return G
end


"""
    DRA_mix_alt(L::AbstractGraph)

Alternative Degree-Aware Rooted Aggregation approach.

# Arguments
- `L::AbstractGraph`: The input graph.

# Output
- `Vector{Vector{Int}}`: Computed aggregate partitions.
"""
function DRA_mix_alt(L)
    deg = degree(L)
    CheckedKnots = []
    Partition = []
    n_c = 0
    while maximum(deg) > -1
        n_c = n_c+1
        maxi = maximum(deg;dims=1)
        b = findfirst(item -> item == maxi[1],deg)
        CurrentPartition=[]
        append!(CurrentPartition,b)
        append!(CheckedKnots,b)
        Nb = neighbors(L,b)
        for i in Nb
            if i ∉ CheckedKnots
                append!(CurrentPartition,i)
                append!(CheckedKnots,i)
            end
        end
        if length(CurrentPartition) <= 6
            Adding =[]
            for i in CurrentPartition
                NCP = neighbors(L,i)
                for j in NCP
                    if j ∉ CheckedKnots
                        append!(Adding,j)
                        append!(CheckedKnots,j)
                    end
                end
            end
            append!(CurrentPartition,Adding)
        end
        deg[CurrentPartition] .= -1
        push!(Partition,CurrentPartition)
    end
    return Partition
end


"""
    DRA_mix(L::AbstractGraph)

Standard Degree-Aware Rooted Aggregation (DRA). Optimized implementation avoiding unnecessary memory re-allocations.

# Arguments
- `L::AbstractGraph`: The input graph.

# Output
- `Vector{Vector{Int}}`: Computed aggregate partitions.
"""
function DRA_mix(L)
    n = nv(L)
    deg = degree(L)
    
    Partition = Vector{Vector{Int}}()
    sizehint!(Partition, n ÷ 4) 

    sorted_nodes = sortperm(deg, rev=true)

    @inbounds for r in sorted_nodes
        deg[r] == -1 && continue

        G = Int[r]
        sizehint!(G, 16) 
        deg[r] = -1      

        for nb in neighbors(L, r)
            if deg[nb] != -1
                push!(G, nb)
                deg[nb] = -1
            end
        end

        n_G = length(G)
        if n_G <= 6
            for i in 1:n_G
                v = G[i]
                for nb in neighbors(L, v)
                    if deg[nb] != -1
                        push!(G, nb)
                        deg[nb] = -1
                    end
                end
            end
        end

        push!(Partition, G)
    end

    return Partition
end


"""
    _solve_singular_laplacian(L_mat::AbstractMatrix{<:Real}, r::AbstractVector{<:Real})

Solves a singular Laplacian system by fixing the first degree of freedom.

# Arguments
- `L_mat::AbstractMatrix{<:Real}`: Singular system matrix.
- `r::AbstractVector{<:Real}`: Right-hand side vector.

# Output
- `Vector{Float64}`: Solution vector.
"""
function _solve_singular_laplacian(L_mat, r)
    m = size(L_mat, 1)
    m == 1 && return zeros(1)
    idx = 2:m
    x   = zeros(m)
    x[idx] = L_mat[idx, idx] \ r[idx]
    return x
end


"""
    (M::SparseMatrixCSC{<:Real, Int})

Checks if the given matrix `M` behaves like a singular Laplacian.

# Arguments
- `M::SparseMatrixCSC{<:Real, Int}`: Matrix to test.

# Output
- `Bool`: True if matrix is singular within tolerance.
"""
function _is_singular_laplacian(M::SparseMatrixCSC)
    v = M * ones(size(M, 1))
    return norm(v, Inf) < 1e-10
end


"""
    prolongation(Partition::Vector{Vector{Int}}, n::Int)

Constructs the prolongation matrix `P` mapping coarse variables to fine ones based on the provided `Partition`.

# Arguments
- `Partition::Vector{Vector{Int}}`: Aggregate vertex partition.
- `n::Int`: Fine-grid dimension size.

# Output
- `SparseMatrixCSC{Float64, Int}`: Prolongation operator matrix.
"""
function prolongation(Partition, n)
    n_c = length(Partition)
    P   = spzeros(n, n_c)
    for i in 1:n_c
        P[Partition[i], i] .= 1
    end
    return P
end

"""
    AMGLevel{Tv, Ti, F1, F2}

Structure holding operators and preallocated workspace arrays for a specific multigrid level.
Eliminates memory allocations during the recursive V/W-cycles.

$FIELDS
"""
struct AMGLevel{Tv, Ti, F1, F2}
    "System matrix"
    A::SparseMatrixCSC{Tv, Ti}          
    "Pre-factorized Pre-Smoother"
    L_fact::F1                          
    "Pre-factorized Post-Smoother"
    U_fact::F2                          
    "Prolongation operator"
    P::SparseMatrixCSC{Float64,Int}     
    "Restriction operator (= P', precomputed)"
    R::SparseMatrixCSC{Float64,Int}     
    "Singular matrix indicator flag"
    singular::Bool
    
    # Workspace vectors (preventing allocations in V-cycle)
    "Workspace vector: fine-grid level dimension"
    work_fine::Vector{Tv}               
    "Workspace vector: coarse-grid level dimension"
    work_coarse::Vector{Tv}             
end


"""
    _safe_lu(M::SparseMatrixCSC{<:Real, Int})

Safe LU factorization: catches singular matrices and applies minimal regularization.

# Arguments
- `M::SparseMatrixCSC{<:Real, Int}`: Input sparse matrix.

# Output
- `LU`: LU factorization object.
"""
function _safe_lu(M::SparseMatrixCSC)
    try
        return lu(M)
    catch
        m = size(M, 1)
        M_reg = copy(M)
        M_reg[1,1] += 1e-12   
        return lu(M_reg)
    end
end

"""
    build_amg_hierarchy(A_fine::AbstractMatrix{<:Real}, Gamma_fine::AbstractGraph; max_coarse_size::Int=50, max_levels::Int=10)

Constructs the full AMG hierarchy (matrices, smoothers, transfer operators) for the given fine-level system `A_fine`.

# Arguments
- `A_fine::AbstractMatrix{<:Real}`: Fine-level system matrix.
- `Gamma_fine::AbstractGraph`: Graph representation for aggregation.
- `max_coarse_size::Int`: Maximum size of the coarsest level (default: 50).
- `max_levels::Int`: Maximum hierarchy depth (default: 10).

# Output
- `Tuple`: `(levels, A_coarsest, coarsest_singular, coarsest_fact)`
"""
function build_amg_hierarchy(A_fine, Gamma_fine;
                             max_coarse_size::Int=50,
                             max_levels::Int=10)

    # Initialize the levels array (consider typing it explicitly like AMGLevel[] later)
    levels = []
    A_current = A_fine
    n_fine = size(A_current, 1)
    
    # Check if the initial system is already small enough
    if n_fine <= max_coarse_size
        println("  Initial system size ($n_fine) <= max_coarse_size ($max_coarse_size). Creating trivial identity level.")

        # Skip the main coarsening loop
        max_levels = 1 
    end

    for ℓ in 1:(max_levels - 1)
        n_current = size(A_current, 1)

        # Stop coarsening if the current system becomes small enough
        if n_current <= max_coarse_size
            break
        end

        # Detect singularity upfront
        is_sing = _is_singular_laplacian(sparse(A_current))

        # ── Factorize smoother ONCE ──
        L_cur = sparse(tril(A_current))
        U_cur = sparse(triu(A_current))
        L_fact = _safe_lu(L_cur)
        U_fact = _safe_lu(U_cur)

        # Partitioning and prolongation
        if ℓ == 1
            Partition = DRA_QC_CE(Gamma_fine)  #  DRA_QC_CE(Gamma_fine)
            P_cur = prolongation(Partition, n_current)
        else
            A_current = (A_current + A_current')/2  
            D_cur = Diagonal(diag(A_current))
            Adj   = D_cur - A_current
            G_cur = SimpleGraph(Adj)
            Partition = DRA_QC_CE(G_cur) #  DRA_QC_CE(Gamma_fine)
            P_cur = prolongation(Partition, n_current)
        end

        P_sparse = sparse(P_cur)
        R_sparse = sparse(P_cur')   # Precompute restriction

        # Galerkin projection
        A_next = R_sparse * A_current * P_sparse
        n_next = size(A_next, 1)

        # Stop: Coarsening stagnates
        if n_next >= n_current * 0.9
            println("  Level $ℓ: Coarsening stagnates ($n_current → $n_next), stopping.")
            break
        end

        # Preallocate workspace
        work_f = zeros(n_current)
        work_c = zeros(n_next)
        
        push!(levels, AMGLevel(A_current, L_fact, U_fact, P_sparse, R_sparse,
                               is_sing, work_f, work_c))
        println("  Level $ℓ → Level $(ℓ+1): $n_current → $n_next$(is_sing ? " [sing.]" : "")")

        A_current = A_next
    end

    # ── Factorize coarsest matrix ONCE ──
    A_coarsest = sparse(A_current)
    coarsest_singular = _is_singular_laplacian(A_coarsest)

    if coarsest_singular
        # Pre-factorize reduced system
        idx = 2:size(A_coarsest, 1)
        A_red = A_coarsest[idx, idx]
        coarsest_fact = lu(A_red)
    else
        coarsest_fact = lu(A_coarsest)
    end

    println("  Coarsest Level ($(length(levels)+1)): $(size(A_coarsest,1))$(coarsest_singular ? " [sing.]" : "")")
    println("  Total: $(length(levels)+1) Level")

    return levels, A_coarsest, coarsest_singular, coarsest_fact
end


"""
    _solve_coarsest!(x::Vector{Float64}, r::Vector{Float64}, A_coarsest::AbstractMatrix{<:Real}, coarsest_singular::Bool, coarsest_fact)

Solves the system accurately on the coarsest level using the pre-computed LU factorization.

# Arguments
- `x::Vector{Float64}`: Solution vector to populate.
- `r::Vector{Float64}`: Right-hand side vector.
- `A_coarsest::AbstractMatrix{<:Real}`: Coarsest system matrix.
- `coarsest_singular::Bool`: Singularity flag.
- `coarsest_fact`: Precomputed LU factorization.

# Returns
- `Vector{Float64}`: Solved vector `x`.
"""
function _solve_coarsest!(x, r, A_coarsest, coarsest_singular, coarsest_fact)
    if coarsest_singular
        m = size(A_coarsest, 1)
        fill!(x, 0.0)
        idx = 2:m
        @views x[idx] .= coarsest_fact \ r[idx]
    else
        x .= coarsest_fact \ r
    end
    return x
end

"""
    mg_preconditioner!(v_out::Vector{Float64}, levels, A_coarsest, coarsest_singular, coarsest_fact, r::Vector{Float64}, cl::Int; inner_iter::Int=2)

Executes a single V-cycle, applying pre-smoothing, restriction, coarse-grid correction, prolongation, and post-smoothing recursively.

# Arguments
- `v_out::Vector{Float64}`: Output vector for the correction.
- `levels`: AMG level vector.
- `A_coarsest`: Coarsest system matrix.
- `coarsest_singular::Bool`: Singularity indicator.
- `coarsest_fact`: Coarsest solver factorisation.
- `r::Vector{Float64}`: Right-hand side residual vector.
- `cl::Int`: Current level index.
- `inner_iter::Int`: Inner FCG iterations (default: 2).

# Returns
- `Vector{Float64}`: Computed preconditioner application `v_out`.
"""
function mg_preconditioner!(v_out::Vector{Float64},
                            levels,
                            A_coarsest, coarsest_singular, coarsest_fact,
                            r::Vector{Float64}, cl::Int;
                            inner_iter::Int=2)

    lvl = levels[cl]
    is_last = (cl == length(levels))

    v1 = lvl.L_fact \ r

    mul!(lvl.work_fine, lvl.A, v1)     
    lvl.work_fine .= r .- lvl.work_fine 

    mul!(lvl.work_coarse, lvl.R, lvl.work_fine)  

    if is_last
        v_c = similar(lvl.work_coarse)
        _solve_coarsest!(v_c, lvl.work_coarse, A_coarsest, coarsest_singular, coarsest_fact)
    else
        v_c, _ = fcg_solve(levels, A_coarsest, coarsest_singular, coarsest_fact,
                           lvl.work_coarse, cl + 1;
                           maxiter=inner_iter, tol=1e-12, inner_iter=inner_iter)
    end

    v2 = lvl.P * v_c

    res2 = lvl.A * v2              
    res2 .= lvl.work_fine .- res2  

    v3 = lvl.U_fact \ res2

    v_out .= v1 .+ v2 .+ v3

    return v_out
end

"""
    fcg_solve(levels, A_coarsest, coarsest_singular::Bool, coarsest_fact, b::Vector{Float64}, current_level::Int; tol::Float64=1e-6, maxiter::Int=10000, inner_iter::Int=2)

Solves the linear system using the Flexible Conjugate Gradient (FCG) method preconditioned by the AMG V-cycle.

# Arguments
- `levels`: AMG level vector.
- `A_coarsest`: Coarsest system matrix.
- `coarsest_singular::Bool`: Singularity indicator.
- `coarsest_fact`: Coarsest factorisation.
- `b::Vector{Float64}`: Right-hand side vector.
- `current_level::Int`: Starting level index.
- `tol::Float64`: Convergence tolerance (default: 1e-6).
- `maxiter::Int`: Maximum iterations (default: 10000).
- `inner_iter::Int`: Inner multigrid iterations (default: 2).

# Returns
- `Tuple{Vector{Float64}, Int}`: Solution vector `x` and total iteration count.
"""
function fcg_solve(levels, A_coarsest,
                   coarsest_singular::Bool, coarsest_fact,
                   b::Vector{Float64}, current_level::Int;
                   tol::Float64=1e-6, maxiter::Int=10_000,
                   inner_iter::Int=2)

    A = levels[current_level].A
    n = length(b)

    x     = zeros(n)
    r     = copy(b)
    z     = zeros(n)
    p     = zeros(n)
    Ap    = zeros(n)
    r_old = zeros(n)
    z_old = zeros(n)

    iter = 0
    for k in 1:maxiter
        iter += 1

        # z = M⁻¹ r
        mg_preconditioner!(z, levels, A_coarsest, coarsest_singular, coarsest_fact,
                           r, current_level; inner_iter=inner_iter)

        if k > 1
            num = dot(z, r) - dot(z, r_old)
            den = dot(z_old, r_old)
            beta = num / den
        else
            beta = 0.0
        end

        p .= z .+ beta .* p

        mul!(Ap, A, p)

        alpha = dot(p, r) / dot(p, Ap)

        x .+= alpha .* p

        r_old .= r
        z_old .= z
        r .-= alpha .* Ap

        if norm(r, Inf) <= tol
            @info "AMG-FPCG converged in $k iterations with residual norm $(norm(r, Inf))"
            break
        end
    end

    return x, iter
end

"""
    amg_solve(levels, A_coarsest, coarsest_singular::Bool, coarsest_fact, b::Vector{Float64}; tol::Float64=1e-8, maxiter::Int=10000, inner_iter::Int=2)

Main user-facing function to trigger the AMG-preconditioned solver.

# Arguments
- `levels`: AMG level vector.
- `A_coarsest`: Coarsest system matrix.
- `coarsest_singular::Bool`: Singularity indicator.
- `coarsest_fact`: Coarsest solver factorisation.
- `b::Vector{Float64}`: Right-hand side vector.
- `tol::Float64`: Tolerance (default: 1e-8).
- `maxiter::Int`: Maximum iterations (default: 10000).
- `inner_iter::Int`: Inner iterations (default: 2).

# Returns
- `Tuple{Vector{Float64}, Int}`: Solution vector and iteration count.
"""
function amg_solve(levels, A_coarsest,
                   coarsest_singular::Bool, coarsest_fact,
                   b::Vector{Float64};
                   tol::Float64=1e-8, maxiter::Int=10_000,
                   inner_iter::Int=2)
    return fcg_solve(levels, A_coarsest, coarsest_singular, coarsest_fact, b, 1;
                     tol=tol, maxiter=maxiter, inner_iter=inner_iter)
end