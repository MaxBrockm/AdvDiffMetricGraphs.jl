"""
    shishkin_grid(L::Real, N::Integer, A::Real, D::Real; ρ=2.0, mode=:symmetric, cap=0.25, pe_switch=1.0)

Generates `N+1` node points for a Shishkin mesh on the interval `[0, L]`.
Falls back to a uniform mesh if the system is diffusion-dominated or advection is vanishing. Half of the discretization points are allocated to the boundary layer region. The boundary layer width is defined via the transition point `τ = min(cap*L, ρ*(D/|A|)*log(N))`, where `ρ` is a user-defined factor (default: `2.0`).

# Arguments
- `L::Real`: Length of the interval.
- `N::Integer`: Number of elements (must be ≥ 2).
- `A::Real`: Advection coefficient.
- `D::Real`: Diffusion coefficient.
- `ρ::Real=2.0`: Shishkin transition parameter factor.
- `mode::Symbol=:symmetric`: Mesh generation mode, either `:symmetric` or `:outflow`.
- `cap::Real=0.25`: Maximum fraction of the interval used for the boundary layer.
- `pe_switch::Real=1.0`: Global Péclet number threshold below which a uniform mesh is returned.

# Output
- `Vector{Float64}`: Array containing the generated `N+1` grid nodes.
"""
function shishkin_grid(L::Real, N::Integer, A::Real, D::Real;
                       ρ::Real=2.0, mode::Symbol=:symmetric, cap::Real=0.25, pe_switch::Real=1.0)
    L = float(L); A = float(A); D = float(D)
    N ≥ 2 || error("Shishkin mesh requires N≥2 elements")

    if D <= 0 || abs(A) < 1e-14
        return collect(range(0.0, L; length=N+1))
    end

    Pe_global = abs(A)*L/(2D)
    if Pe_global ≤ pe_switch
        return collect(range(0.0, L; length=N+1))
    end

    β = abs(A);
    τ = min(cap*L, ρ*(D/β)*log(float(N)))
    τ = max(τ, 0.0)
    
    if τ ≤ 1e-15
        return collect(range(0.0, L; length=N+1))
    end

    if mode == :symmetric
        N1 = N ÷ 4
        if N1 == 0
            return collect(range(0.0, L; length=N+1))
        end
        N2 = N - 2N1
        τ = min(τ, 0.5L - 1e-15)

        x1 = collect(range(0.0, τ;   length=N1+1))
        x2 = collect(range(τ, L-τ;  length=N2+1))[2:end]
        x3 = collect(range(L-τ, L;  length=N1+1))[2:end]
        return vcat(x1, x2, x3)

    elseif mode == :outflow
        Nfine   = N ÷ 2
        Ncoarse = N - Nfine
        if Nfine == 0 || Ncoarse == 0
            return collect(range(0.0, L; length=N+1))
        end
        τ = min(τ, L - 1e-15)
        if A > 0
            x_coarse = collect(range(0.0, L-τ; length=Ncoarse+1))
            x_fine   = collect(range(L-τ, L;  length=Nfine+1))[2:end]
            return vcat(x_coarse, x_fine)
        else
            x_fine   = collect(range(0.0, τ;  length=Nfine+1))
            x_coarse = collect(range(τ, L;    length=Ncoarse+1))[2:end]
            return vcat(x_fine, x_coarse)
        end
    else
        error("mode must be :symmetric or :outflow")
    end
end


"""
    (L::Real, N::Int, eps::Real, a_mag::Real; side=:right, sigma_factor=2.0, pe_switch=1.0)

Generates Bakhvalov-type nodes on `[0, L]` with `N` elements.
Falls back to a uniform mesh if the global Péclet number is small or the system degenerates.

# Arguments
- `L::Real`: Length of the interval.
- `N::Int`: Number of elements (must be ≥ 2).
- `eps::Real`: Diffusion scale coefficient.
- `a_mag::Real`: Advection scale coefficient.
- `side::Symbol=:right`: Specifies where the boundary layer is located (`:left`, `:right`, or `:both`).
- `sigma_factor::Real=2.0`: Transition parameter multiplier.
- `pe_switch::Real=1.0`: Global Péclet number threshold below which a uniform mesh is returned.

# Output
- `Vector{Float64}`: Array containing the generated `N+1` grid nodes.
"""
function bakhvalov_nodes(L::Real, N::Int, eps::Real, a_mag::Real;
                         side::Symbol=:right,
                         sigma_factor::Real=2.0,
                         pe_switch::Real=1.0)

    L   = float(L)
    eps = float(eps)
    a   = max(abs(float(a_mag)), 1e-14)

    N ≥ 2 || error("Bakhvalov mesh requires N≥2 elements")

    if eps <= 0 || a < 1e-14
        return collect(range(0.0, L; length=N+1))
    end

    Pe_global = a*L/(2eps)
    if Pe_global ≤ pe_switch
        return collect(range(0.0, L; length=N+1))
    end

    graded_segment(σ::Float64, Nh::Int) = begin
        σ ≤ 1e-15 && return collect(range(0.0, σ; length=Nh+1))
        z = a*σ/eps                   
        t = range(0.0, 1.0; length=Nh+1)
        return -(eps/a) .* log1p.(t .* expm1(-z))
    end

    if side == :both
        N = (N % 4 == 0) ? N : (N + (4 - N%4))      
        σ = min(0.25L, sigma_factor*eps*log(float(N))/a)
        if σ ≤ 1e-15
            return collect(range(0.0, L; length=N+1))
        end

        Nq = div(N,4)         
        Nm = div(N,2)         

        left = graded_segment(σ, Nq)              
        mid  = collect(range(σ, L-σ; length=Nm+1))[2:end]  
        right_base = graded_segment(σ, Nq)                 
        right = (L-σ) .+ right_base[2:end]                
        return vcat(left, mid, right)

    else
        N = (N % 2 == 0) ? N : (N + 1)        
        σ = min(0.5L, sigma_factor*eps*log(float(N))/a)
        if σ ≤ 1e-15
            return collect(range(0.0, L; length=N+1))
        end

        Nh = div(N,2)                           
        fine = graded_segment(σ, Nh)            

        if side == :right
            coarse = collect(range(0.0, L-σ; length=Nh+1))         
            fine_shifted = (L-σ) .+ fine[2:end]                     
            return vcat(coarse, fine_shifted)

        elseif side == :left
            fine_left = fine                                       
            coarse = collect(range(σ, L; length=Nh+1))[2:end]      
            return vcat(fine_left, coarse)

        else
            error("side must be :left, :right or :both")
        end
    end
end


"""
    local_stiff_std(D::Float64, A::Float64, h_loc::Float64)

Computes the local stiffness matrix for the advection diffusion equation without stabilization.

# Arguments
- `D::Float64`: Diffusion coefficient.
- `A::Float64`: Advection coefficient.
- `h_loc::Float64`: Local element size (width).

# Output
- `Matrix{Float64}`: The `2 x 2` local stiffness matrix.
"""
function local_stiff_std(D::Float64, A::Float64, h_loc::Float64)
    # reine Diffusion + reine Advektion
    return (D/h_loc) * [1.0 -1.0; -1.0  1.0] + (A/2.0) * [ 1.0  1.0; -1.0 -1.0]
end

"""
    local_load_std(f::Function, x1::Float64, x2::Float64)

Computes the local load vector using linear hat functions.

# Arguments
- `f::Function`: The right-hand side source function.
- `x1::Float64`: Left coordinate of the local element.
- `x2::Float64`: Right coordinate of the local element.

# Output
- `Tuple{Float64, Float64}`: A tuple containing the two local load vector entries.
"""
function local_load_std(f::Function, x1::Float64, x2::Float64)
    h_loc = x2 - x1
    φ1(x) = (x2 - x) / h_loc
    φ2(x) = (x - x1) / h_loc
    
    b1, _ = quadgk(x -> f(x) * φ1(x), x1, x2; rtol=1e-10, atol=1e-12)
    b2, _ = quadgk(x -> f(x) * φ2(x), x1, x2; rtol=1e-10, atol=1e-12)
    
    return (float(b1), float(b2))
end


"""
    assemble_system_blocks_shishkin(nv::Int, edge_dofs::AbstractDict, edge_x::AbstractDict, D_edge::AbstractDict, A_edge::AbstractDict, Rhs_edge::AbstractDict; edges_ordered=EDGES_ORDERED)

Assembles the global system into four block matrices (`HEE`, `HEV`, `HVE`, `HVV`) and block right-hand side vectors for a given graph and Shishkin mesh discretization.

# Arguments
- `nv::Int`: Number of vertices in the graph.
- `edge_dofs::AbstractDict`: Dictionary mapping edges to their global degree-of-freedom indices.
- `edge_x::AbstractDict`: Dictionary mapping edges to their node coordinates.
- `D_edge::AbstractDict`: Dictionary mapping edges to their diffusion coefficients.
- `A_edge::AbstractDict`: Dictionary mapping edges to their advection coefficients.
- `Rhs_edge::AbstractDict`: Dictionary mapping edges to their right-hand side functions.
- `edges_ordered`: An ordered collection of the graph's edges (default: `EDGES_ORDERED`).

# Output
- `Tuple`: Contains `(HEE, HEV, HVE, HVV, f_E, f_V)`, where `HEE`, `HEV`, `HVE`, `HVV` are the sparse block matrices, and `f_E`, `f_V` are the right-hand side vectors for edges and vertices respectively.
"""
function assemble_system_blocks_shishkin(nv::Int, edge_dofs::AbstractDict, edge_x::AbstractDict, 
                                D_edge::AbstractDict, A_edge::AbstractDict, Rhs_edge::AbstractDict;
                                edges_ordered=EDGES_ORDERED)
    
    Ntot = maximum(vcat(values(edge_dofs)...))
    NE = Ntot - nv
    
    I_EE = Int[]; J_EE = Int[]; V_EE = Float64[]
    I_EV = Int[]; J_EV = Int[]; V_EV = Float64[]
    I_VE = Int[]; J_VE = Int[]; V_VE = Float64[]
    I_VV = Int[]; J_VV = Int[]; V_VV = Float64[]
    
    sizehint!(I_EE, 3*NE); sizehint!(I_EV, 2*NE); sizehint!(I_VE, 2*NE); sizehint!(I_VV, 4*nv)
    
    f_E = zeros(Float64, NE)
    f_V = zeros(Float64, nv)
    
    @inline function push_block!(n1::Int, n2::Int, val::Float64)
        val == 0.0 && return nothing
        
        is_V1 = (n1 <= nv)
        is_V2 = (n2 <= nv)
        
        i1 = is_V1 ? n1 : (n1 - nv)
        i2 = is_V2 ? n2 : (n2 - nv)
        
        if !is_V1 && !is_V2      
            push!(I_EE, i1); push!(J_EE, i2); push!(V_EE, val)
        elseif !is_V1 && is_V2   
            push!(I_EV, i1); push!(J_EV, i2); push!(V_EV, val)
        elseif is_V1 && !is_V2    
            push!(I_VE, i1); push!(J_VE, i2); push!(V_VE, val)
        else                      
            push!(I_VV, i1); push!(J_VV, i2); push!(V_VV, val)
        end
        return nothing
    end

    for e in edges_ordered
        D = float(D_edge[e])
        A = float(A_edge[e])
        nodes = edge_dofs[e]
        x = edge_x[e]
        f_e = Rhs_edge[e]

        ne_local = length(nodes) - 1
        @assert length(x) == length(nodes)

        for k in 1:ne_local
            n1, n2 = nodes[k], nodes[k+1]
            x1, x2 = float(x[k]), float(x[k+1])
            h_loc  = x2 - x1

            Ke = local_stiff(D, A, h_loc, 0.0)
            b1, b2 = local_load(f_e, x1, x2, A, 0.0)

            push_block!(n1, n1, Ke[1,1])
            push_block!(n1, n2, Ke[1,2])
            push_block!(n2, n1, Ke[2,1])
            push_block!(n2, n2, Ke[2,2])

            if n1 <= nv
                f_V[n1] += b1
            else
                f_E[n1 - nv] += b1
            end

            if n2 <= nv
                f_V[n2] += b2
            else
                f_E[n2 - nv] += b2
            end
        end
    end

    HEE = sparse(I_EE, J_EE, V_EE, NE, NE)
    HEV = sparse(I_EV, J_EV, V_EV, NE, nv)
    HVE = sparse(I_VE, J_VE, V_VE, nv, NE)
    HVV = sparse(I_VV, J_VV, V_VV, nv, nv)

    return HEE, HEV, HVE, HVV, f_E, f_V
end