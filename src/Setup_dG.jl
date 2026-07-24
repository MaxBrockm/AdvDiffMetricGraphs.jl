"""
    createH_DG_AdvectionDiffusion(G::Graph, Levels::AbstractVector{Int}, edge_x::AbstractVector{<:AbstractVector{<:Float64}}, D_edge::AbstractVector{<:Float64}, A_edge::AbstractVector{<:Float64}, f_exact::Function; eta::Real=10.0, p::Int=1, use_SUPG::Bool=false)

Assembles the system matrices and right-hand side vectors for an advection-diffusion problem on a metric graph using a Discontinuous Galerkin (DG) method.

# Arguments
- `G::Graph`: The graph topology.
- `Levels::AbstractVector{Int}`: The refinement levels for each edge, determining the number of elements.
- `edge_x::AbstractVector{<:AbstractVector{<:Float64}}`: The spatial grid points for each edge.
- `D_edge::AbstractVector{<:Float64}`: The diffusion coefficients for each edge.
- `A_edge::AbstractVector{<:Float64}`: The advection coefficients for each edge.
- `f_exact::Function`: The exact source function (right-hand side) evaluated at specific points.
- `eta::Real`: The interior penalty parameter for the Interior Penalty (default is 10.0).
- `p::Int`: The polynomial degree used in the discretization (default is 1).
- `use_SUPG::Bool`: Flag indicating whether SUPG stabilization should be utilized (default is false).

# Output
- `HEE::SparseMatrixCSC`: The edge-edge block of the assembled system matrix.
- `HEV::SparseMatrixCSC`: The edge-vertex block of the assembled system matrix.
- `HVE::SparseMatrixCSC`: The vertex-edge block of the assembled system matrix.
- `HVV::SparseMatrixCSC`: The vertex-vertex block of the assembled system matrix.
- `f_E::Vector{Float64}`: The assembled right-hand side vector for the internal edge degrees of freedom.
- `f_V::Vector{Float64}`: The assembled right-hand side vector for the vertex degrees of freedom.
"""
function createH_DG_AdvectionDiffusion(G::Graph, Levels::AbstractVector{Int},
                                       edge_x::AbstractVector{<:AbstractVector{<:Float64}},
                                       D_edge::AbstractVector{<:Float64},
                                       A_edge::AbstractVector{<:Float64},
                                       f_exact::Function;
                                       eta::Real=10.0, p::Int=1, use_SUPG::Bool=false)
    NV = nv(G)
    M = ne(G)

    N_elements = 2 .^ Levels
    int_dofs = 2 .* N_elements 
    NE = sum(int_dofs)

    edge_offsets = zeros(Int, M)
    curr_offset = 0
    for e in 1:M
        edge_offsets[e] = curr_offset
        curr_offset += int_dofs[e]
    end

    I_EE = Int[]; J_EE = Int[]; V_EE = Float64[]
    I_EV = Int[]; J_EV = Int[]; V_EV = Float64[]
    I_VE = Int[]; J_VE = Int[]; V_VE = Float64[]
    I_VV = Int[]; J_VV = Int[]; V_VV = Float64[]

    sizehint!(I_EE, 6*NE); sizehint!(I_EV, 4*M); sizehint!(I_VE, 4*M); sizehint!(I_VV, 4*M)
    
    f_E = zeros(NE)
    f_V = zeros(NV)

    @inline function push_block!(t1::Symbol, i1::Int, t2::Symbol, i2::Int, val::Float64)
        val == 0.0 && return nothing
        if t1 == :E && t2 == :E
            push!(I_EE, i1); push!(J_EE, i2); push!(V_EE, val)
        elseif t1 == :E && t2 == :V
            push!(I_EV, i1); push!(J_EV, i2); push!(V_EV, val)
        elseif t1 == :V && t2 == :E
            push!(I_VE, i1); push!(J_VE, i2); push!(V_VE, val)
        elseif t1 == :V && t2 == :V
            push!(I_VV, i1); push!(J_VV, i2); push!(V_VV, val)
        end
        return nothing
    end

    function add_sipg_interface!(dofsL, dofsR, hL::Float64, hR::Float64, D::Float64)
        valL = (0.0, 1.0); derL = (-1.0/hL, 1.0/hL)
        valR = (1.0, 0.0); derR = (-1.0/hR, 1.0/hR)
        h_avg = 0.5*(hL + hR)
        pen = float(eta) * (p+1)^2 * D / max(h_avg, 1e-15)

        for s in (:L, :R)
            for i in 1:2
                test_dof = s==:L ? dofsL[i] : dofsR[i]
                jump_v = s==:L ? valL[i] : -valR[i]
                avgDvprime = 0.5 * D * (s==:L ? derL[i] : derR[i])

                for t in (:L, :R)
                    for j in 1:2
                        trial_dof = t==:L ? dofsL[j] : dofsR[j]
                        avgDuprime = 0.5 * D * (t==:L ? derL[j] : derR[j])
                        jump_u = t==:L ? valL[j] : -valR[j]

                        aij = (-avgDuprime * jump_v) + (-avgDvprime * jump_u) + (pen * jump_u * jump_v)
                        push_block!(test_dof[1], test_dof[2], trial_dof[1], trial_dof[2], aij)
                    end
                end
            end
        end
    end

    function add_upwind_interface!(dofsL, dofsR, A::Float64)
        abs(A) < 1e-14 && return
        if A > 0
            push_block!(dofsL[2][1], dofsL[2][2], dofsL[2][1], dofsL[2][2],  A)
            push_block!(dofsR[1][1], dofsR[1][2], dofsL[2][1], dofsL[2][2], -A)
        else
            push_block!(dofsL[2][1], dofsL[2][2], dofsR[1][1], dofsR[1][2],  A)
            push_block!(dofsR[1][1], dofsR[1][2], dofsR[1][1], dofsR[1][2], -A)
        end
    end

    function add_vertex_coupling!(elem_dofs, dof_V, D::Float64, A::Float64, n_e::Float64, h::Float64)
        dofs = [elem_dofs[1], elem_dofs[2], dof_V]
        
        val_e = (n_e == -1.0) ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
        der_n = (n_e == -1.0) ? [1.0/h, -1.0/h, 0.0] : [-1.0/h, 1.0/h, 0.0]
        val_V = [0.0, 0.0, 1.0]
        
        J = val_e .- val_V 
        
        pen = float(eta) * (p+1)^2 * D / h
        
        c_e = A * n_e
        
        for (i, test_dof) in enumerate(dofs)
            for (j, trial_dof) in enumerate(dofs)
                aij = - (D * der_n[j]) * J[i] - (D * der_n[i]) * J[j] + pen * J[i] * J[j]
                
                if c_e > 0
                    aij += c_e * val_e[j] * J[i]
                elseif c_e < 0
                    aij += c_e * val_V[j] * J[i]
                end
                
                push_block!(test_dof[1], test_dof[2], trial_dof[1], trial_dof[2], aij)
            end
        end
    end

    for (m, edge) in enumerate(edges(G))
        u_node = src(edge)
        v_node = dst(edge)

        D = float(D_edge[m])
        A = float(A_edge[m])
        N_e = N_elements[m]
        x_nodes = edge_x[m] 

        elem_dofs = Vector{Tuple{Tuple{Symbol,Int}, Tuple{Symbol,Int}}}(undef, N_e)
        offset = edge_offsets[m]
        
        for k in 1:N_e
            dL = (:E, offset + 2*k - 1)
            dR = (:E, offset + 2*k)
            elem_dofs[k] = (dL, dR)
        end

        for k in 1:N_e
            dL, dR = elem_dofs[k]
            x1 = x_nodes[k]
            x2 = x_nodes[k+1]
            hk = x2 - x1 

            tau = 0.0 
            if use_SUPG
                if (A > 0 && k == N_e) || (A < 0 && k == 1)
                    tau = tau_supg(A, D, hk)
                end
            end

            Ke = local_stiff(D, A, hk, tau) 

            push_block!(dL[1], dL[2], dL[1], dL[2], Ke[1,1])
            push_block!(dL[1], dL[2], dR[1], dR[2], Ke[1,2])
            push_block!(dR[1], dR[2], dL[1], dL[2], Ke[2,1])
            push_block!(dR[1], dR[2], dR[1], dR[2], Ke[2,2])

            f_func(x) = f_exact(x, m) 
            b1, b2 = local_load(f_func, x1, x2, A, tau) 

            f_E[dL[2]] += b1
            f_E[dR[2]] += b2
        end

        for k in 1:(N_e - 1)
            dofsL = elem_dofs[k]
            dofsR = elem_dofs[k+1]
            
            hk_L = x_nodes[k+1] - x_nodes[k]
            hk_R = x_nodes[k+2] - x_nodes[k+1]
            
            add_sipg_interface!(dofsL, dofsR, hk_L, hk_R, D)
            add_upwind_interface!(dofsL, dofsR, A)
        end

        h_start = x_nodes[2] - x_nodes[1]
        add_vertex_coupling!(elem_dofs[1], (:V, u_node), D, A, -1.0, h_start)

        h_end = x_nodes[end] - x_nodes[end-1]
        add_vertex_coupling!(elem_dofs[N_e], (:V, v_node), D, A, 1.0, h_end)
    end

    HEE = sparse(I_EE, J_EE, V_EE, NE, NE)
    HEV = sparse(I_EV, J_EV, V_EV, NE, NV)
    HVE = sparse(I_VE, J_VE, V_VE, NV, NE)
    HVV = sparse(I_VV, J_VV, V_VV, NV, NV)

    return HEE, HEV, HVE, HVV, f_E, f_V
end