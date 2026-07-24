# BSpline Implementation based on Nurbs Book
"""
    get_edge_coeff(coeff, e, ei::Int)

Helper function to extract or evaluate edge coefficients (e.g., diffusion or advection) 
dynamically, supporting multiple container types.

# Arguments
- `coeff`: The coefficient container (Number, AbstractVector, Dict, or Function).
- `e`: The edge object.
- `ei::Int`: The index of the edge.

# Output:
- `Float64`: The evaluated coefficient for the given edge.
"""
@inline function get_edge_coeff(coeff, e, ei::Int)
    if coeff isa Number
        return float(coeff)
    elseif coeff isa AbstractVector
        return float(coeff[ei])
    elseif coeff isa Dict
        return float(coeff[e])
    elseif coeff isa Function
        return float(coeff(e, ei))
    else
        error("Unsupported coeff container: $(typeof(coeff))")
    end
end


"""
    eval_f(f_edge, e, ei::Int, x::Float64)

Evaluates the right-hand side function `f` on a specific edge.

# Arguments
- `f_edge`: The function container.
- `e`: The edge object.
- `ei::Int`: The edge index.
- `x::Float64`: The local coordinate on the edge.

# Output:
- `Float64`: The evaluated function value.
"""
@inline function eval_f(f_edge, e, ei::Int, x::Float64)
    if f_edge === nothing
        return 0.0
    elseif f_edge isa Function
        try
            return float(f_edge(x, e, ei))
        catch
            return float(f_edge(x))
        end
    elseif f_edge isa Dict
        return float(f_edge[e](x))
    else
        error("Unsupported f_edge container: $(typeof(f_edge))")
    end
end


"""
    get_g(g_vertex, v::Int)

Retrieves the boundary or source value `g` at a specific vertex.

# Arguments
- `g_vertex`: The boundary value container.
- `v::Int`: The vertex index.

# Output:
- `Float64`: The boundary value at the vertex.
"""
@inline function get_g(g_vertex, v::Int)
    if g_vertex === nothing
        return 0.0
    elseif g_vertex isa AbstractVector
        return float(g_vertex[v])
    elseif g_vertex isa Dict
        return float(get(g_vertex, v, 0.0))
    elseif g_vertex isa Function
        return float(g_vertex(v))
    else
        error("Unsupported g_vertex container: $(typeof(g_vertex))")
    end
end


"""
    generate_knot_vector_from_breaks(breaks::AbstractVector{<:Real}, p::Int)

Builds an open knot vector accommodating arbitrary breaks for B-splines of degree `p`.

# Arguments
- `breaks::AbstractVector{<:Real}`: The unique break points of the grid.
- `p::Int`: The polynomial degree.

# Output:
- `Vector{Float64}`: The constructed open knot vector.
"""
function generate_knot_vector_from_breaks(breaks::AbstractVector{<:Real}, p::Int)
    T = Float64[]
    a = Float64(first(breaks))
    b = Float64(last(breaks))
    for _ in 1:p; push!(T, a); end
    for x in breaks; push!(T, Float64(x)); end
    for _ in 1:p; push!(T, b); end
    return T
end


"""
    (T::AbstractVector{<:Real}, p::Int, n_basis::Int)

Calculates the Greville abscissae (collocation points) for a given knot vector.

# Arguments
- `T::AbstractVector{<:Real}`: The knot vector.
- `p::Int`: The polynomial degree.
- `n_basis::Int`: The number of basis functions.

# Output:
- `Vector{Float64}`: The computed Greville points.
"""
function greville_points(T::AbstractVector{<:Real}, p::Int, n_basis::Int)
    pts = zeros(Float64, n_basis)
    for i in 1:n_basis
        s = 0.0
        for j in 1:p; s += T[i+j]; end
        pts[i] = s / p
    end
    return pts
end


"""
    find_span(nloc::Int, p::Int, u::Float64, U::AbstractVector{<:Real})

Finds the knot span index such that `u` lies within `[U[span], U[span+1])`.

# Arguments
- `nloc::Int`: Number of local basis functions.
- `p::Int`: Polynomial degree.
- `u::Float64`: Evaluation parameter.
- `U::AbstractVector{<:Real}`: The knot vector.

# Output:
- `Int`: The active knot span index.
"""
function find_span(nloc::Int, p::Int, u::Float64, U::AbstractVector{<:Real})
    if u >= float(U[end])
        return nloc
    end
    low  = p + 1
    high = nloc + 1
    mid  = (low + high) >>> 1
    while (u < float(U[mid])) || (u >= float(U[mid+1]))
        if u < float(U[mid])
            high = mid
        else
            low = mid
        end
        mid = (low + high) >>> 1
    end
    return mid
end


"""
    ders_basis_funs(span::Int, u::Float64, p::Int, nder::Int, U::AbstractVector{<:Real})

Computes the non-zero B-spline basis functions and their derivatives up to order `nder` at a given point `u`.

# Arguments
- `span::Int`: The knot span index.
- `u::Float64`: The evaluation point.
- `p::Int`: The polynomial degree.
- `nder::Int`: The maximum derivative order to compute.
- `U::AbstractVector{<:Real}`: The knot vector.

# Output:
- `Matrix{Float64}`: A matrix of size `(nder + 1) × (p + 1)` containing the derivatives.
"""
function ders_basis_funs(span::Int, u::Float64, p::Int, nder::Int, U::AbstractVector{<:Real})
    ndu  = zeros(Float64, p+1, p+1)
    left = zeros(Float64, p+1)
    right= zeros(Float64, p+1)

    ndu[1,1] = 1.0
    for j in 1:p
        left[j+1]  = u - float(U[span+1-j])
        right[j+1] = float(U[span+j]) - u
        saved = 0.0
        for r in 0:j-1
            ndu[j+1, r+1] = right[r+2] + left[j-r+1]
            temp = ndu[r+1, j] / ndu[j+1, r+1]
            ndu[r+1, j+1] = saved + right[r+2]*temp
            saved = left[j-r+1]*temp
        end
        ndu[j+1, j+1] = saved
    end

    ders = zeros(Float64, nder+1, p+1)
    for j in 0:p
        ders[1, j+1] = ndu[j+1, p+1]
    end

    if nder == 0
        return ders
    end

    a = zeros(Float64, 2, p+1)
    for r in 0:p
        a[1,1] = 1.0
        s1, s2 = 1, 2
        for k in 1:nder
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k
                a[s2,1] = a[s1,1] / ndu[pk+2, rk+1]
                d = a[s2,1]*ndu[rk+1, pk+1]
            end
            j1 = (rk >= -1) ? 1 : -rk
            j2 = (r-1 <= pk) ? (k-1) : (p-r)
            for j in j1:j2
                a[s2, j+1] = (a[s1, j+1] - a[s1, j]) / ndu[pk+2, rk+j+1]
                d += a[s2, j+1]*ndu[rk+j+1, pk+1]
            end
            if r <= pk
                a[s2, k+1] = -a[s1, k] / ndu[pk+2, r+1]
                d += a[s2, k+1]*ndu[r+1, pk+1]
            end
            ders[k+1, r+1] = d
            s1, s2 = s2, s1
        end
    end

    # Factorial scaling: k-th derivative gets p*(p-1)*...*(p-k+1)
    rfac = p
    for k in 1:nder
        for j in 0:p
            ders[k+1, j+1] *= rfac
        end
        rfac *= (p-k)
    end

    return ders
end


"""
    assemble_bspline_collocation(edges_ordered, edge_x::Dict; nv::Int, p::Int, A_edge, D_edge, f_edge, dirichlet_nodes::Vector{Int}, g_vertex, n_sign::Function=normal_sign)

Assembles the global B-spline collocation system matrix and right-hand side vector using a fast local support formulation.

# Arguments
- `edges_ordered`: The collection of edges in the graph.
- `edge_x::Dict`: Dictionary containing the physical break points per edge.
- `nv::Int`: Number of vertices.
- `p::Int`: B-spline polynomial degree.
- `A_edge`, `D_edge`: Advection and diffusion coefficients per edge.
- `f_edge`: Right-hand side forcing function per edge.
- `dirichlet_nodes::Vector{Int}`: Vertices designated as Dirichlet boundaries.
- `g_vertex`: Dirichlet boundary values.
- `n_sign::Function`: Function defining the outward normal signs.

# Output:
- `Tuple{SparseMatrixCSC, Vector{Float64}, NamedTuple}`: The global stiffness matrix, 
  the right-hand side vector, and a metadata tuple for evaluation.
"""
function assemble_bspline_collocation(edges_ordered, edge_x::Dict;
                                      nv::Int,
                                      p::Int,
                                      A_edge,
                                      D_edge,
                                      f_edge,
                                      dirichlet_nodes::Vector{Int},
                                      g_vertex,
                                      n_sign::Function=normal_sign)

    dirset = Set(dirichlet_nodes)

    incident = [Vector{typeof(first(edges_ordered))}() for _ in 1:nv]
    for e in edges_ordered
        push!(incident[src(e)], e)
        push!(incident[dst(e)], e)
    end

    Te     = Dict{typeof(first(edges_ordered)), Vector{Float64}}()
    nloc_e = Dict{typeof(first(edges_ordered)), Int}()
    offset = Dict{typeof(first(edges_ordered)), Int}()

    ndof = 0
    for e in edges_ordered
        br = edge_x[e]
        T = generate_knot_vector_from_breaks(br, p)
        nloc = length(br) - 1 + p
        
        Te[e] = T
        nloc_e[e] = nloc
        offset[e] = ndof + 1
        ndof += nloc
    end

    # Pre-allocate slightly larger arrays for SparseMatrixCSC
    expected_nnz = ndof * (p + 1)
    I = sizehint!(Int[], expected_nnz)
    J = sizehint!(Int[], expected_nnz)
    V = sizehint!(Float64[], expected_nnz)
    
    B = zeros(Float64, ndof)
    row = 0

    # PDE collocation per edge at Greville points i=2..n-1
    for (ei, e) in enumerate(edges_ordered)
        T = Te[e]; nloc = nloc_e[e]; base = offset[e]
        A = get_edge_coeff(A_edge, e, ei)
        D = get_edge_coeff(D_edge, e, ei)
        x_col = greville_points(T, p, nloc)

        for i in 2:(nloc-1)
            x_val = x_col[i]
            row += 1
            
            span = find_span(nloc, p, x_val, T)
            ders = ders_basis_funs(span, x_val, p, 2, T)
            first_j = span - p
            
            # Loop ONLY over the p+1 non-zero basis functions
            for k in 0:p
                j_local = first_j + k
                val = -D * ders[3, k+1] + A * ders[2, k+1] # Row 3 is 2nd deriv, Row 2 is 1st deriv
                push!(I, row); push!(J, base + j_local - 1); push!(V, val)
            end
            B[row] = eval_f(f_edge, e, ei, x_val)
        end
    end

    endpoint_x(e, v) = (v == src(e)) ? float(first(edge_x[e])) : float(last(edge_x[e]))

    # Vertex continuity + Dirichlet/Kirchhoff coupling
    for v in 1:nv
        E = incident[v]
        isempty(E) && continue

        if v in dirset
            for e in E
                row += 1
                x_val = endpoint_x(e, v)
                T = Te[e]; nloc = nloc_e[e]; base = offset[e]
                
                span = find_span(nloc, p, x_val, T)
                ders = ders_basis_funs(span, x_val, p, 0, T)
                first_j = span - p
                
                for k in 0:p
                    push!(I, row); push!(J, base + first_j + k - 1); push!(V, ders[1, k+1])
                end
                B[row] = get_g(g_vertex, v)
            end
        else
            eref = E[1]
            x_ref = endpoint_x(eref, v)
            T_ref = Te[eref]; nloc_ref = nloc_e[eref]; base_ref = offset[eref]

            span_ref = find_span(nloc_ref, p, x_ref, T_ref)
            ders_ref = ders_basis_funs(span_ref, x_ref, p, 0, T_ref)
            first_j_ref = span_ref - p

            # Continuity: u_e(v) - u_ref(v) = 0
            for e in E[2:end]
                row += 1
                x_val = endpoint_x(e, v)
                T = Te[e]; nloc = nloc_e[e]; base = offset[e]

                span = find_span(nloc, p, x_val, T)
                ders = ders_basis_funs(span, x_val, p, 0, T)
                first_j = span - p

                for k in 0:p
                    push!(I, row); push!(J, base + first_j + k - 1); push!(V, ders[1, k+1])
                end
                for k in 0:p
                    push!(I, row); push!(J, base_ref + first_j_ref + k - 1); push!(V, -ders_ref[1, k+1])
                end
                B[row] = 0.0
            end

            # Kirchhoff: Σ_e (-D u'_e(v) + A u_e(v)) n_e(v) = 0
            row += 1
            for (ei, e) in enumerate(edges_ordered)
                if !(e in E); continue; end

                A = get_edge_coeff(A_edge, e, ei)
                D = get_edge_coeff(D_edge, e, ei)
                nrm = n_sign(e, v)
                
                x_val = endpoint_x(e, v)
                T = Te[e]; nloc = nloc_e[e]; base = offset[e]
                
                span = find_span(nloc, p, x_val, T)
                ders = ders_basis_funs(span, x_val, p, 1, T)
                first_j = span - p
                
                for k in 0:p
                    val = nrm * (-D * ders[2, k+1] + A * ders[1, k+1])
                    push!(I, row); push!(J, base + first_j + k - 1); push!(V, val)
                end
            end
            B[row] = 0.0
        end
    end

    @assert row == ndof "System not square: rows=$row, ndof=$ndof."

    Aglob = sparse(I, J, V, ndof, ndof)
    meta = (Te=Te, nloc=nloc_e, offset=offset, p=p, edges_ordered=edges_ordered, edge_x=edge_x, nv=nv)
    
    return Aglob, B, meta
end


"""
    solve_bspline_collocation(edges_ordered, edge_x::Dict; nv::Int, p::Int=3, A_edge=1.0, D_edge=1.0, f_edge=nothing, dirichlet_nodes::Vector{Int}=Int[], g_vertex=nothing, n_sign::Function=normal_sign)

Wrapper function that assembles and directly solves the B-spline collocation system.

# Arguments
- `edges_ordered`: Ordered list of edges in the graph.
- `edge_x::Dict`: Dictionary mapping edges to their physical break points.
- `nv::Int`: Total number of vertices in the graph.
- `p::Int`: B-spline polynomial degree (default: 3).
- `A_edge`: Advection coefficient container (default: 1.0).
- `D_edge`: Diffusion coefficient container (default: 1.0).
- `f_edge`: Right-hand side forcing function container (default: nothing).
- `dirichlet_nodes::Vector{Int}`: Indices of vertices with Dirichlet boundary conditions.
- `g_vertex`: Boundary values at Dirichlet vertices.
- `n_sign::Function`: Outward normal sign evaluation function (default: normal_sign).

# Output:
- `NamedTuple`: Contains the solution coefficients `c`, the system matrix `A`, 
  the RHS vector `b`, and the metadata `meta`.
"""
function solve_bspline_collocation(edges_ordered, edge_x::Dict;
                                   nv::Int,
                                   p::Int=3,
                                   A_edge=1.0,
                                   D_edge=1.0,
                                   f_edge=nothing,
                                   dirichlet_nodes::Vector{Int}=Int[],
                                   g_vertex=nothing,
                                   n_sign::Function=normal_sign)

    A, B, meta = assemble_bspline_collocation(edges_ordered, edge_x;
        nv=nv, p=p, A_edge=A_edge, D_edge=D_edge, f_edge=f_edge,
        dirichlet_nodes=dirichlet_nodes, g_vertex=g_vertex, n_sign=n_sign)
    
    c = A \ B
    return (c=c, A=A, b=B, meta=meta)
end

"""
    eval_edge_u_ders(sol, e, x::Float64, max_der::Int)

Evaluates the numerical solution and its derivatives up to `max_der` on a specific edge.

# Arguments
- `sol`: Solution structure containing coefficients and metadata.
- `e`: The edge object.
- `x::Float64`: Evaluation coordinate on the edge.
- `max_der::Int`: Maximum derivative order to compute.

# Output:
- `Vector{Float64}`: Evaluated derivative values at point `x`.
"""
function eval_edge_u_ders(sol, e, x::Float64, max_der::Int)
    meta = sol.meta
    T = meta.Te[e]  
    p = meta.p
    nloc = meta.nloc[e]
    base = meta.offset[e]
    
    ce = @view sol.c[base:base+nloc-1]

    m_der = min(max_der, p)
    u_ders = zeros(Float64, max_der + 1)
    
    span = find_span(nloc, p, x, T)
    
    ders = ders_basis_funs(span, x, p, m_der, T)
    first_j = span - p
    
    for k in 0:m_der
        s = 0.0
        for j in 0:p
            s += ce[first_j + j] * ders[k+1, j+1]
        end
        u_ders[k+1] = s
    end
    
    return u_ders
end


"""
    eval_edge_u(sol, e, x_vals::AbstractVector{<:Real})

Evaluates the physical solution values (0-th derivative) along an array of points on an edge.

# Arguments
- `sol`: Solution structure containing coefficients and metadata.
- `e`: The edge object.
- `x_vals::AbstractVector{<:Real}`: Collection of evaluation points along the edge.

# Output:
- `Vector{Float64}`: Interpolated solution values.
"""
function eval_edge_u(sol, e, x_vals::AbstractVector{<:Real})
    meta = sol.meta
    T = meta.Te[e]
    p = meta.p
    nloc = meta.nloc[e]
    base = meta.offset[e]
    
    ce = @view sol.c[base:base+nloc-1]

    u = zeros(Float64, length(x_vals))
    for (k, x) in enumerate(x_vals)
        x_fl = Float64(x)
        
        span = find_span(nloc, p, x_fl, T)
        ders = ders_basis_funs(span, x_fl, p, 0, T)
        first_j = span - p
        
        s = 0.0
        for j in 0:p
            s += ce[first_j + j] * ders[1, j+1]
        end
        u[k] = s
    end
    return u
end

"""
    get_exact_solution_ders(case, e, edge_idx::Int, x::Float64, max_der::Int)

Extracts the exact analytical derivatives up to `max_der` from the test case geometry.

# Arguments
- `case`: Test case definition containing exact solutions or derivatives.
- `e`: The edge object.
- `edge_idx::Int`: Index of the edge.
- `x::Float64`: Evaluation coordinate.
- `max_der::Int`: Maximum derivative order required.

# Output:
- `Vector{Float64}`: The exact derivative values.
"""
function get_exact_solution_ders(case, e, edge_idx::Int, x::Float64, max_der::Int)
    exact_ders = zeros(Float64, max_der + 1)
    
    if hasproperty(case, :exact_k_th_derivative) && case.exact_k_th_derivative !== nothing
        for k in 0:max_der
            exact_ders[k+1] = case.exact_k_th_derivative(x, k)[edge_idx]
        end
    elseif hasproperty(case, :exakte_Loesung) && case.exakte_Loesung !== nothing
        exact_ders[1] = case.exakte_Loesung(x)[edge_idx]
    end
    
    return exact_ders
end


"""
    calculate_errors_collocation(sol, case; n_quad::Int = 100)

Calculates the L2, H1 and total Hp error utilizing a composite trapezoidal rule.

# Arguments
- `sol`: Solution structure containing coefficients and metadata.
- `case`: Test case definition containing exact solutions.
- `n_quad::Int`: Number of quadrature points per knot span (default: 100).

# Output:
- `Tuple{Float64, Float64, Float64, Vector{Float64}}`: `(L2_err, H1_err, Hp_total_err, Hp_err_sq)`
"""
function calculate_errors_collocation(sol, case; n_quad::Int = 100)
    p_degree = sol.meta.p
    
    Hp_err_sq = zeros(Float64, p_degree + 1)

    for (i, e) in enumerate(case.edges)
        breaks = case.edge_x[e]
        
        for b_idx in 1:(length(breaks)-1)
            x_start = float(breaks[b_idx])
            x_end = float(breaks[b_idx+1])
            hx = (x_end - x_start) / n_quad
            
            for q in 0:n_quad
                x_val = x_start + q * hx
                
                weight = (q == 0 || q == n_quad) ? hx / 2.0 : hx
                
                num_ders = eval_edge_u_ders(sol, e, x_val, p_degree)
                
                exact_ders = get_exact_solution_ders(case, e, i, x_val, p_degree)
                
                for k in 0:p_degree
                    diff = num_ders[k+1] - exact_ders[k+1]
                    Hp_err_sq[k+1] += (diff^2) * weight
                end
            end
        end
    end
    
    L2_err = sqrt(Hp_err_sq[1])
    H1_err = sqrt(Hp_err_sq[1] + Hp_err_sq[2])
    
    Hp_total_err = sqrt(sum(Hp_err_sq))
    
    return L2_err, H1_err, Hp_total_err, Hp_err_sq
end


"""
    calculate_errors_infty(sol, case; n_eval_per_span::Int = 200)

Calculates the maximum absolute deviation errors over a highly resolved evaluation grid.

# Arguments
- `sol`: Solution structure containing coefficients and metadata.
- `case`: Test case definition containing exact solutions.
- `n_eval_per_span::Int`: Number of evaluation points per sub-interval (default: 200).

# Output:
- `Tuple{Float64, Float64, Float64, Vector{Float64}}`: `(L_inf_err, W1_inf_err, Wp_inf_total_err, Winf_err_semi)`
"""
function calculate_errors_infty(sol, case; n_eval_per_span::Int = 200)
    p_degree = sol.meta.p
    
    Winf_err_semi = zeros(Float64, p_degree + 1)

    for (i, e) in enumerate(case.edges)
        breaks = case.edge_x[e]
        
        for b_idx in 1:(length(breaks)-1)
            x_start = float(breaks[b_idx])
            x_end = float(breaks[b_idx+1])
            
            eval_points = range(x_start, x_end, length=n_eval_per_span)
            
            for x_val in eval_points
                num_ders = eval_edge_u_ders(sol, e, x_val, p_degree)
                
                exact_ders = get_exact_solution_ders(case, e, i, x_val, p_degree)
                
                for k in 0:p_degree
                    diff = abs(num_ders[k+1] - exact_ders[k+1])
                    Winf_err_semi[k+1] = max(Winf_err_semi[k+1], diff)
                end
            end
        end
    end
    
    L_inf_err = Winf_err_semi[1]
    W1_inf_err = max(Winf_err_semi[1], Winf_err_semi[2])
    Wp_inf_total_err = maximum(Winf_err_semi)
    
    return L_inf_err, W1_inf_err, Wp_inf_total_err, Winf_err_semi
end


"""
    plot_convergence_rates_trig(; p_degree=5, J_range=2:6)

Executes a full convergence analysis and plots the resulting Sobolev errors over the step size grid. Uses the test case `testcase_star_trig` for the star graph geometry with trigonometric functions.
Saves the resulting plot as a PDF file in the current working directory.

# Arguments
- `p_degree::Int`: B-spline polynomial degree (default: 5).
- `J_range`: Range of refinement levels (default: 2:6).

# Output:
- `Tuple{Vector{Float64}, Matrix{Float64}, Plots.Plot}`: Generated step sizes, corresponding error matrix, and the plot object.
"""
function plot_convergence_rates_trig(; p_degree=5, J_range=2:6)
    println("Starting convergence analysis for p = $p_degree ...")
    
    N_values = [2^J for J in J_range]
    h_vals   = Float64[]
    err_matrix = zeros(length(J_range), p_degree + 1)

    for (i, J) in enumerate(J_range)
        N = N_values[i]
        println("  -> Calculating for J = $J (N = $N)")
        
        h = 1.0 / N
        push!(h_vals, h)

        my_case = testcase_star_trig(0.1, 1.0, J)

        sol = solve_bspline_collocation(
            my_case.edges, 
            my_case.edge_x;
            nv = my_case.nv,
            p = p_degree,
            A_edge = my_case.a_edge,
            D_edge = my_case.eps_edge,
            f_edge = my_case.f_edge,
            dirichlet_nodes = collect(keys(my_case.dirichlet)),
            g_vertex = my_case.dirichlet,
            n_sign = normal_sign
        )

        L_inf_err, W1_inf_err, Wp_inf_total_err, Winf_err_semi = calculate_errors_infty(
            sol, my_case; n_eval_per_span=200
        )

        err_matrix[i, :] .= Winf_err_semi
    end

    plt = plot(
        xaxis = :log10, yaxis = :log10,
        xlabel = "Step size h",
        ylabel = "Sobolev Error W^{k, ∞}",
        title  = "Coupled B-Spline Collocation (p=$p_degree, Star Graph)",
        legend = :bottomright, grid = true, linewidth = 2, size = (850, 520), leftmargin=5Plots.mm
    )

    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5, :hexagon]
    cols    = palette(:tab10)

    for k in 0:p_degree
        m = mod1(k+1, length(markers))
        c = mod1(k+1, length(cols))

        plot!(plt, h_vals, err_matrix[:, k+1],
            marker = markers[m], color = cols[c],
            label  = "|u^($k)|_∞",  lw = 2, markersize = 5)
-
        if k == 0 || k == 1
            if isodd(p_degree)
                r = p_degree - 1
            else # p_degree is even
                r = p_degree
            end
        else # 2 <= k <= p
            r = p_degree + 1 - k
        end
        
        if r > 0
            offset = 0.5
            ref = (h_vals ./ h_vals[1]).^r .* (err_matrix[1, k+1] * offset)
            plot!(plt, h_vals, ref,
                linestyle = :dash, color = cols[c], alpha = 0.6,
                label = "O(h^$r)", lw = 1.5)
        end
    end

    if !isdir(joinpath("Figures", "plot_convergence_rates_trig"))
        mkdir(joinpath("Figures", "plot_convergence_rates_trig"))
    end
    savefig(plt, joinpath("Figures", "plot_convergence_rates_trig", "collocation_star_trig_convergence_$p_degree.pdf"))
    
    return h_vals, err_matrix, plt
end