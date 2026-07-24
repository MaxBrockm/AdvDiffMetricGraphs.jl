"""
Copyright Anna Weller, Universität zu Köln 2023
Translated and adapted for package integration.

    plot_graph_3d(G::AbstractGraph, l_vec::AbstractVector{<:Real}, coords_v, ue::AbstractVector, uv::AbstractVector, int_nodes::AbstractVector{Int})

Plots functions defined on metric graphs in a 3D visualization.
The graph itself (all vertices and edges) is plotted in the x-y plane. 
The function values along the edges are plotted in the z-coordinate.

# Arguments
- `G::AbstractGraph`: The graph to be plotted.
- `l_vec::AbstractVector{<:Real}`: Vector containing the lengths of all edges.
- `coords_v`: Coordinates of all vertices in the x-y plane.
- `ue::AbstractVector`: Function values on internal edge nodes (nested or segmented).
- `uv::AbstractVector`: Function values on the vertices.
- `int_nodes::AbstractVector{Int}`: Number of internal nodes per edge.

# Output:
- `Plots.Plot`: The 3D plot object. The plot is automatically displayed upon calling.
"""
function plot_graph_3d(G::AbstractGraph, l_vec::AbstractVector{<:Real}, coords_v, ue::AbstractVector, uv::AbstractVector, int_nodes::AbstractVector{Int})
    plt = plot3d(grid=true, tickfontsize=8, dpi=1200, legend=false, legendfontsize=14) # legend=(0.3,0.3)
    
    n_edges = ne(G)

    for (m, e) in enumerate(edges(G))
        u_node = src(e)
        v_node = dst(e)
        xe = LinRange(coords_v[u_node][1], coords_v[v_node][1], 100)
        ye = LinRange(coords_v[u_node][2], coords_v[v_node][2], 100)
        ze = zeros(length(xe))
        
        labelstring = (m == 1) ? L"\mathcal{G}" : ""
        plot3d!(xe, ye, ze, label=labelstring, color="gray", lw=4)
    end
    
    for (m, e) in enumerate(edges(G))
        u_node = src(e)
        v_node = dst(e)
        
        n_points = int_nodes[m] + 2
        xe = LinRange(coords_v[u_node][1], coords_v[v_node][1], n_points)
        ye = LinRange(coords_v[u_node][2], coords_v[v_node][2], n_points)
        
        ze = [uv[u_node], ue[m]..., uv[v_node]] 
        
        if n_edges < 8
            stiel_step = max(1, div(n_points, 15)) 
            
            for i in 1:stiel_step:n_points
                plot3d!([xe[i], xe[i]], [ye[i], ye[i]], [0, ze[i]], 
                        color="gray", lw=0.8, alpha=0.3, label="", markershape=:none)
            end
        end
        
        labelstring = (m == 1) ? L"u" : ""
        plot3d!(xe, ye, ze, label=labelstring, color="orange", lw=6)
        
    end
    
    return plt
end


"""
    plot_case_3d(case, ue::AbstractVector, uv::AbstractVector)

Unpacks the NamedTuple case environment and prepares all data to automatically generate 
the 3D plot mapping numeric solutions back to the metric graph topology.

# Arguments
- `case`: NamedTuple containing the graph definitions and properties.
- `ue::AbstractVector`: Solution vector for internal edge nodes.
- `uv::AbstractVector`: Solution vector for graph vertices.

# Output:
- `Plots.Plot`: The generated 3D plot object.
"""
function plot_case_3d(case, ue, uv)
    
    M = length(case.edges)
    NV = case.nv
    
    G = SimpleDiGraph(NV)
    for e in case.edges
        add_edge!(G, e)
    end
    
    l_vec = zeros(Float64, M)
    int_nodes = zeros(Int, M)
    
    for (i, e) in enumerate(case.edges)
        l_vec[i] = case.edge_x[e][end] - case.edge_x[e][1]
        int_nodes[i] = case.n_e[e] - 1
    end
    
    ue_split = Array{Any}(undef, M)
    lgt = 1
    for j = 1:M
        ue_split[j] = ue[lgt : lgt + int_nodes[j] - 1]
        lgt += int_nodes[j]
    end
    
    coords_v = Vector{Vector{Float64}}(undef, NV)
    
    if occursin("stargraph", case.name)
        coords_v[1] = [0.0, 0.0] 
        N_orig = min(4, NV) 
        
        for i in 2:N_orig
            angle = 2 * pi * (i - 2) / (N_orig - 1)
            edge_idx = findfirst(e -> (src(e)==1 && dst(e)==i) || (src(e)==i && dst(e)==1), case.edges)
            r = l_vec[edge_idx]
            coords_v[i] = [r * cos(angle), r * sin(angle)]
        end
        
        for i in (N_orig + 1):NV
            p_node = neighbors(G, i)[1]
            edge_idx = findfirst(e -> (src(e)==p_node && dst(e)==i) || (src(e)==i && dst(e)==p_node), case.edges)
            L_new = l_vec[edge_idx]
            
            if p_node == 1
                coords_v[i] = [L_new, 0.0]
            else
                dir = coords_v[p_node] ./ norm(coords_v[p_node])
                coords_v[i] = coords_v[p_node] .+ dir .* L_new
            end
        end
    else
        R_max = maximum(l_vec) * 1.5 
        for i in 1:NV
            angle = 2 * pi * (i - 1) / NV
            coords_v[i] = [R_max * cos(angle), R_max * sin(angle)]
        end
    end
    
    plt = plot_graph_3d(G, l_vec, coords_v, ue_split, uv, int_nodes)
    
    return plt
end


"""
    plot_shishkin_3d(case, ue::AbstractVector, uv::AbstractVector; title="3D Solution on Metric Graph")

Generates a 3D visualization specifically adapted for layer-adapted meshes (like Shishkin grids), 
correctly mapping unevenly spaced grid points to their physical lengths.

# Arguments
- `case`: NamedTuple containing the graph definitions.
- `ue::AbstractVector`: Solution vector for internal edge nodes.
- `uv::AbstractVector`: Solution vector for graph vertices.
- `title::String`: Plot title.

# Output:
- `Plots.Plot`: The generated 3D plot object.
"""
function plot_shishkin_3d(case, ue, uv; title="3D Solution on Metric Graph")
    NV = case.nv
    M = length(case.edges)

    G = SimpleGraph(NV)
    for e in case.edges
        add_edge!(G, e)
    end

    coords_v = Vector{Vector{Float64}}(undef, NV)
    
    if occursin("star", lowercase(case.name)) || (NV == 4 && M == 3)
        degs = degree(G)
        center_node = argmax(degs)
        
        coords_v[center_node] = [0.0, 0.0] 
        
        leaves = neighbors(G, center_node)
        for (idx, leaf) in enumerate(leaves)
            angle = 2 * pi * (idx - 1) / length(leaves)
            
            edge_idx = findfirst(e -> (src(e)==center_node && dst(e)==leaf) || (src(e)==leaf && dst(e)==center_node), case.edges)
            e = case.edges[edge_idx]
            
            L_edge = case.edge_x[e][end] - case.edge_x[e][1]
            coords_v[leaf] = [L_edge * cos(angle), L_edge * sin(angle)]
        end
        
        for i in 1:NV
            if !isassigned(coords_v, i)
                coords_v[i] = [1.0, 1.0] 
            end
        end
    else
        max_L = maximum([case.edge_x[e][end] - case.edge_x[e][1] for e in case.edges])
        R_max = max_L * 1.5 
        for i in 1:NV
            angle = 2 * pi * (i - 1) / NV
            coords_v[i] = [R_max * cos(angle), R_max * sin(angle)]
        end
    end

    plt = plot3d(grid=true, legend=false, title=title, dpi=300)

    for e in case.edges
        u, v = src(e), dst(e)
        plot3d!(plt, [coords_v[u][1], coords_v[v][1]], 
                     [coords_v[u][2], coords_v[v][2]], 
                     [0.0, 0.0], color=:gray, lw=3, alpha=0.5)
    end

    lgt = 1
    for e in case.edges
        u, v = src(e), dst(e)
        
        x_nodes = case.edge_x[e]
        L = x_nodes[end] - x_nodes[1]
        t = (x_nodes .- x_nodes[1]) ./ L  
        
        X = coords_v[u][1] .+ t .* (coords_v[v][1] - coords_v[u][1])
        Y = coords_v[u][2] .+ t .* (coords_v[v][2] - coords_v[u][2])
        
        int_nodes_count = case.n_e[e] - 1
        ue_kante = ue[lgt : lgt + int_nodes_count - 1]
        lgt += int_nodes_count
        
        Z = [uv[u]; ue_kante; uv[v]]
        
        plot3d!(plt, X, Y, Z, color=:orange, lw=4)
        
        step = max(1, length(Z) ÷ 15)
        for i in 1:step:length(Z)
            plot3d!(plt, [X[i], X[i]], [Y[i], Y[i]], [0.0, Z[i]], 
                    color=:gray, lw=1, alpha=0.4)
        end
    end
    
    return plt
end


"""
    plot_case_edge_difference(case, ue::AbstractVector, uv::AbstractVector, plot_title="Difference u_h - u_exact per edge"; nsamp_per_elem::Int=20, layout=(2,2))

Plots the local error distribution (u_h - u_exact) mapped independently onto each metric edge.

# Arguments
- `case`: NamedTuple containing the exact solution and grid definitions.
- `ue::AbstractVector`: Solution vector for internal edge nodes.
- `uv::AbstractVector`: Solution vector for graph vertices.
- `plot_title::String`: Top-level title for the layout.
- `nsamp_per_elem::Int`: Sampling resolution per linear element.
- `layout`: Subplot grid layout.

# Output:
- `Plots.Plot`: Grid of subplot error curves.
"""
function plot_case_edge_difference(case, ue::AbstractVector, uv::AbstractVector, 
                                 plot_title="Difference u_h - u_exact per edge";
                                 nsamp_per_elem::Int=20, layout=(2,2))
    
    p = plot(layout=layout, legend=false, plot_title=plot_title, size=layout.*(400,300))
    
    if isnothing(case.exakte_Loesung)
        @warn "Keine exakte Lösung im Case vorhanden."
        return p
    end

    current_ue_idx = 0

    for (idx, e) in enumerate(case.edges)
        x_nodes = case.edge_x[e]      
        n_elems = length(x_nodes) - 1 
        
        u_start = uv[src(e)]
        u_end   = uv[dst(e)]
        
        n_inner = n_elems - 1
        u_inner = ue[current_ue_idx+1 : current_ue_idx+n_inner]
        current_ue_idx += n_inner
        
        u_edge_nodes = [u_start; u_inner; u_end]

        S_plot, ERR_plot = Float64[], Float64[]
        
        for i in 1:n_elems
            x1, x2 = x_nodes[i], x_nodes[i+1]
            val1, val2 = u_edge_nodes[i], u_edge_nodes[i+1]
            h = x2 - x1

            for t in range(0, 1; length=nsamp_per_elem)
                s = (1-t)*x1 + t*x2
                uh = ((x2-s)/h)*val1 + ((s-x1)/h)*val2
                
                uex = case.exakte_Loesung(s)[idx]
                
                push!(S_plot, s)
                push!(ERR_plot, uh - uex)
            end
        end

        plot!(p, S_plot, ERR_plot, lw=2, 
              xlabel="s (edge $(idx): $(src(e))->$(dst(e)))", 
              ylabel="Error", subplot=idx)
        hline!(p, [0.0], linestyle=:dash, color=:black, alpha=0.5, subplot=idx)
    end

    display(p)
    return p
end


"""
    plot_case_num_vs_exact(case, ue::AbstractVector, uv::AbstractVector, plot_title="Numerical vs. Exact Solution"; nsamp::Int=400, layout=(2,2), markersize=4)

Compares numerical and exact solutions side-by-side on individual 2D edge plots.

# Output:
- `Plots.Plot`: Subplots showcasing discrete numeric samples alongside the exact continuous curve.
"""
function plot_case_num_vs_exact(case, ue::AbstractVector, uv::AbstractVector, 
                                plot_title="Numerical vs. Exact Solution";
                                nsamp::Int=400, layout=(2,2), markersize=4)
    
    p = plot(layout=layout, legend=false, plot_title=plot_title, size=layout.*(400,300))
    
    if isnothing(case.exakte_Loesung)
        @warn "No exact solution available in case. Cannot plot comparison."
        return p
    end

    current_ue_idx = 0

    for (idx, e) in enumerate(case.edges)
        x_nodes = case.edge_x[e]
        n_elems = length(x_nodes) - 1
        
        u_start = uv[src(e)]
        u_end   = uv[dst(e)]
        n_inner = n_elems - 1
        u_inner = ue[current_ue_idx+1 : current_ue_idx+n_inner]
        current_ue_idx += n_inner
        
        u_diskrete_punkte = [u_start; u_inner; u_end]

        x_fine = range(first(x_nodes), last(x_nodes); length=nsamp)
        y_exact = [case.exakte_Loesung(s)[idx] for s in x_fine]

        plot!(p, x_fine, y_exact, lw=2.0, color=:black, alpha=0.7,
              xlabel="s (edge $(idx); ($(src(e))->$(dst(e))))", ylabel="u(s)", subplot=idx)
        
        scatter!(p, x_nodes, u_diskrete_punkte, 
                 ms=markersize, mc=:orange, markerstrokewidth=0, 
                 subplot=idx)
    end

    display(p)
    return p
end


"""
    plot_edges_2d_simple(case, ue::AbstractVector, uv::AbstractVector)

Plots individual edge solutions sequentially in a vertically stacked layout.
"""
function plot_edges_2d_simple(case, ue, uv)
    M = length(case.edges)
    
    plt = plot(layout=(M, 1), size=(800, 300*M), legend=:topright)
    
    lgt = 1
    for (i, e) in enumerate(case.edges)
        u_node = src(e)
        v_node = dst(e)
        
        X = case.edge_x[e]
        
        int_nodes_count = case.n_e[e] - 1
        ue_kante = ue[lgt : lgt + int_nodes_count - 1]
        lgt += int_nodes_count
        
        Y_num = [uv[u_node]; ue_kante; uv[v_node]]
        
        Pe_e = abs(case.a_edge[e]) * (case.edge_x[e][2] -case.edge_x[e][1]) / (2*case.eps_edge[e])

        plot!(plt, X, Y_num, subplot=i, marker=:circle, markersize=3, lw=2,
              label="Numeric solution", title="edge $i:  ($(u_node),$(v_node)); Pe = $(round(Pe_e, sigdigits=3))",
              xlabel="x", ylabel="u")
              
        if case.exakte_Loesung !== nothing
            Y_exakt = [case.exakte_Loesung(x)[i] for x in X]
            plot!(plt, X, Y_exakt, subplot=i, lw=2, linestyle=:dash, color=:black, label="exact solution")
        end
    end
    
    display(plt)
    return plt
end


"""
    calculate_case_l2_error(ue::AbstractVector, uv::AbstractVector, case)

Calculates the L2 error based on the NamedTuple `case` structure using a 3-Point 
Gauss-Legendre Quadrature rule to ensure high accuracy over the piecewise linear elements.

# Arguments
- `ue::AbstractVector`: Vector of internal node values (computed numerical solution).
- `uv::AbstractVector`: Vector of graph vertex values.
- `case`: NamedTuple holding the exact solution and grid coordinates.

# Output:
- `Tuple{Float64, Float64}`: Absolute and relative L2 errors `(L2_err, rel_err)`.
"""
function calculate_case_l2_error(ue, uv, case)
    if isnothing(case.exakte_Loesung)
        @warn "No exact solution defined in case!"
        return NaN, NaN
    end

    err2_global = 0.0
    nrm2_global = 0.0
    curr_offset = 1

    gauss_nodes   = [-sqrt(3/5), 0.0, sqrt(3/5)]
    gauss_weights = [5/9, 8/9, 5/9]

    for (i, e) in enumerate(case.edges)
        x_points = case.edge_x[e]
        n_intervals = case.n_e[e]
        n_int = n_intervals - 1 

        u_start = uv[src(e)]
        u_end   = uv[dst(e)]
        
        u_internal = @view ue[curr_offset : curr_offset + n_int - 1]

        function get_u(j)
            if j == 1 return u_start end
            if j == n_intervals + 1 return u_end end
            return u_internal[j - 1]
        end

        for j in 1:n_intervals
            xa = x_points[j]
            xb = x_points[j+1]
            
            h_local = xb - xa 
            
            va = get_u(j)
            vb = get_u(j+1)
            
            half_h = h_local / 2.0
            mid_x  = (xa + xb) / 2.0

            e2_local = 0.0
            n2_local = 0.0

            for q in 1:3
                xi = gauss_nodes[q]
                wq = gauss_weights[q]
                
                s = half_h * xi + mid_x
                
                uh_val = va * (xb - s)/h_local + vb * (s - xa)/h_local
                
                u_ex_val = case.exakte_Loesung(s)[i]
                
                e2_local += wq * (uh_val - u_ex_val)^2
                n2_local += wq * (u_ex_val)^2
            end
            
            err2_global += half_h * e2_local
            nrm2_global += half_h * n2_local
        end
        
        curr_offset += n_int
    end

    L2_err = sqrt(err2_global)
    rel_err = L2_err / max(sqrt(nrm2_global), 1e-14)
    
    return L2_err, rel_err
end


"""
    calculate_case_D_error(ue::AbstractVector, uv::AbstractVector, case; use_SUPG::Bool=false)

Calculates the energy norm (D-norm) error and optionally the SUPG-norm error 
using a 5-Point Gauss-Legendre Quadrature rule.

# Arguments
- `ue::AbstractVector`: Internal node values.
- `uv::AbstractVector`: Graph vertex values.
- `case`: NamedTuple holding the exact continuous derivatives.
- `use_SUPG::Bool`: If true, includes the stabilization tau_K parameter in the error norm calculation.

# Output:
- `Tuple{Float64, Float64}`: Absolute and relative Energy/SUPG-norm errors.
"""
function calculate_case_D_error(ue, uv, case; use_SUPG::Bool=false)
    if isnothing(case.exakte_Loesung)
        @warn "No exact solution defined in case!"
        return NaN, NaN
    end
    
    if !hasfield(typeof(case), :exact_derivative) || isnothing(case.exact_derivative)
        @warn "You need an 'exact_derivative' function to compute the SUPG-norm!"
        return NaN, NaN
    end

    err2_global = 0.0
    nrm2_global = 0.0
    curr_offset = 1

    # 5-Point Gauss-Legendre Quadrature rules on [-1, 1]
    gauss_nodes   = [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664]
    gauss_weights = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]

    for (i, e) in enumerate(case.edges)
        x_points = case.edge_x[e]
        n_intervals = case.n_e[e]
        n_int = n_intervals - 1 

        D_e = case.eps_edge[e] 
        A_e = case.a_edge[e]
        A_mag = max(abs(A_e), 1e-14)

        u_start = uv[src(e)]
        u_end   = uv[dst(e)]
        u_internal = @view ue[curr_offset : curr_offset + n_int - 1]

        function get_u(j)
            if j == 1 return u_start end
            if j == n_intervals + 1 return u_end end
            return u_internal[j - 1]
        end

        for j in 1:n_intervals
            xa = x_points[j]
            xb = x_points[j+1]
            h_local = xb - xa 
            
            # --- SUPG TOGGLE LOGIC ---
            tau_K = use_SUPG ? tau_supg(A_mag, D_e, h_local) : 0.0

            va = get_u(j)
            vb = get_u(j+1)
            
            duh_val = (vb - va) / h_local
            
            half_h = h_local / 2.0
            mid_x  = (xa + xb) / 2.0

            e2_L2_local = 0.0
            e2_H1_local = 0.0
            e2_SUPG_local = 0.0
            
            n2_L2_local = 0.0
            n2_H1_local = 0.0
            n2_SUPG_local = 0.0

            for q in 1:5
                xi = gauss_nodes[q]
                wq = gauss_weights[q]
                
                s = half_h * xi + mid_x
                
                uh_val   = va * (xb - s)/h_local + vb * (s - xa)/h_local
                u_ex_val = case.exakte_Loesung(s)[i]
                du_ex_val = case.exact_derivative(s)[i] 
                
                e2_L2_local += wq * (uh_val - u_ex_val)^2
                e2_H1_local += wq * (duh_val - du_ex_val)^2
                e2_SUPG_local += wq * (A_e * (duh_val - du_ex_val))^2
                
                n2_L2_local += wq * (u_ex_val)^2
                n2_H1_local += wq * (du_ex_val)^2
                n2_SUPG_local += wq * (A_e * du_ex_val)^2
            end
            
            # If use_SUPG is false, tau_K is 0.0, and the third term cleanly vanishes!
            err2_global += half_h * (e2_L2_local + D_e * e2_H1_local + tau_K * e2_SUPG_local)
            # err2_global += half_h * (e2_L2_local + tau_K * e2_SUPG_local)
            nrm2_global += half_h * (n2_L2_local + D_e * n2_H1_local + tau_K * n2_SUPG_local)
        end
        
        curr_offset += n_int
    end

    SUPG_err = sqrt(err2_global)
    rel_SUPG_err = SUPG_err / max(sqrt(nrm2_global), 1e-14)
    
    return SUPG_err, rel_SUPG_err
end