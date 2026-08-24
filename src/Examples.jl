"""
    evalBSpline(x::Number, koeffs::Vector, ℓ::Number, deriv::Int)

Evaluates linear B-spline hat functions (or their derivatives) at a given evaluation point `x`.

# Arguments
- `x::Number`: The spatial coordinate where the spline is evaluated.
- `koeffs::Vector`: Vector of B-spline coefficients.
- `ℓ::Number`: Length of the given edge or interval.
- `deriv::Int`: Order of the derivative to evaluate (e.g., 0 for function value, 1 for first derivative).

# Output
- `Float64`: The evaluated scalar value of the B-spline or its derivative.
"""
function evalBSpline(x::Number, koeffs::Vector, ℓ::Number, deriv::Int)
    knots = LinRange(0, ℓ, length(koeffs))
    he = knots[2] - knots[1]
    BSB = BSplineBasis(2, knots)
    i = ceil(Int, x / he)
    bseval = bsplines(BSB, x, Derivative(deriv))
    return dot(koeffs[firstindex(bseval):lastindex(bseval)], bseval)
end


"""
    MG_example_stargraph(v1::Int, v2::Int; do_plots=false)

Executes a multigrid solver example on a star graph topology for the diffusion-reaction equation.
Calculates and plots convergence rates as well as absolute pointwise errors in 3D.

# Arguments
- `v1::Int`: Number of pre-smoothing steps (e.g., 5).
- `v2::Int`: Number of post-smoothing steps (e.g., 3).
- `do_plots::Bool`: Flag to enable or disable plotting of results (default: false).

# Output
- `ue::Vector{Float64}`: Final numerical solution on the internal edge nodes.
- `uv::Vector{Float64}`: Final numerical solution on the graph vertices.
"""
function MG_example_stargraph(v1::Int, v2::Int; do_plots=false)

    Γ = star_graph(4)
    Levels = 10 * ones(Int, ne(Γ))
    J = 10
    Pot = 0.1
    edgelength = 3 / 2 * pi * ones(ne(Γ))

    nem1 = 2^J - 1

    HEE, HEV, HVV = createH(Γ, Levels, edgelength, Pot)

    f_exakt(x, k, Pot) = (
        if k == 1
            return -2 * sin(x) - 2 * Pot * sin(x)
        elseif k in [2, 3]
            return sin(x) + Pot * sin(x)
        end
    )

    u(x, k) = (
        if k == 1
            return -2 * sin(x)
        elseif k in [2, 3]
            return sin(x)
        end
    )

    up(x, k) = (
        if k == 1
            return -2 * cos(x)
        elseif k in [2, 3]
            return cos(x)
        end
    )

    fe, fv = righthandside(f_exakt, Γ, Levels, edgelength, Pot)

    Scale_HEV = Rest_HEV_setup(HEV)

    ue = ones(size(fe))
    uv = ones(size(fv))

    rate_e_vec = []
    rate_v_vec = []
    rate_vec = []
    L2_error = []
    H1_error = []
    kmax = 0
    
    while ((kmax < 250) && (sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2)) > 10^(-8))
        
        ue, uv = Multigrid_Graph(HEE, HEV, HVV, Levels, J, fv, fe, uv, ue,
            ne(Γ), Scale_HEV, v1, v2, 0.5, 1)
        push!(rate_vec, sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2))
        push!(rate_e_vec, norm(fe - HEE * ue - HEV * uv))
        push!(rate_v_vec, norm(fv - transpose(HEV) * ue - HVV * uv))

        # L_2 Error
        l2 = quadgk(x -> (u(x, 1) - evalBSpline(x, [uv[1]; ue[1:nem1]; uv[2]], 3 / 2 * pi, 0))^2, 0, 3 / 2 * pi)[1]
        l2 += quadgk(x -> (u(x, 2) - evalBSpline(x, [uv[1]; ue[nem1+1:2*nem1]; uv[3]], 3 / 2 * pi, 0))^2, 0, 3 / 2 * pi)[1]
        l2 += quadgk(x -> (u(x, 3) - evalBSpline(x, [uv[1]; ue[2*nem1+1:end]; uv[4]], 3 / 2 * pi, 0))^2, 0, 3 / 2 * pi)[1]
        push!(L2_error, sqrt(l2))

        ## H1-norm
        h1 = quadgk(x -> (up(x, 1) - evalBSpline(x, [uv[1]; ue[1:nem1]; uv[2]], 3 / 2 * pi, 1))^2, 0, 3 / 2 * pi)[1]
        h1 += quadgk(x -> (up(x, 2) - evalBSpline(x, [uv[1]; ue[nem1+1:2*nem1]; uv[3]], 3 / 2 * pi, 1))^2, 0, 3 / 2 * pi)[1]
        h1 += quadgk(x -> (up(x, 3) - evalBSpline(x, [uv[1]; ue[2*nem1+1:end]; uv[4]], 3 / 2 * pi, 1))^2, 0, 3 / 2 * pi)[1]
        push!(H1_error, sqrt(l2 + h1))

        kmax += 1
    end

    if do_plots
        Plot_Res = plot(2:1:kmax, [rate_vec[2:end] ./ rate_vec[1:end-1], rate_e_vec[2:end] ./ rate_e_vec[1:end-1], rate_v_vec[2:end] ./ rate_v_vec[1:end-1]],
            label=["convergence rate" "rate on internal vertices" "rate on vertices"], xlabel="cycle", ylabel="convergence rate",
            title="Number of MG-cycles: $kmax", legend=:topleft, lw=3, tickfontsize=8, linestyle=[:solid :dash :dash], dpi=1200,legendfontsize=10, ylims=(0.05, 0.1))
        if !isdir(joinpath("Figures", "MG_example_stargraph"))
            mkpath(joinpath("Figures", "MG_example_stargraph"))
        end
        savefig(Plot_Res, joinpath("Figures", "MG_example_stargraph", "convergencestargraph.pdf"))

        Plot_err = plot(1:1:kmax, [L2_error, H1_error], label=[L"$L_2$-error" L"$H^1$-error"], ylabel="error",
            xlabel="cycle", title=L"$L_2/H^1$-norm of error", lw=5, tickfontsize=8, yaxis=:log, dpi=1200,legendfontsize=16)
        savefig(Plot_err, joinpath("Figures", "MG_example_stargraph", "L2_H1_error.pdf"))

        ue_split = Vector{Any}(undef, ne(Γ))
        lgt = 1
        for j = 1:ne(Γ)
            ue_split[j] = ue[lgt:lgt+2^Levels[j]-2]
            lgt += 2^Levels[j] - 1
        end
        coords_v = ([0, 0], [0, edgelength[3]], [edgelength[1], 0], [-edgelength[2], 0])
        plt_graph = plot_graph_3d(Γ, edgelength, coords_v, ue_split, uv, 2 .^ Levels .- 1)
        savefig(plt_graph, joinpath("Figures", "MG_example_stargraph", "solutionstargraph.pdf"))

        uv_exakt = zeros(nv(Γ))
        uv_exakt[1] = u(0, 1) 
        
        ue_exakt = zeros(length(ue))
        lgt_err = 1
        for (i, e) in enumerate(edges(Γ))
            uv_exakt[dst(e)] = u(edgelength[i], i)
            
            N_e = 2^Levels[i] - 1
            h_e = edgelength[i] / (N_e + 1)
            for j = 1:N_e
                ue_exakt[lgt_err] = u(j * h_e, i)
                lgt_err += 1
            end
        end

        ue_err = abs.(ue .- ue_exakt)
        uv_err = abs.(uv .- uv_exakt)

        ue_err_split = Vector{Any}(undef, ne(Γ))
        lgt_err = 1
        for j = 1:ne(Γ)
            ue_err_split[j] = ue_err[lgt_err:lgt_err+2^Levels[j]-2]
            lgt_err += 2^Levels[j] - 1
        end

        plt_err_graph = plot_graph_3d(Γ, edgelength, coords_v, ue_err_split, uv_err, 2 .^ Levels .- 1)
        savefig(plt_err_graph, joinpath("Figures", "MG_example_stargraph", "error_3D_stargraph.pdf"))

        plt_exact_graph = plot_graph_3d(Γ, edgelength, coords_v, ue_split, uv_exakt, 2 .^ Levels .- 1)
        title!(plt_exact_graph, "Exact solution on the graph")
        savefig(plt_exact_graph, joinpath("Figures", "MG_example_stargraph", "exact_solution_stargraph.pdf"))
    end
    return ue, uv
end


"""
    MG_example_Barabasi(v1::Int, v2::Int; nodes::Int=5000, do_plots=false)

Executes a multigrid solver example on a large Barabási-Albert graph with uniform edge discretization rules to solve the diffusion-reaction equation. For reproducability the random seed is set to 1.

# Arguments
- `v1::Int`: Number of pre-smoothing steps.
- `v2::Int`: Number of post-smoothing steps.
- `nodes::Int`: Number of nodes in the graph (default: 5000).
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
- `ue::Vector{Float64}`: Solution vector for the internal edge nodes.
- `uv::Vector{Float64}`: Solution vector for the graph vertices.
"""
function MG_example_Barabasi(v1, v2; nodes::Int=5000, do_plots=false)

    Γ = barabasi_albert(nodes, 13, complete=true, seed=1)

    Levels = repeat([7, 8], floor(Int, ne(Γ) / 2))
    J = 8

    edge_length = repeat([1.0, 2.0], floor(Int, ne(Γ) / 2))
    if length(Levels) != ne(Γ)
        push!(Levels, 7)
        push!(edge_length, 1)
    end

    estimated_dofs = estimate_dofs(Γ, Levels)
    sparse_index_type = choose_sparse_index_type(estimated_dofs)
    @info "Estimated DoFs: $estimated_dofs | sparse index type: $sparse_index_type"

    HEE, HEV, HVV = createH(Γ, Levels, edge_length, 0.1; index_type=sparse_index_type)
    f_exakt(x, k, Pot) = cos(2 * pi * x/edge_length[k] * (k % 4))
    fe, fv = righthandside(f_exakt, Γ, Levels, edge_length, 0.1; index_type=sparse_index_type)

    Scale_HEV = Rest_HEV_setup(HEV; index_type=sparse_index_type)

    ue, uv = (ones(size(fe)), ones(size(fv)))

    rate_e_vec = []
    rate_v_vec = []
    rate_vec = []
    kmax = 0

    while ((kmax < 250) && (sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2)) > 10^(-8))
        ue, uv = Multigrid_Graph(HEE, HEV, HVV, Levels, J, fv, fe, uv, ue, ne(Γ), Scale_HEV, v1, v2, 0.5, 1)

        push!(rate_vec, sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2))
        push!(rate_e_vec, norm(fe - HEE * ue - HEV * uv))
        push!(rate_v_vec, norm(fv - transpose(HEV) * ue - HVV * uv))

        kmax += 1
    end

    if do_plots
        Plot_Res = plot(2:1:kmax, [rate_vec[2:end] ./ rate_vec[1:end-1], rate_e_vec[2:end] ./ rate_e_vec[1:end-1],
                rate_v_vec[2:end] ./ rate_v_vec[1:end-1]],
            label=["convergence rate" "rate on internal vertices" "rate on vertices"], xlabel="cycle", ylabel="convergence rate",
            title="Number of MG-cycles: $kmax", legend=:topleft, lw=5, tickfontsize=8, ylims = (0.05, 0.075), linestyle=[:solid :dash :dash], dpi=1200,legendfontsize=16)
        if !isdir(joinpath("Figures", "MG_example_Barabasi"))
            mkpath(joinpath("Figures", "MG_example_Barabasi"))
        end
        savefig(Plot_Res, joinpath("Figures", "MG_example_Barabasi", "convergencebarabasi.pdf"))
    end
    return ue, uv
end


"""
    run_MG_barabasi(N_nodes::Int, J_max::Int, v1::Int, v2::Int)

Helper function encapsulating the setup and multigrid solving process for a configurable Barabási-Albert graph.

# Arguments
- `N_nodes::Int`: Number of nodes to generate in the graph.
- `J_max::Int`: Base discretization level for edge elements.
- `v1::Int`: Number of pre-smoothing iterations.
- `v2::Int`: Number of post-smoothing iterations.

# Output
Returns a tuple containing:
- `kmax::Int`: Total number of multigrid cycles executed.
- `DoF::Int`: Degrees of freedom.
- `comp_time::Float64`: Elapsed computation time during the solve phase.
- `rate_vec::Vector{Float64}`: Recorded total residual errors per cycle.
- `rate_e_vec::Vector{Float64}`: Recorded internal edge residual errors.
- `rate_v_vec::Vector{Float64}`: Recorded vertex residual errors.
- `M::Int`: Number of edges.
- `NV::Int`: Number of vertices.
"""
function run_MG_barabasi(N_nodes::Int, J_max::Int, v1::Int, v2::Int)
    Γ = barabasi_albert(N_nodes, 13, complete=true, seed=1)
    M = ne(Γ)
    # Alternate levels between J_max - 1 and J_max
    Levels = J_max * ones(Int, M)
    edge_length = 2.0 * ones(M)

    estimated_dofs = estimate_dofs(Γ, Levels)
    sparse_index_type = choose_sparse_index_type(estimated_dofs)
    @info "Estimated DoFs: $estimated_dofs | sparse index type: $sparse_index_type"

    HEE, HEV, HVV = createH(Γ, Levels, edge_length, 0.1; index_type=sparse_index_type)
    
    f_exakt(x, k, Pot) = cos(2 * pi * x/edge_length[k] * (k % 4))
    fe, fv = righthandside(f_exakt, Γ, Levels, edge_length, 0.1; index_type=sparse_index_type)

    DoF = length(fe) + length(fv)

    Scale_HEV = Rest_HEV_setup(HEV; index_type=sparse_index_type)

    mg_hierarchy = setup_MG_hierarchy(HEE, HEV, HVV, Levels, J_max, Scale_HEV; index_type=sparse_index_type)
    ue, uv = (ones(size(fe)), ones(size(fv)))

    rate_e_vec = Float64[]
    rate_v_vec = Float64[]
    rate_vec = Float64[]
    kmax = 0

    comp_time = @elapsed begin
        while ((kmax < 250) && (sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2)) > 10^(-8))
            ue, uv = Multigrid_Graph_solve!(mg_hierarchy, J_max, fv, fe, uv, ue, v1, v2, 0.5, 1)

            push!(rate_vec, sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2))
            push!(rate_e_vec, norm(fe - HEE * ue - HEV * uv))
            push!(rate_v_vec, norm(fv - transpose(HEV) * ue - HVV * uv))
            
            kmax += 1
        end
    end

    return kmax, DoF, comp_time, rate_vec, rate_e_vec, rate_v_vec, M, nv(Γ)
end


"""
    MG_example_Barabasi_extended(v1::Int, v2::Int; do_plots=false)

Evaluates and outputs performance metrics of the multigrid algorithm over multiple configurations of the Barabási-Albert graph (varying node count and refinement levels).

# Arguments
- `v1::Int`: Number of pre-smoothing iterations.
- `v2::Int`: Number of post-smoothing iterations.
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
- `table_data::Matrix`: Formatted experimental data displaying performance, complexity, and computation time.
"""
function MG_example_Barabasi_extended(v1::Int, v2::Int; do_plots=false)
    Random.seed!(1)
    kmax_base, _, _, rate_vec, rate_e_vec, rate_v_vec, _, _ = run_MG_barabasi(5000, 8, v1, v2)

    if do_plots
        Plot_Res = plot(
                2:1:kmax_base, [rate_vec[2:end] ./ rate_vec[1:end-1], 
                rate_e_vec[2:end] ./ rate_e_vec[1:end-1],
                rate_v_vec[2:end] ./ rate_v_vec[1:end-1]],
                label=["convergence rate" "rate on internal vertices" "rate on vertices"], 
                xlabel="cycle", ylabel="convergence rate",
                title="Number of MG-cycles: $kmax_base", 
                legend=:topleft, 
                lw=3, tickfontsize=8, 
                linestyle=[:solid :dash :dash], 
                dpi=1200, 
                legendfontsize=10
            )
        if !isdir(joinpath("Figures", "MG_example_Barabasi_extended"))
            mkpath(joinpath("Figures", "MG_example_Barabasi_extended"))
        end
        savefig(Plot_Res, joinpath("Figures", "MG_example_Barabasi_extended", "convergencebarabasi_extended.pdf"))
    end

    configs = [
        (2000, 8),
        (2000, 10),
        (4000, 8),
        (4000, 10),
        (6000, 8)
    ]
    
    nodes_arr  = Int[]
    edges_arr  = Int[]
    j_arr      = Int[]
    dof_arr    = Int[]
    cycles_arr = Int[]
    time_arr   = Float64[]
    
    for (N_nodes, J_val) in configs
        @info "Solving for Nodes = $N_nodes, J = $J_val..."
        kmax, DoF, comp_time, _, _, _, M_edges, V_nodes = run_MG_barabasi(N_nodes, J_val, v1, v2)
        
        push!(nodes_arr, V_nodes)
        push!(edges_arr, M_edges)
        push!(j_arr, J_val)
        push!(dof_arr, DoF)
        push!(cycles_arr, kmax)
        push!(time_arr, comp_time)
    end

    header = ["Nodes", "Edges", "Level J", "DoF", "MG-Cycles", "Time [s]"]
    
    table_data = hcat(nodes_arr, edges_arr, j_arr, dof_arr, cycles_arr, round.(time_arr, digits=3))
    
    pretty_table(
        table_data; 
        column_labels = header, 
        title = "Multigrid Performance on Barabasi-Albert Graphs",
        fit_table_in_display_vertically=false,
        fit_table_in_display_horizontally=false
    )
    return table_data
end


"""
    MG_examples_varying_parameters(; do_plots=false)

Generates numerical results for the multigrid method evaluating varying smoothing
parameters (μ, ν_1, ν_2). Evaluates on a Barabási-Albert graph configuration and plots convergence metrics.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).
"""
function MG_examples_varying_parameters(; do_plots=false)
    Γ = barabasi_albert(2500, 13, complete=true, seed=1)

    Levels = repeat([6, 7], floor(Int, ne(Γ) / 2))
    J = 7

    edge_length = repeat([1.0, 2.0], floor(Int, ne(Γ) / 2))
    if length(Levels) != ne(Γ)
        push!(Levels, 6)
        push!(edge_length, 1)
    end

    estimated_dofs = estimate_dofs(Γ, Levels)
    sparse_index_type = choose_sparse_index_type(estimated_dofs)
    @info "Estimated DoFs: $estimated_dofs | sparse index type: $sparse_index_type"

    HEE, HEV, HVV = createH(Γ, Levels, edge_length, 0.1; index_type=sparse_index_type)
    f_exakt(x, k, Pot) = cos(2 * pi * x/edge_length[k] * (k % 4))
    fe, fv = righthandside(f_exakt, Γ, Levels, edge_length, 0.1; index_type=sparse_index_type)

    Grade = degree(Γ)
    Scale_HEV = Rest_HEV_setup(HEV; index_type=sparse_index_type)

    smooth_vec_V = [[(1, 1), (3, 1), (5, 1), (7, 1)],
        [(1, 3), (3, 3), (5, 3), (7, 3)],
        [(5, 7), (7, 5), (7, 7)]]

    smooth_vec_W = [[(1, 1), (4, 1), (3, 3), (5, 3)]]

    for mu in [1, 2]
        mu_s = "MG"
        smooth_vec = []
        lim_y = 0.3
        if mu == 1
            smooth_vec = smooth_vec_V
            mu_s = "V"
        elseif mu == 2
            smooth_vec = smooth_vec_W
            mu_s = "W"
        end

        for (savez, set) in enumerate(smooth_vec)
            plotvec = []
            if set == [(5, 7), (7, 5), (7, 7)]
                lim_y = 0.2
            end

            for (count, (v1, v2)) in enumerate(set)
                ticksvec = []

                ue = ones(size(fe))
                uv = ones(size(fv))

                rate_e_vec = []
                rate_v_vec = []
                rate_vec = []
                kmax = 0

                while ((kmax < 250) && (sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2)) > 10^(-8))

                    ue, uv = Multigrid_Graph(HEE, HEV, HVV, Levels, J, fv, fe,
                        uv, ue, ne(Γ), Scale_HEV, v1, v2, 0.5, mu)

                    push!(rate_vec, sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - transpose(HEV) * ue - HVV * uv)^2))
                    push!(rate_e_vec, norm(fe - HEE * ue - HEV * uv))
                    push!(rate_v_vec, norm(fv - transpose(HEV) * ue - HVV * uv))

                    kmax += 1
                end

                if kmax - 1 > 9
                    ticksvec = 2:2:kmax
                else
                    ticksvec = 2:1:kmax
                end
                
                if do_plots
                    if count == length(set)
                        Plot_Res = plot(2:1:kmax, [rate_vec[2:end] ./ rate_vec[1:end-1], rate_e_vec[2:end] ./ rate_e_vec[1:end-1],
                                rate_v_vec[2:end] ./ rate_v_vec[1:end-1]],
                            label=["convergence rate" "rate on internal vertices" "rate on vertices"],
                            title=L"$\nu_1$: %$v1; $\nu_2$: %$v2 - %$mu_s-cycles: %$kmax",
                            legend=:topleft, lw=5, tickfontsize=14, ylims=(0, lim_y), titlefontsize=20,
                            legendfontsize=16, xticks=ticksvec, linestyle=[:solid :dash :dash], dpi=1200)
                    else
                        Plot_Res = plot(2:1:kmax, [rate_vec[2:end] ./ rate_vec[1:end-1], rate_e_vec[2:end] ./ rate_e_vec[1:end-1],
                                rate_v_vec[2:end] ./ rate_v_vec[1:end-1]],
                            title=L"$\nu_1$: %$v1; $\nu_2$: %$v2 - %$mu_s-cycles: %$kmax",
                            legend=false, lw=5, tickfontsize=14, ylims=(0, lim_y),
                            titlefontsize=20, xticks=ticksvec, linestyle=[:solid :dash :dash], dpi=1200)
                    end
                    push!(plotvec, Plot_Res)
                end
            end
            
            if do_plots
                Plot_ges = plot(plotvec..., size=(450 * length(set), 650),
                    layout=(1, length(set)), margin=5Plots.mm, dpi=1200)
                if !isdir(joinpath("Figures", "MG_examples_varying_parameters"))
                    mkpath(joinpath("Figures", "MG_examples_varying_parameters"))
                end
                savefig(Plot_ges, joinpath("Figures", "MG_examples_varying_parameters", "vergleichsplots$mu$savez.pdf"))
            end
        end
    end
end


"""
    PCG_example()

Compares standard Conjugate Gradient (CG) performance versus Preconditioned Conjugate Gradient (PCG) applied to the Schur complement formulation on a Barabási-Albert graph.
"""
function PCG_example(;vertices_vec = [1000, 1500, 2000])

    Pot = 0.1
    
    edges_vec = zeros(Int, length(vertices_vec))
    CG_iter = zeros(Int, length(vertices_vec))
    PCG_iter = zeros(Int, length(vertices_vec))

    for i in 1:length(vertices_vec)
        @show vertices_vec[i]
        
        Γ = barabasi_albert(vertices_vec[i], 13, complete=true, seed=1)
        edges_vec[i] = ne(Γ)
        Levels = repeat([6, 7], floor(Int, ne(Γ) / 2))

        edge_length = repeat([1.0, 2.0], floor(Int, ne(Γ) / 2))
        if length(Levels) != ne(Γ)
            push!(Levels, 5)
            push!(edge_length, 1)
        end
        
        HEE, HEV, HVV = createH(Γ, Levels, edge_length, 0.1)
        f_exakt(x, k, Pot) = cos(2 * pi * x/edge_length[k] * (k % 4))
        fe, fv = righthandside(f_exakt, Γ, Levels, edge_length, 0.1)
        f_Schur = fv - transpose(HEV) * (HEE\fe)

        Dinv = Diagonal(1 ./sqrt.(degree(Γ)))

        ucg, iter_cg = PCG_mod(HEE, HEV, HVV, f_Schur, 
                            zeros(length(f_Schur)), 10^(-8); Dinv=I(nv(Γ)))
        upcg, iter_pcg = PCG_mod(HEE, HEV, HVV, f_Schur,
                            zeros(length(f_Schur)), 10^(-8); Dinv=Dinv)
        CG_iter[i] = iter_cg
        PCG_iter[i] = iter_pcg
    end
    PCGTab = pretty_table(hcat(vertices_vec, edges_vec, CG_iter, PCG_iter); 
            column_labels=["n", "m", "iterations CG", "iterations PCG"])

end


"""
    MG_example_AdvectionDiffusion(v1::Int=3, v2::Int=3; do_plots=false)

Testcase demonstrating the multigrid approach applied to the advection-diffusion equation on a star graph. Utilizes SUPG stabilization, matrix-dependent intergrid operators, and appropriate GMRES smoothing for asymmetric systems.

# Arguments
- `v1::Int`: Number of pre-smoothing steps (default: 3).
- `v2::Int`: Number of post-smoothing steps (default: 3).
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
- `ue::Vector{Float64}`: Solution vector for the internal edge nodes.
- `uv::Vector{Float64}`: Solution vector for the graph vertices.
"""
function MG_example_AdvectionDiffusion(v1::Int=3, v2::Int=3; do_plots=false)

    Γ = reverse(star_digraph(4))
    J = 8
    Levels = J * ones(Int, ne(Γ))
    edgelength = 1.0 * ones(ne(Γ))

    D_edge = 0.1 * ones(ne(Γ)) 
    A_edge = 8.0  * ones(ne(Γ))

    f_exakt(x, k) = (
        if k == 1
            return sin(pi * x)
        else
            return 1.0
        end
    )

    problem_nodes = check_inflow_vertices(Γ, A_edge)
    
    dirichlet_nodes = [2,3] 
    dirichlet_values = [0.0,0.0]
    
    if !isempty(problem_nodes)
        fix_inflow_nodes!(Γ, problem_nodes, Levels, edgelength, D_edge, A_edge, dirichlet_nodes, dirichlet_values, L_new = 1, D_new = 0.1, A_new = 0)
    end

    HEE, HEV, HVE, HVV, fe, fv = createH_AdvectionDiffusion(Γ, Levels, edgelength, D_edge, A_edge, f_exakt; use_SUPG=true)

    
    HEE, HEV, HVE, HVV, fe, fv = apply_dirichlet_blocks!(HEE, HEV, HVE, HVV, fe, fv, 
                                                        dirichlet_nodes, dirichlet_values)
    
    macroscopic_order = get_macroscopic_sort(Γ, A_edge)
    mg_hierarchy = setup_advection_hierarchy(HEE, HEV, HVE, HVV, Levels, J, Γ, A_edge, macroscopic_order)
    Scale_HEV = Rest_HEV_setup(HEV)
    ue = ones(size(fe))
    uv = ones(size(fv))

    for (i, v) in enumerate(dirichlet_nodes)
        uv[v] = dirichlet_values[i]
    end

    rate_e_vec = []
    rate_v_vec = []
    rate_vec = []
    kmax = 0

    while ((kmax < 250) && (sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - HVE * ue - HVV * uv)^2)) > 10^(-10)) 

        ue, uv = solve_advection_MG!(mg_hierarchy, J, fv, fe, uv, ue, v1, v2, 1)

        res_e = norm(fe - HEE * ue - HEV * uv)
        res_v = norm(fv - HVE * ue - HVV * uv)
        res_total = sqrt(res_e^2 + res_v^2)
        
        push!(rate_vec, res_total) 
        push!(rate_e_vec, res_e) 
        push!(rate_v_vec, res_v) 

        kmax += 1 
        
        if kmax % 5 == 0
            println("Cycle $kmax: Residual = $res_total")
        end
    end

    if do_plots
        Plot_Res = plot(2:1:kmax, [rate_vec[2:end] ./ rate_vec[1:end-1], rate_e_vec[2:end] ./ rate_e_vec[1:end-1], 
                rate_v_vec[2:end] ./ rate_v_vec[1:end-1]], 
            label=["Overall rate" "Rate internal nodes" "Rate vertices"], xlabel="cycle", ylabel="Convergence rate", 
            title="Advection-Diffusion MG cycles: $kmax", legend=:topright, lw=3, tickfontsize=8, linestyle=[:solid :dash :dash], dpi=300) 
        
        if !isdir(joinpath("Figures", "MG_example_AdvectionDiffusion"))
            mkpath(joinpath("Figures", "MG_example_AdvectionDiffusion"))
        end
        savefig(Plot_Res, joinpath("Figures", "MG_example_AdvectionDiffusion", "convergence_advection_diffusion.pdf"))
        
        int_nodes = max.(2 .^ Levels .- 1, 0)

        ue_split = Array{Any}(undef, ne(Γ))
        lgt = 1
        for j = 1:ne(Γ)
            ue_split[j] = ue[lgt : lgt + int_nodes[j] - 1]
            lgt += int_nodes[j]
        end

        coords_v = Vector{Vector{Float64}}(undef, nv(Γ))
        coords_v[1] = [0.0, 0.0] 
        
        N_orig = 4
        for i in 2:N_orig
            angle = 2 * pi * (i - 2) / (N_orig - 1)
            r = edgelength[i-1]
            coords_v[i] = [r * cos(angle), r * sin(angle)]
        end
        
        for i in (N_orig + 1):nv(Γ)
            p_node = all_neighbors(Γ, i)[1]
            
            edge_idx = 0
            for (m, e) in enumerate(edges(Γ))
                if (src(e) == p_node && dst(e) == i) || (src(e) == i && dst(e) == p_node)
                    edge_idx = m
                    break
                end
            end
            L_new = edgelength[edge_idx]
            
            if p_node == 1
                coords_v[i] = [L_new, 0.0] 
            else
                dir = coords_v[p_node] ./ norm(coords_v[p_node])
                coords_v[i] = coords_v[p_node] .+ dir .* L_new
            end
        end
        plt_graph = plot_graph_3d(Γ, edgelength, coords_v, ue_split, uv, int_nodes)
        savefig(plt_graph, joinpath("Figures", "MG_example_AdvectionDiffusion", "solution_advection_diffusion_3d.pdf"))
    end
    return ue, uv 
end


"""
    run_MG_stargraph(; do_plots=false)

Execution runner that specifically sets up and solves a basic star graph problem 
formulated via `testcase_stargraph`. Evaluates exact and numerical differences.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
- `ue::Vector{Float64}`: Solved internal edge variables.
- `uv::Vector{Float64}`: Solved vertex variables.
"""
function run_MG_stargraph(; do_plots=false)
    my_case = testcase_stargraph(8.0, 0.1, 6; dirichlet_nodes=[2,3,4], dirichlet_values=[0.0,0.0,0.0])

    ue, uv, residuen, _, _ = run_MG_from_case(my_case; v1=3, v2=3)

    check_vertex_flux_condition_MG(ue, uv, my_case)

    if !isnothing(my_case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, my_case)
        println("Result L2-Error:")
        println("Absolute: $L2_abs")
        println("Relative: $L2_rel")
    end

    if do_plots
        plt_3d = plot_case_3d(my_case, ue, uv)
        if !isdir(joinpath("Figures", "run_MG_stargraph"))
            mkpath(joinpath("Figures", "run_MG_stargraph"))
        end
        savefig(plt_3d, joinpath("Figures", "run_MG_stargraph", "solution_stargraph_3d.pdf"))
    end
    return ue, uv 
end


"""
    run_MG_DiffAdv_example(; do_plots=false)

Tests an exact solution triangle graph for the diffusion-advection multigrid logic.
It verifies flux conservation, plots 3D visual outputs, and analyzes numerical-exact differences.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).
"""
function run_MG_DiffAdv_example(; do_plots=false)
    triangle_case = testcase_triangle_exact_WOx(8.0, 0.1, 8; show_plots_exact=false)
    ue, uv, residuen, _, _ = run_MG_from_case(triangle_case; 
                                        v1=3, v2=3, 
                                        A_fix=0, D_fix=0.1, L_fix=1, autofix=true)

    check_vertex_flux_condition_MG(ue, uv, triangle_case)
    
    if triangle_case.exakte_Loesung !== nothing
        uv_exakt = zeros(triangle_case.nv)
        ue_exakt = Float64[]

        for (i, e) in enumerate(triangle_case.edges)
            x_nodes = triangle_case.edge_x[e]
            
            u_vals = [triangle_case.exakte_Loesung(x)[i] for x in x_nodes]
            uv_exakt[src(e)] = u_vals[1]
            uv_exakt[dst(e)] = u_vals[end]
            
            append!(ue_exakt, u_vals[2:end-1])
        end

        N_ue_orig = length(ue_exakt)
        N_uv_orig = length(uv_exakt)

        ue_err = abs.(ue[1:N_ue_orig] .- ue_exakt)
        uv_err = abs.(uv[1:N_uv_orig] .- uv_exakt)
        
        max_err = max(maximum(ue_err), maximum(uv_err))
        println("=> Maximum absolute error: ", round(max_err, sigdigits=4))
        
        if do_plots
            plt = plot_case_3d(triangle_case, ue, uv)
            title!(plt, "Numerical Solution")
            if !isdir(joinpath("Figures", "run_MG_DiffAdv_example"))
                mkpath(joinpath("Figures", "run_MG_DiffAdv_example"))
            end
            savefig(plt, joinpath("Figures", "run_MG_DiffAdv_example", "solution_diff_adv_3d.pdf")) 

            plt_exakt = plot_case_3d(triangle_case, ue_exakt, uv_exakt)
            title!(plt_exakt, "Exact solution")
            savefig(plt_exakt, joinpath("Figures", "run_MG_DiffAdv_example", "exact_solution_diff_adv_3d.pdf"))

            plt_err = plot_case_3d(triangle_case, ue_err, uv_err)
            title!(plt_err, "Absolute Error |u_h - u_exakt|")
            savefig(plt_err, joinpath("Figures", "run_MG_DiffAdv_example", "absolute_error_diff_adv_3d.pdf"))
        end
    else
        @warn "No exact solution available."
    end

    if !isnothing(triangle_case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, triangle_case)
        println("Result L2-Error:")
        println("  Absolute: $L2_abs")
        println("  Relative: $L2_rel")
    end

    if do_plots
        if !isdir(joinpath("Figures", "run_MG_DiffAdv_example"))
            mkpath(joinpath("Figures", "run_MG_DiffAdv_example"))
        end
        pltnumvex = plot_case_num_vs_exact(triangle_case, ue, uv, "Numerical vs. Exact")
        savefig(pltnumvex, joinpath("Figures", "run_MG_DiffAdv_example", "numerical_vs_exact_diff_adv.pdf"))

        plt_local_error = plot_case_edge_difference(triangle_case, ue, uv, "Local Error Distribution")
        savefig(plt_local_error, joinpath("Figures", "run_MG_DiffAdv_example", "local_error_diff_adv.pdf"))
    end
end


"""
    run_barabasi_example()

Simple test invocation running an advection-diffusion multigrid resolution across 
a smaller setup of the Barabási-Albert model (2000 nodes). Evaluates basic properties.
"""
function run_barabasi_example()
    my_barabasi_case = testcase_barabasi(2000, 5.0, 0.1, 6; 
                                        dirichlet_nodes=[1, 20], 
                                        dirichlet_values=[0.0, 1.0])

    ue, uv, residuen, _, _ = run_MG_from_case(my_barabasi_case; 
                                        v1=3, v2=3, 
                                        A_fix=5.0, D_fix=0.01, L_fix=1.0, use_SUPG=true)

    if !isnothing(my_barabasi_case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, my_barabasi_case)
        println("Result L2-Error:")
        println("Absolute: $L2_abs")
        println("Relative: $L2_rel")
    end
end


"""
    run_barabasi_example_with_cycle()

Runs a directed Barabási-Albert configuration test utilizing cycle boundaries. Checks cyclic loop resolutions.
"""
function run_barabasi_example_with_cycle()
    my_barabasi_case = testcase_barabasi_with_cycle(2000, 5.0, 1, 6; 
                                        dirichlet_nodes=[1, 20], 
                                        dirichlet_values=[0.0, 1.0])

    ue, uv, residuen, _, _ = run_MG_from_case(my_barabasi_case; 
                                        v1=7, v2=5, 
                                        A_fix=0.01, D_fix=1.0, L_fix=1.0, use_SUPG=true)

    if !isnothing(my_barabasi_case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, my_barabasi_case)
        println("Result L2-Error:")
        println("Absolute: $L2_abs")
        println("Relative: $L2_rel")
    end
end


"""
    run_star_KumarLeugering(; do_plots=false)

Applies the specialized analytical star-graph benchmark (Kumar-Leugering model) 
and investigates accurate resolution alongside pointwise error derivation.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).
"""
function run_star_KumarLeugering(; do_plots=false)
    case = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, 8, layer_mesh=:uniform)

    ue, uv, residuen, _, _ = run_MG_from_case(case; 
                                        v1=3, v2=3, autofix=false, use_SUPG=true)

    check_vertex_flux_condition_MG(ue, uv, case)

    if do_plots
        plt = plot_case_3d(case, ue, uv)
        title!(plt, "Numerical solution")
        if !isdir(joinpath("Figures", "run_star_KumarLeugering"))
            mkpath(joinpath("Figures", "run_star_KumarLeugering"))
        end
        savefig(plt, joinpath("Figures", "run_star_KumarLeugering", "Kumar_Leugering_solution_MG.pdf"))
    end

    if case.exakte_Loesung !== nothing
        uv_exakt = zeros(case.nv)
        ue_exakt = Float64[]

        for (i, e) in enumerate(case.edges)
            x_nodes = case.edge_x[e]
            
            u_vals = [case.exakte_Loesung(x)[i] for x in x_nodes]
            
            uv_exakt[src(e)] = u_vals[1]
            uv_exakt[dst(e)] = u_vals[end]
            
            append!(ue_exakt, u_vals[2:end-1])
        end

        N_ue_orig = length(ue_exakt)    
        N_uv_orig = length(uv_exakt)

        ue_err = abs.(ue[1:N_ue_orig] .- ue_exakt)
        uv_err = abs.(uv[1:N_uv_orig] .- uv_exakt)
        
        max_err = max(maximum(ue_err), maximum(uv_err))
        println("Maximum error: ", round(max_err, sigdigits=4))
        
        if do_plots
            plt_exakt = plot_case_3d(case, ue_exakt, uv_exakt)
            savefig(plt_exakt, joinpath("Figures", "run_star_KumarLeugering", "Kumar_Leugering_solution_exact.pdf"))
            p_both = plot(plt, plt_exakt, size=(450, 900), layout=(2, 1), margin=5Plots.mm, dpi=1200)

            plt_err = plot_case_3d(case, ue_err, uv_err)
            title!(plt_err, "Absolute error |u_h - u|")
            savefig(plt_err, joinpath("Figures", "run_star_KumarLeugering", "MG_Kumar_Leugering_solution_error.pdf"))
        end
    else
        @warn "No exact solution available for error analysis."
    end

    if !isnothing(case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, case)
        println("Result L2-Error:")
        println("Absolute: $L2_abs")
        println("Relative: $L2_rel")
    end

    if do_plots
        p = plot_edges_2d_simple(case, ue, uv)
        savefig(p, joinpath("Figures", "run_star_KumarLeugering", "Kumar_Leugering_solution_edgewise.pdf"))
    end
end


"""
    plot_convergence_rate_shishkin(; do_plots=false)

Computes and illustrates the convergence behavior specifically focused on Shishkin and Bakhvalov meshes
incorporating high advection-diffusion layers.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
- `plt_conv`: Plotted convergence rate graph representation, or `nothing` if plots are disabled.
- `h_values::Vector`: Approximate spatial refinement increments.
- `errors_L2::Vector`: Computed absolute convergence errors in corresponding refinement runs.
"""
function plot_convergence_rate_shishkin(; do_plots=false)
    N_values = 2:16
    errors_L2 = Float64[]
    
    for N in N_values
        case = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, N, layer_mesh=:bakhvalov, shishkin_mode=:outflow, ρ=1.0)

        ue, uv = run_star_shishkin_block_backslash(case)
        if !isnothing(case.exakte_Loesung)
            L2_abs, L2_rel = calculate_case_D_error(ue, uv, case)
            push!(errors_L2, L2_abs)
            println("N = 2^$N: L2 Error = $L2_abs")
        else
            error("Exact solution missing for convergence plot!")
        end
    end
    h_values = 1.0 ./ (2.0.^N_values)

    plt_conv = nothing

    if do_plots
        ref_O1 = h_values .* (errors_L2[1] / h_values[1]) 
        ref_O2 = (h_values.^2) .* (errors_L2[1] / h_values[1]^2)
        ref_ln_O1 = (h_values .* log.(h_values.^-1)) .* (1.35*errors_L2[1]  / (h_values[1] * log(h_values[1].^-1)))

        plt_conv = plot(h_values, errors_L2, 
                        m=:square, lw=2, 
                        label="Calculated Error in ||.||_D norm", 
                        xaxis=:log10, yaxis=:log10,
                        xlabel="Mesh size h (approx. 1/N)", 
                        ylabel="Absolute Error",
                        title="Convergence Rate",
                        legend=:bottomright,
                        dpi=600)
                        
        plot!(plt_conv, h_values, ref_O1, lw=2, ls=:dash, color=:gray, label="O(h) Reference")
        plot!(plt_conv, h_values, ref_O2, lw=2, ls=:dot, color=:gray, label="O(h^2) Reference")
        plot!(plt_conv, h_values, ref_ln_O1, lw=2, ls=:dashdot, color=:gray, label="O(h log(h⁻¹)) Reference", m=:circle)
        if !isdir(joinpath("Figures", "plot_convergence_rate_shishkin"))
            mkpath(joinpath("Figures", "plot_convergence_rate_shishkin"))
        end
        savefig(plt_conv, joinpath("Figures", "plot_convergence_rate_shishkin", "Shishkin_Convergence_bakhvalov.pdf"))
    end

    return plt_conv, h_values, errors_L2
end


"""
    plot_convergence_rate_SUPG(; do_plots=false)

Conducts a convergence review utilizing (SUPG) stabilization methodology on standard generated meshes.

# Arguments
- `do_plots::Bool`: Flag to enable or disable intermediate solution visualizations (default: false).

# Output
- `plt_conv`: Plotted convergence rate graph representation, or `nothing` if plots are disabled.
- `h_values::Vector`: Approximate spatial refinement increments.
- `errors_L2::Vector`: Computed absolute convergence errors.
"""
function plot_convergence_rate_SUPG(; do_plots=false)
    case = testcase_star_MMS_smooth(1e-6, 1e-6, 1e-6, 6)
    ue, uv, residuen, _, _ = run_MG_from_case(case; 
                                v1=3, v2=3, autofix=false, use_SUPG=false,
                                use_additional_SUPG=false)
    if do_plots
        plt = plot_case_3d(case, ue, uv)
        title!(plt, "Numerical solution, N=6")
        if !isdir(joinpath("Figures", "plot_convergence_rate_SUPG"))
            mkpath(joinpath("Figures", "plot_convergence_rate_SUPG"))
        end
        savefig(plt, joinpath("Figures", "plot_convergence_rate_SUPG", "Tiny_diff_numerical.pdf"))
    end
    N_values = 2:10
    errors_L2 = Float64[]
    
    for N in N_values
        case = testcase_star_MMS_smooth(1e-6, 1e-6, 1e-6, N)

        ue, uv, residuen, _, _ = run_MG_from_case(case; 
                                    v1=3, v2=3, autofix=false, use_SUPG=true, use_additional_SUPG=true)
        if do_plots
            plt = plot_case_3d(case, ue, uv)
            title!(plt, "Numerical solution, N=$N")
            savefig(plt, joinpath("Figures", "plot_convergence_rate_SUPG", "Tiny_diff_numerical_N$N.pdf"))
        end

        if !isnothing(case.exakte_Loesung)
            L2_abs, L2_rel = calculate_case_D_error(ue, uv, case, use_SUPG=true)
            push!(errors_L2, L2_abs)
            println("N = 2^$N: L2 Error = $L2_abs")
        else
            error("Exact solution missing for convergence plot!")
        end

        if case.exakte_Loesung !== nothing 
            uv_exakt = zeros(case.nv)
            ue_exakt = Float64[]

            for (i, e) in enumerate(case.edges)
                x_nodes = case.edge_x[e]
            
                u_vals = [case.exakte_Loesung(x)[i] for x in x_nodes]
                uv_exakt[src(e)] = u_vals[1]
                uv_exakt[dst(e)] = u_vals[end]
                
                append!(ue_exakt, u_vals[2:end-1])
            end

            N_ue_orig = length(ue_exakt)    
            N_uv_orig = length(uv_exakt)

            ue_err = abs.(ue[1:N_ue_orig] .- ue_exakt)
            uv_err = abs.(uv[1:N_uv_orig] .- uv_exakt)
            
            max_err = max(maximum(ue_err), maximum(uv_err))
            println("=> Maximum absolute discretization error: ", round(max_err, sigdigits=4))

            if do_plots
                plt_exakt = plot_case_3d(case, ue_exakt, uv_exakt)
                title!(plt_exakt, "Exact analytical solution")
                savefig(plt_exakt, joinpath("Figures", "plot_convergence_rate_SUPG", "Tiny_diff_exact.pdf"))
                
                p_both = plot(plt, plt_exakt, size=(450, 900), layout=(2, 1), margin=5Plots.mm, dpi=1200)
                savefig(p_both, joinpath("Figures", "plot_convergence_rate_SUPG", "Kumar_Leugering_solution_comparison.pdf"))

                plt_err = plot_case_3d(case, ue_err, uv_err)
                title!(plt_err, "Absolute Error |u_h - u_exact|")
                savefig(plt_err, joinpath("Figures", "plot_convergence_rate_SUPG", "Tiny_diff_absolute_error.pdf"))
            end
        else
            @warn "No exact solution is available for this case."
        end
    end

    h_values = 1.0 ./ (2.0.^N_values)
    plt_conv = nothing

    if do_plots
        ref_O1 = h_values .* (errors_L2[1] / h_values[1])      
        ref_O2 = (h_values.^2) .* (errors_L2[1] / h_values[1]^2) 
        ref_O32 = (h_values .^(3/2)) .* (1.5*errors_L2[1]  / h_values[1]^(3/2)) 
        ref_O1log = (h_values .* log.(1.0 ./ h_values)) .* (errors_L2[1] / (h_values[1] * log(1.0 / h_values[1])))

        plt_conv = plot(h_values, errors_L2, 
                        m=:square, lw=2, 
                        label="Calculated Error in ||.||_D norm", 
                        xaxis=:log10, yaxis=:log10,
                        xlabel="Mesh size h (approx. 1/N)", 
                        ylabel="Absolute Error",
                        title="Convergence Rate",
                        legend=:bottomright,
                        dpi=600)
                        
        plot!(plt_conv, h_values, ref_O1, lw=2, ls=:dash, color=:gray, label="O(h) Reference")
        plot!(plt_conv, h_values, ref_O2, lw=2, ls=:dot, color=:gray, label="O(h^2) Reference")
        plot!(plt_conv, h_values, ref_O32, lw=2, ls=:dashdot, color=:gray, label="O(h^(3/2)) Reference", m=:circle)
        plot!(plt_conv, h_values, ref_O1log, lw=2, ls=:dashdotdot, color=:gray, label="O(h log(1/h)) Reference")
        
        if !isdir(joinpath("Figures", "plot_convergence_rate_SUPG"))
            mkpath(joinpath("Figures", "plot_convergence_rate_SUPG"))
        end
        savefig(plt_conv, joinpath("Figures", "plot_convergence_rate_SUPG", "SUPG_Convergence.pdf"))
    end

    return plt_conv, h_values, errors_L2
end


"""
    plot_convergence_rate_SUPG_comparison(; do_plots=false)

Compares algorithmic SUPG convergence boundaries structurally side-by-side with variants. Validates if additional SUPG node projections stabilize or limit metric behaviors.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
- Returns plotted convergence rates and raw arrays tracking internal iteration accuracy metrics.
"""
function plot_convergence_rate_SUPG_comparison(; do_plots=false)
    case_init = testcase_star_MMS_smooth(1e-6, 1e-6, 1e-6, 6, A3 = 1.0)
    ue_init, uv_init, res_init, _, _ = run_MG_from_case(case_init; 
                                                v1=3, v2=3, autofix=false, use_SUPG=false,
                                                use_additional_SUPG=false)
    if do_plots
        plt_init = plot_case_3d(case_init, ue_init, uv_init)
        title!(plt_init, "Numerical solution, N=6")
        if !isdir(joinpath("Figures", "plot_convergence_rate_SUPG"))
            mkpath(joinpath("Figures", "plot_convergence_rate_SUPG"))
        end
        savefig(plt_init, joinpath("Figures", "plot_convergence_rate_SUPG", "Initial_Numerical_Solution_N6.pdf"))
    end

    N_values = 2:10
    
    errors_L2_without = Float64[]
    errors_L2_with = Float64[]
    
    for N in N_values
        case = testcase_star_MMS_smooth(1e-6, 1e-6, 1e-6, N, A3 = 1.0)

        ue_no, uv_no, _, _, _ = run_MG_from_case(case; 
                                        v1=3, v2=3, autofix=false, 
                                        use_SUPG=true, use_additional_SUPG=false)
        
        ue_yes, uv_yes, _, _, _ = run_MG_from_case(case; 
                                            v1=3, v2=3, autofix=false, 
                                            use_SUPG=true, use_additional_SUPG=true)

        if do_plots
            plt_no = plot_case_3d(case, ue_no, uv_no)
            title!(plt_no, "Numerical solution (Without add. SUPG), N=$N")
            savefig(plt_no, joinpath("Figures", "plot_convergence_rate_SUPG", "SUPG_Comparison_NoSUPG_N$N.pdf"))

            plt_yes = plot_case_3d(case, ue_yes, uv_yes)
            title!(plt_yes, "Numerical solution (With add. SUPG), N=$N")
            savefig(plt_yes, joinpath("Figures", "plot_convergence_rate_SUPG", "SUPG_Comparison_WithSUPG_N$N.pdf"))
        end

        if !isnothing(case.exakte_Loesung)
            L2_abs_no, L2_rel_no = calculate_case_D_error(ue_no, uv_no, case, use_SUPG=true)
            push!(errors_L2_without, L2_abs_no)
            
            L2_abs_yes, L2_rel_yes = calculate_case_D_error(ue_yes, uv_yes, case, use_SUPG=true)
            push!(errors_L2_with, L2_abs_yes)
            
            println("N = 2^$N:")
            println("  -> L2 Error (without add. SUPG) = $L2_abs_no")
            println("  -> L2 Error (with add. SUPG)    = $L2_abs_yes")
        else
            error("Exact solution missing for convergence plot!")
        end
        
        if case.exakte_Loesung !== nothing 
            uv_exakt = zeros(case.nv)
            ue_exakt = Float64[]

            for (i, e) in enumerate(case.edges)
                x_nodes = case.edge_x[e]
                u_vals = [case.exakte_Loesung(x)[i] for x in x_nodes]
                uv_exakt[src(e)] = u_vals[1]
                uv_exakt[dst(e)] = u_vals[end]
                append!(ue_exakt, u_vals[2:end-1])
            end

            N_ue_orig = length(ue_exakt)    
            N_uv_orig = length(uv_exakt)

            ue_err = abs.(ue_no[1:N_ue_orig] .- ue_exakt)
            uv_err = abs.(uv_no[1:N_uv_orig] .- uv_exakt)
            
            ue_err_yes = abs.(ue_yes[1:N_ue_orig] .- ue_exakt)
            uv_err_yes = abs.(uv_yes[1:N_uv_orig] .- uv_exakt)
            
            max_err = max(maximum(ue_err_yes), maximum(uv_err_yes))
            println("Maximum absolute discretization error (With add. SUPG): ", round(max_err, sigdigits=4))

            if do_plots
                plt_exakt = plot_case_3d(case, ue_exakt, uv_exakt)
                title!(plt_exakt, "Exact analytical solution")
                savefig(plt_exakt, joinpath("Figures", "plot_convergence_rate_SUPG", "SUPG_Comparison_Exact_N$N.pdf"))
                
                plt = plot_case_3d(case, ue_yes, uv_yes)
                title!(plt, "Numerical solution (With add. SUPG), N=$N")
                p_both = plot(plt, plt_exakt, size=(450, 900), layout=(2, 1), margin=5Plots.mm, dpi=1200)

                plt_err_no = plot_case_3d(case, ue_err, uv_err)
                title!(plt_err_no, "Absolute Error |u_h - u_exakt| (No add. SUPG)")
                savefig(plt_err_no, joinpath("Figures", "plot_convergence_rate_SUPG", "SUPG_Comparison_Error_NoSUPG_N$N.pdf"))
                
                plt_err_yes = plot_case_3d(case, ue_err_yes, uv_err_yes)
                title!(plt_err_yes, "Absolute Error |u_h - u_exakt| (With add. SUPG)")
                savefig(plt_err_yes, joinpath("Figures", "plot_convergence_rate_SUPG", "SUPG_Comparison_Error_WithSUPG_N$N.pdf"))
            end
        else
            @warn "No exact solution is provided for this case."
        end
    end
    
    h_values = 1.0 ./ (2.0 .^ N_values)
    rates_no = zeros(length(h_values))
    rates_yes = zeros(length(h_values))
    
    rates_no[1] = NaN
    rates_yes[1] = NaN
    
    for i in 2:length(h_values)
        rates_no[i]  = log(errors_L2_without[i-1] / errors_L2_without[i]) / log(h_values[i-1] / h_values[i])
        rates_yes[i] = log(errors_L2_with[i-1] / errors_L2_with[i]) / log(h_values[i-1] / h_values[i])
    end

    println("\nConvergence Rates Comparison:")
    println("-" ^ 95)
    @printf("%-15s | %-10s | %-18s | %-10s | %-18s | %-10s\n", 
            "N (Elements)", "h", "Error (Standard)", "Rate", "Error (Additional)", "Rate")
    println("-" ^ 95)
    
    for i in 1:length(N_values)
        val_N     = string(2^N_values[i])
        val_h     = @sprintf("%.2e", h_values[i])
        val_err_no= @sprintf("%.4e", errors_L2_without[i])
        val_rat_no= i == 1 ? "-" : @sprintf("%.4f", rates_no[i])
        val_err_yes=@sprintf("%.4e", errors_L2_with[i])
        val_rat_yes=i == 1 ? "-" : @sprintf("%.4f", rates_yes[i])
        
        @printf("%-15s | %-10s | %-18s | %-10s | %-18s | %-10s\n", 
                val_N, val_h, val_err_no, val_rat_no, val_err_yes, val_rat_yes)
    end
    println("-" ^ 95)
    
    plt_conv = nothing

    if do_plots
        anchor_err = errors_L2_without[1]
        ref_O1 = h_values .* (anchor_err / h_values[1])               
        ref_O2 = (h_values.^2) .* (anchor_err / h_values[1]^2)        
        ref_O32 = (h_values .^(3/2)) .* (1.5 * anchor_err / h_values[1]^(3/2)) 

        plt_conv = plot(h_values, errors_L2_without, 
                        m=:circle, lw=2, 
                        label="Without add. SUPG", 
                        xaxis=:log10, yaxis=:log10,
                        xlabel="Mesh size h (approx. 1/N)", 
                        ylabel="Absolute Error ||.||_D",
                        title="SUPG Convergence Rate Comparison",
                        legend=:bottomright,
                        dpi=600)
                        
        plot!(plt_conv, h_values, errors_L2_with, ls =:dash,
            lw=2, 
            label="With add. SUPG")
                        
        plot!(plt_conv, h_values, ref_O1, lw=2, ls=:dash, color=:gray, label="O(h) Reference")
        plot!(plt_conv, h_values, ref_O2, lw=2, ls=:dot, color=:gray, label="O(h^2) Reference")
        plot!(plt_conv, h_values, ref_O32, lw=2, ls=:dashdot, color=:gray, label="O(h^(3/2)) Reference", m=:utriangle)
        
        if !isdir(joinpath("Figures", "plot_convergence_rate_SUPG_comparison"))
            mkpath(joinpath("Figures", "plot_convergence_rate_SUPG_comparison"))
        end
        savefig(plt_conv, joinpath("Figures", "plot_convergence_rate_SUPG_comparison", "SUPG_Convergence_Comparison.pdf"))
    end

    return plt_conv, h_values, errors_L2_without, errors_L2_with
end


"""
    run_star_KumarLeugering_dG(; eta::Float64=10.0, p::Int=1, do_plots=false)

Evaluates the Kumar-Leugering graph case specifically mapping the analytical limits using a discontinuous Galerkin (dG) strategy and visualizing interface discontinuities.

# Arguments
- `eta::Float64`: Penalty factor scaling across discontinuous interfaces.
- `p::Int`: Basis polynomial degrees representation factor (default: 1).
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
None. 
"""
function run_star_KumarLeugering_dG(; eta=10.0, p=1, do_plots=false)
    configs = [
        (:uniform, plot_case_3d, "Uniform-DG"),
        (:shishkin, plot_shishkin_3d, "Shishkin-DG")
    ]

    for (mesh_type, plot_func, title_str) in configs

        case = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, 8, layer_mesh=mesh_type, ρ=1.0)

        G = SimpleGraph(case.nv)
        for e in case.edges
            add_edge!(G, src(e), dst(e))
        end

        Levels = [Int(log2(case.n_e[e])) for e in case.edges]
        D_edge = [case.eps_edge[e] for e in case.edges]
        A_edge = [case.a_edge[e] for e in case.edges]
        edge_x = [case.edge_x[e] for e in case.edges]
        
        f_exakt_wrapper(x, m) = case.f_edge[case.edges[m]](x)

        HEE, HEV, HVE, HVV, f_E, f_V = createH_DG_AdvectionDiffusion(
            G, Levels, edge_x, D_edge, A_edge, f_exakt_wrapper;
            eta=eta, p=p, use_SUPG=false
        )

        NE = length(f_E)
        NV = length(f_V)

        A_global = [HEE HEV; 
                    HVE HVV]
        b_global = [f_E; f_V]

        for (node, val) in case.dirichlet
            idx = NE + node 
            A_global[idx, :] .= 0.0
            A_global[idx, idx] = 1.0
            b_global[idx] = val
        end

        u_full = A_global \ b_global

        ue = u_full[1:NE]
        uv = u_full[NE+1:end]

        if case.exakte_Loesung !== nothing
            uv_exakt = zeros(case.nv)
            ue_exakt = Float64[]

            for (m, e) in enumerate(case.edges)
                x_nodes = case.edge_x[e]
                L_edge = x_nodes[end]

                uv_exakt[src(e)] = case.exakte_Loesung(0.0)[m]
                uv_exakt[dst(e)] = case.exakte_Loesung(L_edge)[m]
                
                N_e = case.n_e[e]
                
                for k in 1:N_e
                    x_left = x_nodes[k]
                    x_right = x_nodes[k+1]
                    
                    push!(ue_exakt, case.exakte_Loesung(x_left)[m]) 
                    push!(ue_exakt, case.exakte_Loesung(x_right)[m]) 
                end
            end

            ue_err = abs.(ue .- ue_exakt)
            uv_err = abs.(uv .- uv_exakt)
            
            max_err = max(maximum(ue_err), maximum(uv_err))
            println("=> Maximum absolute Vertex-Error: ", round(max_err, sigdigits=4))
        else
            @warn "No exact solution available."
        end

        if do_plots
            ue_cg_mapped = Float64[]
            for i in 1:2:length(ue)
                push!(ue_cg_mapped, 0.5 * (ue[i] + ue[i+1]))
            end

            plt = plot_func(case, ue_cg_mapped, uv)
            title!(plt, "$title_str - Numerical Solution")
            if !isdir(joinpath("Figures", "run_star_KumarLeugering_dG"))
                mkpath(joinpath("Figures", "run_star_KumarLeugering_dG"))
            end
            savefig(plt, joinpath("Figures", "run_star_KumarLeugering_dG", "Kumar_Leugering_$(title_str)_solution.pdf"))

            if case.exakte_Loesung !== nothing
                ue_exakt_cg = [ue_exakt[i] for i in 1:2:length(ue_exakt)]
                
                plt_exakt = plot_func(case, ue_exakt_cg, uv_exakt)
                title!(plt_exakt, "$title_str - Exact Solution")
                savefig(plt_exakt, joinpath("Figures", "run_star_KumarLeugering_dG", "Kumar_Leugering_$(title_str)_exact_solution.pdf"))

                plt_diff = plot_func(case, abs.(ue_cg_mapped .- ue_exakt_cg), abs.(uv .- uv_exakt))
                title!(plt_diff, "$title_str - Difference (Numerical - Exact)")
                savefig(plt_diff, joinpath("Figures", "run_star_KumarLeugering_dG", "Kumar_Leugering_$(title_str)_difference.pdf"))
            end
        end
    end
end


"""
    plot_convergence_rate_dG_comparison(; eta::Float64=10.0, p::Int=1, do_plots::Bool=false)

Contrasts discontinuous Galerkin mapping boundaries between uniform meshes evaluating the H1-Norm and Shishkin meshes evaluating the D-Norm.

# Arguments
- `eta::Float64`: dG integration penalty parameter.
- `p::Int`: Basis expansion limit parameters.
- `do_plots::Bool`: Generates supplementary mapping graphs explicitly when true.

# Output
- `plt_conv`: Comparative render frame output.
"""
function plot_convergence_rate_dG_comparison(; eta=10.0, p=1, do_plots=false)
    N_values = 2:16
    
    errors_H1_uniform = Float64[]
    errors_D_shishkin = Float64[]
    
    println("Starting DG convergence study...")
    println(" -> Uniform mesh evaluated in H1-Norm (use_SUPG=false)")
    println(" -> Shishkin mesh evaluated in D-Norm (use_SUPG=true)\n")
    
    for N in N_values
        case_uni = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, N, layer_mesh=:uniform, ρ=1.0)
        # case_uni = testcase_star_MMS_smooth(1e-6, 1e-6, 1e-6, N, A3 = 1.0, layer_mesh=:uniform)
        
        G = SimpleGraph(case_uni.nv)
        for e in case_uni.edges; add_edge!(G, src(e), dst(e)); end
        
        Levels_uni = [Int(log2(case_uni.n_e[e])) for e in case_uni.edges]
        edge_x_uni = [case_uni.edge_x[e] for e in case_uni.edges]
        D_edge = [case_uni.eps_edge[e] for e in case_uni.edges]
        A_edge = [case_uni.a_edge[e] for e in case_uni.edges]
        
        f_exakt_wrapper_uni(x, m) = case_uni.f_edge[case_uni.edges[m]](x)

        HEE, HEV, HVE, HVV, f_E, f_V = createH_DG_AdvectionDiffusion(
            G, Levels_uni, edge_x_uni, D_edge, A_edge, f_exakt_wrapper_uni;
            eta=eta, p=p, use_SUPG=false
        )

        NE = length(f_E); NV = length(f_V)
        A_global = [HEE HEV; HVE HVV]; b_global = [f_E; f_V]

        for (node, val) in case_uni.dirichlet
            idx = NE + node; A_global[idx, :] .= 0.0; A_global[idx, idx] = 1.0; b_global[idx] = val
        end

        u_full_uni = A_global \ b_global
        ue_uni = u_full_uni[1:NE]; uv_uni = u_full_uni[NE+1:end]

        case_shi = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, N, layer_mesh=:shishkin, ρ=1.0)
        # case_shi = testcase_star_MMS_smooth(1e-6, 1e-6, 1e-6, N, A3 = 1.0, layer_mesh=:shishkin, shishkin_mode=:outflow)

        edge_x_shi = [case_shi.edge_x[e] for e in case_shi.edges]
        f_exakt_wrapper_shi(x, m) = case_shi.f_edge[case_shi.edges[m]](x)

        HEE, HEV, HVE, HVV, f_E, f_V = createH_DG_AdvectionDiffusion(
            G, Levels_uni, edge_x_shi, D_edge, A_edge, f_exakt_wrapper_shi;
            eta=eta, p=p, use_SUPG=false
        )

        A_global = [HEE HEV; HVE HVV]; b_global = [f_E; f_V]

        for (node, val) in case_shi.dirichlet
            idx = NE + node; A_global[idx, :] .= 0.0; A_global[idx, idx] = 1.0; b_global[idx] = val
        end

        u_full_shi = A_global \ b_global
        ue_shi = u_full_shi[1:NE]; uv_shi = u_full_shi[NE+1:end]

        err_uni, _ = calculate_case_dG_energy_error(ue_uni, uv_uni, case_uni)
        push!(errors_H1_uniform, err_uni)
        
        err_shi, _ = calculate_case_dG_energy_error(ue_shi, uv_shi, case_shi)
        push!(errors_D_shishkin, err_shi)

        if do_plots
            ue_cg_mapped_uni = Float64[]
            for i in 1:2:length(ue_uni); push!(ue_cg_mapped_uni, 0.5 * (ue_uni[i] + ue_uni[i+1])); end
            plt_uni = plot_case_3d(case_uni, ue_cg_mapped_uni, uv_uni)
            title!(plt_uni, "Uniform DG Solution, N=$N")
            if !isdir(joinpath("Figures", "plot_convergence_rate_dG_comparison"))
                mkpath(joinpath("Figures", "plot_convergence_rate_dG_comparison"))
            end
            savefig(plt_uni, joinpath("Figures", "plot_convergence_rate_dG_comparison", "Uniform_DG_Solution_N$N.pdf"))

            ue_cg_mapped_shi = Float64[]
            for i in 1:2:length(ue_shi); push!(ue_cg_mapped_shi, 0.5 * (ue_shi[i] + ue_shi[i+1])); end
            plt_shi = plot_shishkin_3d(case_shi, ue_cg_mapped_shi, uv_shi)
            title!(plt_shi, "Shishkin DG Solution, N=$N")
            savefig(plt_shi, joinpath("Figures", "plot_convergence_rate_dG_comparison", "Shishkin_DG_Solution_N$N.pdf"))
        end
    end

    h_values = 1.0 ./ (2.0 .^ N_values)
    rates_uni = zeros(length(h_values)); rates_shi = zeros(length(h_values))
    rates_uni[1] = NaN; rates_shi[1] = NaN
    
    for i in 2:length(h_values)
        rates_uni[i] = log(errors_H1_uniform[i-1] / errors_H1_uniform[i]) / log(h_values[i-1] / h_values[i])
        rates_shi[i] = log(errors_D_shishkin[i-1] / errors_D_shishkin[i]) / log(h_values[i-1] / h_values[i])
    end

    println("\nDG Convergence Rates Comparison:")
    println("-" ^ 95)
    @printf("%-15s | %-10s | %-18s | %-10s | %-18s | %-10s\n", 
            "N (Elements)", "h (approx)", "H1-Error (Uniform)", "Rate", "D-Error (Shishkin)", "Rate")
    println("-" ^ 95)
    
    for i in 1:length(N_values)
        val_N     = string(2^N_values[i])
        val_h     = @sprintf("%.2e", h_values[i])
        val_err_u = @sprintf("%.4e", errors_H1_uniform[i])
        val_rat_u = i == 1 ? "-" : @sprintf("%.4f", rates_uni[i])
        val_err_s = @sprintf("%.4e", errors_D_shishkin[i])
        val_rat_s = i == 1 ? "-" : @sprintf("%.4f", rates_shi[i])
        
        @printf("%-15s | %-10s | %-18s | %-10s | %-18s | %-10s\n", 
                val_N, val_h, val_err_u, val_rat_u, val_err_s, val_rat_s)
    end
    println("-" ^ 95)

    plt_conv = nothing

    if do_plots
        anchor_err = errors_D_shishkin[1]
        ref_O1  = h_values .* (anchor_err / h_values[1])               
        ref_O2  = (h_values.^2) .* (anchor_err / h_values[1]^2)        
        ref_O32 = (h_values .^(3/2)) .* (1.5 * anchor_err / h_values[1]^(3/2)) 
        ref_O1log = (h_values .* log.(1.0 ./ h_values)) .* (anchor_err / (h_values[1] * log(1.0 / h_values[1])))

        plt_conv = plot(h_values, errors_H1_uniform, 
                        m=:circle, lw=2,
                        label="Uniform DG (H1-Norm)", 
                        xaxis=:log10, yaxis=:log10,
                        xlabel="Mesh size h (approx. 1/N)", 
                        ylabel="Absolute Error",
                        title="Convergence: Uniform (H1) vs. Shishkin (D-Norm)",
                        legend=:bottomright,
                        dpi=600)
                        
        plot!(plt_conv, h_values, errors_D_shishkin, 
            m=:square, lw=2, ls=:solid,
            label="Shishkin DG (D-Norm)")
                        
        plot!(plt_conv, h_values, ref_O1, lw=2, ls=:dash, color=:gray, label="O(h) Reference")
        plot!(plt_conv, h_values, ref_O2, lw=2, ls=:dot, color=:gray, label="O(h^2) Reference")
        plot!(plt_conv, h_values, ref_O32, lw=2, ls=:dashdot, color=:gray, label="O(h^(3/2)) Reference", m=:utriangle)
        plot!(plt_conv, h_values, ref_O1log, lw=2, ls=:dashdotdot, color=:gray, label="O(h log(1/h)) Reference")

        if !isdir(joinpath("Figures", "plot_convergence_rate_dG_comparison"))
            mkpath(joinpath("Figures", "plot_convergence_rate_dG_comparison"))
        end
        savefig(plt_conv, joinpath("Figures", "plot_convergence_rate_dG_comparison", "DG_Shishkin_DNorm_Convergence.pdf"))
    end

    return plt_conv, h_values, errors_H1_uniform, errors_D_shishkin
end


"""
    calculate_case_dG_energy_error(ue::Vector{Float64}, uv::Vector{Float64}, case::NamedTuple; eta::Float64=1.0)

Calculates the absolute and relative errors strictly within the discontinuous Galerkin (dG) energy norm bounds.

# Arguments
- `ue::Vector{Float64}`: Numerical solution vector targeting internal edge element components.
- `uv::Vector{Float64}`: Numerical solution vector targeting physical node vertices.
- `case::NamedTuple`: The predefined structural test case payload specifying derivative conditions and limits.
- `eta::Float64`: The discontinuous interface penalty tuning parameter (default: 1.0).

# Output
- `D_err::Float64`: Computed absolute error evaluated within the dG energy norm parameterization.
- `rel_D_err::Float64`: Computed relative error derived against standard system magnitudes.
"""
function calculate_case_dG_energy_error(ue, uv, case; eta::Float64=1.0)
    if isnothing(case.exakte_Loesung)
        @warn "No exact solution defined in case!"
        return NaN, NaN
    end
    
    if !hasfield(typeof(case), :exact_derivative) || isnothing(case.exact_derivative)
        @warn "No 'exact_derivative' function to compute the dG norm!"
        return NaN, NaN
    end

    err2_global = 0.0
    nrm2_global = 0.0
    edge_offsets = zeros(Int, length(case.edges))
    curr = 0
    for m in 1:length(case.edges)
        edge_offsets[m] = curr
        curr += 2 * case.n_e[case.edges[m]]
    end

    gauss_nodes   = [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664]
    gauss_weights = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]

    for (m, e) in enumerate(case.edges)
        x_points = case.edge_x[e]
        N_e = case.n_e[e]
        D_e = case.eps_edge[e] 
        offset = edge_offsets[m]

        for k in 1:N_e
            va = ue[offset + 2*k - 1]
            vb = ue[offset + 2*k]

            xa = x_points[k]
            xb = x_points[k+1]
            h_local = xb - xa 
            
            duh_val = (vb - va) / h_local
            
            half_h = h_local / 2.0
            mid_x  = (xa + xb) / 2.0

            e2_H1_local = 0.0
            n2_H1_local = 0.0

            for q in 1:5
                xi = gauss_nodes[q]
                wq = gauss_weights[q]
                
                s = half_h * xi + mid_x
                
                du_ex_val = case.exact_derivative(s)[m] 
                
                e2_H1_local += wq * (duh_val - du_ex_val)^2
                n2_H1_local += wq * (du_ex_val)^2
            end
            
            err2_global += half_h * (D_e * e2_H1_local)
            nrm2_global += half_h * (D_e * n2_H1_local)
        end

        for k in 1:N_e-1
            vb_left  = ue[offset + 2*k]           
            va_right = ue[offset + 2*(k+1) - 1]   
            
            jump_sq = (vb_left - va_right)^2
            h_inter = x_points[k+1] - x_points[k] 
            
            err2_global += (eta * D_e / h_inter) * jump_sq
        end

        v_start = src(e)
        v_end   = dst(e)
        
        va_first = ue[offset + 1]
        u_hat_start = uv[v_start]
        h_first = x_points[2] - x_points[1]
        err2_global += (eta * D_e / h_first) * (va_first - u_hat_start)^2
        
        vb_last = ue[offset + 2*N_e]
        u_hat_end = uv[v_end]
        h_last = x_points[end] - x_points[end-1]
        err2_global += (eta * D_e / h_last) * (vb_last - u_hat_end)^2
    end

    D_err = sqrt(err2_global)
    rel_D_err = D_err / max(sqrt(nrm2_global), 1e-14)
    
    return D_err, rel_D_err
end


"""
    plot_convergence_rate_pure_dG_comparison(; eta::Float64=10.0, p::Int=1, do_plots::Bool=false)

Computes the convergence boundaries mapped natively out of the uniform mesh strategy and strictly verifies stability scaling in the pure DG format. Outputs plots automatically if queried.

# Arguments
- `eta::Float64`: Interface penalty scalar component.
- `p::Int`: Basis limiting evaluation degree.
- `do_plots::Bool`: Flag tracking if analytical renders apply immediately post-processing.

# Output
- Returns comprehensive structural chart renders.
"""
function plot_convergence_rate_pure_dG_comparison(; eta=10.0, p=1, do_plots=false)
    N_values = 2:16
    
    errors_H1_uniform = Float64[]
    
    for N in N_values
        # case_uni = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, N, layer_mesh=:uniform, ρ=1.0)
        case_uni = testcase_star_MMS_smooth(1e-6, 1e-6, 1e-6, N, A3 = 1.0, layer_mesh=:uniform)
        
        G = SimpleGraph(case_uni.nv)
        for e in case_uni.edges; add_edge!(G, src(e), dst(e)); end
        
        Levels_uni = [Int(log2(case_uni.n_e[e])) for e in case_uni.edges]
        edge_x_uni = [case_uni.edge_x[e] for e in case_uni.edges]
        D_edge = [case_uni.eps_edge[e] for e in case_uni.edges]
        A_edge = [case_uni.a_edge[e] for e in case_uni.edges]
        
        f_exakt_wrapper_uni(x, m) = case_uni.f_edge[case_uni.edges[m]](x)

        HEE, HEV, HVE, HVV, f_E, f_V = createH_DG_AdvectionDiffusion(
            G, Levels_uni, edge_x_uni, D_edge, A_edge, f_exakt_wrapper_uni;
            eta=eta, p=p, use_SUPG=false
        )

        NE = length(f_E); NV = length(f_V)
        A_global = [HEE HEV; HVE HVV]; b_global = [f_E; f_V]

        for (node, val) in case_uni.dirichlet
            idx = NE + node; A_global[idx, :] .= 0.0; A_global[idx, idx] = 1.0; b_global[idx] = val
        end

        u_full_uni = A_global \ b_global
        ue_uni = u_full_uni[1:NE]; uv_uni = u_full_uni[NE+1:end]
        err_uni, _ = calculate_case_dG_energy_error(ue_uni, uv_uni, case_uni)
        push!(errors_H1_uniform, err_uni)

        if do_plots
            ue_cg_mapped_uni = Float64[]
            curr_idx = 1
            for e in case_uni.edges
                N_e = case_uni.n_e[e]
                for k in 1:(N_e - 1)
                    val_L = ue_uni[curr_idx + 2*k - 1]
                    val_R = ue_uni[curr_idx + 2*k]
                    push!(ue_cg_mapped_uni, 0.5 * (val_L + val_R))
                end
                curr_idx += 2 * N_e
            end
            plt_uni = plot_case_3d(case_uni, ue_cg_mapped_uni, uv_uni)
            title!(plt_uni, "Uniform Pure DG Solution, N=$N")
            if !isdir(joinpath("Figures", "plot_convergence_rate_pure_dG_comparison"))
                mkpath(joinpath("Figures", "plot_convergence_rate_pure_dG_comparison"))
            end
            savefig(plt_uni, joinpath("Figures", "plot_convergence_rate_pure_dG_comparison", "PureDG_Uniform_Solution_N$N.pdf"))
        end
    end

    h_values = 1.0 ./ (2.0 .^ N_values)
    rates_uni = zeros(length(h_values))
    rates_uni[1] = NaN
    
    for i in 2:length(h_values)
        rates_uni[i] = log(errors_H1_uniform[i-1] / errors_H1_uniform[i]) / log(h_values[i-1] / h_values[i])
    end

    println("\nPure DG Convergence Rates Comparison:")
    println("-" ^ 95)
    @printf("%-15s | %-10s | %-18s | %-10s\n", 
            "N (Elements)", "h (approx)", "H1-Error (Uniform)", "Rate")
    println("-" ^ 95)
    
    for i in 1:length(N_values)
        val_N     = string(2^N_values[i])
        val_h     = @sprintf("%.2e", h_values[i])
        val_err_u = @sprintf("%.4e", errors_H1_uniform[i])
        val_rat_u = i == 1 ? "-" : @sprintf("%.4f", rates_uni[i])
        
        @printf("%-15s | %-10s | %-18s | %-10s\n", 
                val_N, val_h, val_err_u, val_rat_u)
    end
    println("-" ^ 95)

    plt_conv = nothing

    if do_plots
        anchor_err = errors_H1_uniform[1]
        ref_O1  = h_values .* (anchor_err / h_values[1])               
        ref_O2  = (h_values.^2) .* (anchor_err / h_values[1]^2)        
        ref_O32 = (h_values .^(3/2)) .* (1.5 * anchor_err / h_values[1]^(3/2)) 
        ref_O1log = (h_values .* log.(1.0 ./ h_values)) .* (anchor_err / (h_values[1] * log(1.0 / h_values[1])))

        plt_conv = plot(h_values, errors_H1_uniform, 
                        m=:circle, lw=2,
                        label="DG-norm", 
                        xaxis=:log10, yaxis=:log10,
                        xlabel="Mesh size h (approx. 1/N)", 
                        ylabel="Absolute Error",
                        title="Convergence: dG energy norm",
                        legend=:bottomright,
                        dpi=600)
                        
        plot!(plt_conv, h_values, ref_O1, lw=2, ls=:dash, color=:gray, label="O(h) Reference")
        plot!(plt_conv, h_values, ref_O2, lw=2, ls=:dot, color=:gray, label="O(h^2) Reference")
        plot!(plt_conv, h_values, ref_O32, lw=2, ls=:dashdot, color=:gray, label="O(h^(3/2)) Reference", m=:utriangle)
        plot!(plt_conv, h_values, ref_O1log, lw=2, ls=:dashdotdot, color=:gray, label="O(h log(1/h)) Reference")
        if !isdir(joinpath("Figures", "plot_convergence_rate_pure_dG_comparison"))
            mkpath(joinpath("Figures", "plot_convergence_rate_pure_dG_comparison"))
        end
        savefig(plt_conv, joinpath("Figures", "plot_convergence_rate_pure_dG_comparison", "PureDG_Uniform_H1_Convergence.pdf"))
    end

    return plt_conv, h_values, errors_H1_uniform
end


"""
    run_star_KumarLeugering_pure_dG(; eta::Float64=10.0, p::Int=1, do_plots=false)

Test execution wrapper assessing explicit discontinuity integration directly evaluating independent boundary 
components modeled purely utilizing non-coupled edge elements.

# Arguments
- `eta::Float64`: Interface penalty variable component.
- `p::Int`: Formulation configuration degree scalar.
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
None. Creates analytical visualizations mapped locally into designated output drives.
"""
function run_star_KumarLeugering_pure_dG(; eta=10.0, p=1, do_plots=false)

    configs = [
        (:uniform, plot_case_3d, "dG"),
        (:shishkin, plot_shishkin_3d, "Shishkin Pure-DG")
    ]

    for (mesh_type, plot_func, title_str) in configs
        case = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, 8, layer_mesh=mesh_type, ρ=1.0)

        G = SimpleGraph(case.nv)
        for e in case.edges
            add_edge!(G, src(e), dst(e))
        end

        Levels = [Int(log2(case.n_e[e])) for e in case.edges]
        D_edge = [case.eps_edge[e] for e in case.edges]
        A_edge = [case.a_edge[e] for e in case.edges]
        edge_x = [case.edge_x[e] for e in case.edges]
        
        f_exakt_wrapper(x, m) = case.f_edge[case.edges[m]](x)

        HEE, HEV, HVE, HVV, f_E, f_V = createH_DG_AdvectionDiffusion(
            G, Levels, edge_x, D_edge, A_edge, f_exakt_wrapper;
            eta=eta, p=p, use_SUPG=false
        )

        NE = length(f_E)
        NV = length(f_V)

        A_global = [HEE HEV; 
                    HVE HVV]
        b_global = [f_E; f_V]

        for (node, val) in case.dirichlet
            idx = NE + node 
            A_global[idx, :] .= 0.0
            A_global[idx, idx] = 1.0
            b_global[idx] = val
        end

        u_full = A_global \ b_global

        ue = u_full[1:NE]
        uv = u_full[NE+1:end]

        if case.exakte_Loesung !== nothing
            uv_exakt = zeros(case.nv)
            ue_exakt = Float64[]

            for (m, e) in enumerate(case.edges)
                x_nodes = case.edge_x[e]
                L_edge = x_nodes[end]

                uv_exakt[src(e)] = case.exakte_Loesung(0.0)[m]
                uv_exakt[dst(e)] = case.exakte_Loesung(L_edge)[m]
                
                N_e = case.n_e[e]
                
                for k in 1:N_e
                    u_val_L = case.exakte_Loesung(x_nodes[k])[m]
                    u_val_R = case.exakte_Loesung(x_nodes[k+1])[m]
                    
                    push!(ue_exakt, u_val_L) 
                    push!(ue_exakt, u_val_R) 
                end
            end

            ue_err = abs.(ue .- ue_exakt)
            uv_err = abs.(uv .- uv_exakt)

            max_err = max(maximum(ue_err), maximum(uv_err))
            println("Maximum absolute vertex error: ", round(max_err, sigdigits=4))
        else
            @warn "No exact solution available for this case."
        end

        if do_plots
            ue_cg_mapped = Float64[]
            curr_idx = 1
            for e in case.edges
                N_e = case.n_e[e]
                for k in 1:(N_e - 1)
                    val_L = ue[curr_idx + 2*k - 1]
                    val_R = ue[curr_idx + 2*k]
                    push!(ue_cg_mapped, 0.5 * (val_L + val_R))
                end
                curr_idx += 2 * N_e
            end
            plt = plot_func(case, ue_cg_mapped, uv)
            title!(plt, "$title_str - Numerical Solution")
            if !isdir(joinpath("Figures", "run_star_KumarLeugering_pure_dG"))
                mkpath(joinpath("Figures", "run_star_KumarLeugering_pure_dG"))
            end
            savefig(plt, joinpath("Figures", "run_star_KumarLeugering_pure_dG", "kumarleugering_PureDG_uniform.pdf"))

            if case.exakte_Loesung !== nothing
                ue_exakt_cg = Float64[]
                curr_idx_ex = 1
                for e in case.edges
                    N_e = case.n_e[e]
                    for k in 1:(N_e - 1)
                        val_L = ue_exakt[curr_idx_ex + 2*k - 1]
                        val_R = ue_exakt[curr_idx_ex + 2*k]
                        push!(ue_exakt_cg, 0.5 * (val_L + val_R))
                    end
                    curr_idx_ex += 2 * N_e
                end
                
                plt_exakt = plot_func(case, ue_exakt_cg, uv_exakt)
                title!(plt_exakt, "$title_str - Exact Solution")

                plt_diff = plot_func(case, abs.(ue_cg_mapped .- ue_exakt_cg), abs.(uv .- uv_exakt))
                title!(plt_diff, "$title_str - Difference (Num - Exact)")
                savefig(plt_diff, joinpath("Figures", "run_star_KumarLeugering_pure_dG", "kumarleugering_Difference_PureDG.pdf"))
            end
        end
    end
end


"""
    run_MG_cycle(; do_plots=false)

Solves an advection-diffusion cyclic graph geometry implementation with integrated SUPG parameters, ensuring continuous numerical loop validation algorithms function appropriately alongside visual confirmation tests.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).
"""
function run_MG_cycle(; do_plots=false)
    my_case = testcase_cycle_advection(0.1, 8.0, 10)

    ue, uv, residuen, _, _ = run_MG_from_case(my_case; v1=3, v2=3, autofix=false, use_SUPG = true)

    check_vertex_flux_condition_MG(ue, uv, my_case)
    
    if do_plots
        plt = plot_case_3d(my_case, ue, uv)
        title!(plt, "Numerical Solution")
        if !isdir(joinpath("Figures", "run_MG_cycle"))
            mkpath(joinpath("Figures", "run_MG_cycle"))
        end
        savefig(plt, joinpath("Figures", "run_MG_cycle", "cyclegraph_solution_diff_adv_3d.pdf"))
    end

    if my_case.exakte_Loesung !== nothing
        uv_exakt = zeros(my_case.nv)
        ue_exakt = Float64[]

        for (i, e) in enumerate(my_case.edges)
            x_nodes = my_case.edge_x[e]
            
            u_vals = [my_case.exakte_Loesung(x)[i] for x in x_nodes]
            uv_exakt[src(e)] = u_vals[1]
            uv_exakt[dst(e)] = u_vals[end]
            
            append!(ue_exakt, u_vals[2:end-1])
        end

        N_ue_orig = length(ue_exakt)
        N_uv_orig = length(uv_exakt)

        ue_err = abs.(ue[1:N_ue_orig] .- ue_exakt)
        uv_err = abs.(uv[1:N_uv_orig] .- uv_exakt)
        
        max_err = max(maximum(ue_err), maximum(uv_err))
        println("Maximum absolute error: ", round(max_err, sigdigits=4))

        if do_plots
            plt_exakt = plot_case_3d(my_case, ue_exakt, uv_exakt)
            title!(plt_exakt, "Exact solution")
            savefig(plt_exakt, joinpath("Figures", "run_MG_cycle", "cyclegraph_exact_solution_diff_adv_3d.pdf"))
            
            plt_err = plot_case_3d(my_case, ue_err, uv_err)
            title!(plt_err, "Absolute Error |u_h - u_exakt|")
            savefig(plt_err, joinpath("Figures", "run_MG_cycle", "cyclegraph_absolute_error_diff_adv_3d.pdf"))
        end
    else
        @warn "No exact solution available."
    end

    if !isnothing(my_case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, my_case)
        println("Result L2-Error:")
        println("Absolute: $L2_abs")
        println("Relative: $L2_rel")
    end

    if do_plots
        pltnumvex = plot_case_num_vs_exact(my_case, ue, uv, "Numerical vs. Exact")
        savefig(pltnumvex, joinpath("Figures", "run_MG_cycle", "cyclegraph_numerical_vs_exact_diff_adv.pdf"))

        plt_local_error = plot_case_edge_difference(my_case, ue, uv, "Local Error Distribution")
        savefig(plt_local_error, joinpath("Figures", "run_MG_cycle", "cyclegraph_local_error_diff_adv.pdf"))
    end
end


"""
    MG_advection_varying_parameters(; do_plots=false)

Iterates a thorough parameter sweep across a prebuilt advection-diffusion Barabási-Albert geometry.
Modifies inner logic testing various internal iteration counts (nu_1, nu_2) verifying convergence limits.

# Arguments
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).
"""
function MG_advection_varying_parameters(; do_plots=false)
    my_barabasi_case = testcase_barabasi(2000, 5.0, 0.1, 6; 
                                         dirichlet_nodes=[1, 20], 
                                         dirichlet_values=[0.0, 1.0])
                                         
    M_orig = length(my_barabasi_case.edges)
    NV_orig = my_barabasi_case.nv
    
    Levels = zeros(Int, M_orig)
    edge_length = zeros(Float64, M_orig)
    D_vec = zeros(Float64, M_orig)
    A_vec = zeros(Float64, M_orig)

    for (i, e) in enumerate(my_barabasi_case.edges)
        Levels[i] = Int(log2(my_barabasi_case.n_e[e])) 
        edge_length[i] = my_barabasi_case.edge_x[e][end] - my_barabasi_case.edge_x[e][1]
        D_vec[i] = my_barabasi_case.eps_edge[e]
        A_vec[i] = my_barabasi_case.a_edge[e]
    end

    G = SimpleDiGraph(NV_orig)
    for e in my_barabasi_case.edges
        add_edge!(G, e)
    end

    d_nodes = collect(keys(my_barabasi_case.dirichlet))
    d_vals = collect(values(my_barabasi_case.dirichlet))

    # Auto-Fix
    problem_nodes = check_inflow_vertices(G, A_vec)
    if !isempty(problem_nodes)
        fix_inflow_nodes!(G, problem_nodes, Levels, edge_length, D_vec, A_vec, 
                          d_nodes, d_vals; 
                          L_new=1.0, D_new=0.01, A_new=5.0, 
                          Level_new=3, val_new=0.0)
    end

    @info "Final number of vertices and edges after Auto-Fix: $(nv(G)), $(ne(G))"
    function f_exakt_wrapper(x, i)
        if i <= M_orig
            return my_barabasi_case.f_edge[my_barabasi_case.edges[i]](x)
        else
            return 0.0 
        end
    end

    HEE, HEV, HVE, HVV, fe, fv = createH_AdvectionDiffusion(G, Levels, edge_length, 
                                                            D_vec, A_vec, f_exakt_wrapper; 
                                                            use_SUPG=true)

    HEE, HEV, HVE, HVV, fe, fv = apply_dirichlet_blocks!(HEE, HEV, HVE, HVV, fe, fv, d_nodes, d_vals)
    
    J_max = maximum(Levels)
    macroscopic_order = get_macroscopic_sort(G, A_vec)

    mg_hierarchy = setup_advection_hierarchy(HEE, HEV, HVE, HVV, Levels, J_max, G, A_vec, macroscopic_order)

    smooth_vec_V = [[(1, 1), (3, 1), (5, 1), (7, 1)],
                    [(1, 3), (3, 3), (5, 3), (7, 3)],
                    [(5, 7), (7, 5), (7, 7)]]

    smooth_vec_W = [[(1, 1), (4, 1), (3, 3)]]

    for mu in [1, 2]
        mu_s = mu == 1 ? "V" : "W"
        smooth_vec = mu == 1 ? smooth_vec_V : smooth_vec_W
        lim_y = 0.1

        for (savez, set) in enumerate(smooth_vec)
            plotvec = []
            if set == [(5, 7), (7, 5), (7, 7)]
                lim_y = 0.05
            end

            for (zaehl, (v1, v2)) in enumerate(set)
                @info "Testing parameters: v1=$v1, v2=$v2, cycle=$mu_s"
                
                ue = ones(size(fe))
                uv = ones(size(fv))
                for (n, val) in zip(d_nodes, d_vals)
                    uv[n] = val
                end

                rate_e_vec = Float64[]
                rate_v_vec = Float64[]
                rate_vec = Float64[]
                kmax = 0

                while ((kmax < 250) && (sqrt(norm(fe - HEE * ue - HEV * uv)^2 + norm(fv - HVE * ue - HVV * uv)^2)) > 1e-8)

                    solve_advection_MG!(mg_hierarchy, J_max, fv, fe, uv, ue, v1, v2, mu)

                    res_e = norm(fe - HEE * ue - HEV * uv)
                    res_v = norm(fv - HVE * ue - HVV * uv)
                    res_total = sqrt(res_e^2 + res_v^2)
                    
                    push!(rate_e_vec, res_e)
                    push!(rate_v_vec, res_v)
                    push!(rate_vec, res_total)

                    kmax += 1
                end

                if do_plots
                    ticksvec = kmax - 1 > 9 ? (2:2:kmax) : (2:1:kmax)
                    plot_data = [rate_vec[2:end] ./ rate_vec[1:end-1], 
                                 rate_e_vec[2:end] ./ rate_e_vec[1:end-1],
                                 rate_v_vec[2:end] ./ rate_v_vec[1:end-1]]
                                 
                    title_str = L"$\nu_1$: %$v1; $\nu_2$: %$v2 - %$mu_s-cycles: %$kmax"

                    if zaehl == length(set)
                        Plot_Res = plot(2:1:kmax, plot_data,
                            label=["total rate" "rate internal nodes" "rate vertices"],
                            title=title_str, legend=:topleft, lw=5, tickfontsize=14, 
                            ylims=(0, lim_y), titlefontsize=20, legendfontsize=16, 
                            xticks=ticksvec, linestyle=[:solid :dash :dash], dpi=1200)
                    else
                        Plot_Res = plot(2:1:kmax, plot_data,
                            title=title_str, legend=false, lw=5, tickfontsize=14, 
                            ylims=(0, lim_y), titlefontsize=20, xticks=ticksvec, 
                            linestyle=[:solid :dash :dash], dpi=1200)
                    end
                    push!(plotvec, Plot_Res)
                end
            end
            
            if do_plots
                Plot_ges = plot(plotvec..., size=(450 * length(set), 650),
                    layout=(1, length(set)), margin=5Plots.mm, dpi=1200)
                if !isdir(joinpath("Figures", "MG_advection_varying_parameters"))
                    mkpath(joinpath("Figures", "MG_advection_varying_parameters"))
                end
                savefig(Plot_ges, joinpath("Figures", "MG_advection_varying_parameters", "advection_comparison_plots_$(mu)_$(savez).pdf"))
            end
        end
    end
end


"""
    MG_advection_Barabasi_extended(v1::Int=3, v2::Int=3; do_plots=false)

Generates iterative convergence benchmarks evaluating standard parameters alongside varying spatial properties 
on standard generated complex structures evaluating computation efficiency metrics internally. 

# Arguments
- `v1::Int`: Pre-smoothing iteration constraint limits (default: 3).
- `v2::Int`: Post-smoothing iteration constraint limits (default: 3).
- `do_plots::Bool`: Flag to enable or disable plotting (default: false).

# Output
- `table_data::Matrix`: Data table storing generated cycle logic metrics structurally evaluated in sequential configurations.
"""
function MG_advection_Barabasi_extended(v1::Int=3, v2::Int=3; do_plots=false)
    Random.seed!(1)
    
    base_nodes = 2000
    base_J = 8
    
    base_case = testcase_barabasi(base_nodes, 5.0, 0.01, base_J; 
                                  dirichlet_nodes=[1, base_nodes], 
                                  dirichlet_values=[0.0, 1.0])

    _, _, _, _, _ = run_MG_from_case(base_case; v1=v1, v2=v2, 
                               A_fix=5.0, D_fix=0.01, L_fix=1.0, use_SUPG=true)

    configs = [
        (2000, 8),
        (2000, 10),
        (4000, 6),
        (4000, 8),
        (6000, 8)
    ]
    
    nodes_arr  = Int[]
    edges_arr  = Int[]
    j_arr      = Int[]
    dof_arr    = Int[]
    cycles_arr = Int[]
    time_arr   = Float64[]
    
    for (N_nodes, J_val) in configs
        @info "Solving for Nodes = $N_nodes, J = $J_val..."
        
        current_case = testcase_barabasi(N_nodes, 5.0, 0.01, J_val; 
                                         dirichlet_nodes=[1, N_nodes], 
                                         dirichlet_values=[0.0, 1.0])
        
        V_nodes = current_case.nv
        M_edges = length(current_case.edges)
        DoF = M_edges * (2^J_val - 1) + V_nodes 

        comp_time = @elapsed begin
            _, _, curr_rates, V_nodes, M_edges = run_MG_from_case(current_case; v1=v1, v2=v2, 
                                                A_fix=5.0, D_fix=0.01, L_fix=1.0, use_SUPG=true)
        end
        
        push!(nodes_arr, V_nodes)
        push!(edges_arr, M_edges)
        push!(j_arr, J_val)
        push!(dof_arr, M_edges * (2^J_val - 1) + V_nodes )
        push!(cycles_arr, length(curr_rates)) 
        push!(time_arr, comp_time)
    end

    header = ["Nodes", "Edges", "Level J", "DoF", "MG-Cycles", "Time [s]"]
    
    table_data = hcat(nodes_arr, edges_arr, j_arr, dof_arr, cycles_arr, round.(time_arr, digits=3))
    
    pretty_table(
            table_data; 
            column_labels = header, 
            title = "Multigrid Performance on Barabasi-Albert Graphs (Advection-Diffusion)",
            fit_table_in_display_vertically=false,
            fit_table_in_display_horizontally=false
        )

    return table_data
end