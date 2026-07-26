"""
    run_star_shishkin(; do_plots=false)

Executes and visualizes a star-graph test case with an exact solution by Kumar & Leugering using a Shishkin layer mesh.

This function generates a specific test case, computes the numerical solution using a block backslash solver, and checks the vertex flux conditions. It proceeds to visualize the numerical solution, the exact analytical solution, and the absolute error between them. Finally, it calculates and prints the maximum absolute error and the L2 error to the console.
"""
function run_star_shishkin(; do_plots=false)
    case = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, 5,
                               layer_mesh=:shishkin, shishkin_mode=:outflow)

    ue, uv = run_star_shishkin_block_backslash(case)
    check_vertex_flux_condition_MG(ue, uv, case)

    if do_plots
        plt = plot_shishkin_3d(case, ue, uv)
        title!(plt, "Solution of Testcase from Kumar & Leugering")
        if !isdir(joinpath("Figures", "run_star_shishkin"))
            mkdir(joinpath("Figures", "run_star_shishkin"))
        end
        savefig(plt, joinpath("Figures", "run_star_shishkin", "solution_star_shishkin.pdf"))
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

        if do_plots
            plt_exakt = plot_shishkin_3d(case, ue_exakt, uv_exakt)
            title!(plt_exakt, "Exact analytical solution")
            if !isdir(joinpath("Figures", "run_star_shishkin"))
                mkdir(joinpath("Figures", "run_star_shishkin"))
            end
            savefig(plt_exakt, joinpath("Figures", "run_star_shishkin", "exact_solution_star_shishkin.pdf"))
        end

        N_ue_orig = length(ue_exakt)
        N_uv_orig = length(uv_exakt)

        ue_err = abs.(ue[1:N_ue_orig] .- ue_exakt)
        uv_err = abs.(uv[1:N_uv_orig] .- uv_exakt)
        
        if do_plots
            plt_err = plot_shishkin_3d(case, ue_err, uv_err)
            title!(plt_err, "Absolute error |u_h - u_exakt|")
            savefig(plt_err, joinpath("Figures", "run_star_shishkin", "error_star_shishkin.pdf"))
        end
        
        max_err = max(maximum(ue_err), maximum(uv_err))
        println("Maximum absolute error of the discretization: ", round(max_err, sigdigits=4))
    else
        @warn "No exact solution is available for this case."
    end

    if !isnothing(case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, case)
        println("L2 error results:")
        println("Absolute: $L2_abs")
        println("Relative: $L2_rel")
    end
end


"""
    run_star_bakhvalov(; do_plots=false)

Executes and visualizes a star-graph test case with an exact solution by Kumar & Leugering using a Bakhvalov layer mesh.

Similar to `run_star_shishkin`, this function generates the test case but relies on a Bakhvalov mesh (`layer_mesh=:bakhvalov`) with a stretching parameter `ρ=1.2`. It solves the problem, checks vertex conditions, visualizes the numerical/exact solutions along with the absolute error, and prints the error metrics.
"""
function run_star_bakhvalov(; do_plots=false)
    case = testcase_star_exact_Kumar_Leugering(0.003, 0.01, 0.007, 5,
                               layer_mesh=:bakhvalov, shishkin_mode=:outflow, ρ=1.2)
    
    ue, uv = run_star_shishkin_block_backslash(case; shishkin_mode=:outflow)
    check_vertex_flux_condition_MG(ue, uv, case)

    if do_plots
        plt = plot_shishkin_3d(case, ue, uv)
        title!(plt, "Solution of Testcase from Kumar & Leugering")
        if !isdir(joinpath("Figures", "run_star_bakhvalov"))
            mkdir(joinpath("Figures", "run_star_bakhvalov"))
        end
        savefig(plt, joinpath("Figures", "run_star_bakhvalov", "solution_star_bakhvalov.pdf"))
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

        if do_plots
            plt_exakt = plot_shishkin_3d(case, ue_exakt, uv_exakt)
            title!(plt_exakt, "Exact analytical solution")
            if !isdir(joinpath("Figures", "run_star_bakhvalov"))
                mkdir(joinpath("Figures", "run_star_bakhvalov"))
            end
            savefig(plt_exakt, joinpath("Figures", "run_star_bakhvalov", "exact_solution_star_bakhvalov.pdf"))
        end

        N_ue_orig = length(ue_exakt)
        N_uv_orig = length(uv_exakt)

        ue_err = abs.(ue[1:N_ue_orig] .- ue_exakt)
        uv_err = abs.(uv[1:N_uv_orig] .- uv_exakt)
        
        if do_plots
            plt_err = plot_shishkin_3d(case, ue_err, uv_err)
            title!(plt_err, "Absolute error |u_h - u_exakt|")
            savefig(plt_err, joinpath("Figures", "run_star_bakhvalov", "error_star_bakhvalov.pdf"))
        end
        
        max_err = max(maximum(ue_err), maximum(uv_err))
        println("=> Maximum absolute error of the discretization: ", round(max_err, sigdigits=4))
    else
        @warn "No exact solution is available for this case."
    end

    if !isnothing(case.exakte_Loesung)
        L2_abs, L2_rel = calculate_case_l2_error(ue, uv, case)
        println("L2 error results:")
        println("Absolute: $L2_abs")
        println("Relative: $L2_rel")
    end
end