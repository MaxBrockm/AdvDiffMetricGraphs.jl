module AdvDiffMetricGraphs

    using Graphs
    using LinearAlgebra
    using SparseArrays
    using BSplines
    using QuadGK
    using Plots
    using Printf
    using Random
    using PrettyTables
    using LaTeXStrings
    using ILUZero
    using DocStringExtensions

    export  # AMG_Helper
            AMGLevel,

            DRA_QC_CE,
            DRA_QC_CE_detailed,
            DRA_QC,
            _dra_qc_ce_internal,
            _compute_gamma,
            _dra_tentative_adj,
            _ext_weight,
            _int_weight,
            _build_AG_XG,
            _criterion_16,
            _bad_vertices_removal,
            _cholesky_quality_test,
            _subgroup_extraction,
            _qc_extract,
            DRA_mix_alt,
            DRA_mix,
            _solve_singular_laplacian,
            _is_singular_laplacian,
            prolongation,
            _safe_lu,
            build_amg_hierarchy,
            _solve_coarsest!,
            mg_preconditioner!,
            fcg_solve,
            amg_solve,

            # Multigrid
            MGLevel,

            setup_MG_hierarchy,
            Rest_int_nodes,
            Rest_HEV_setup,
            Rest_HEV,
            Multigrid_Graph,
            Multigrid_Graph_solve!,
            smoother_Jac,
            smoother_Jac!,

            # Multigrid_DiffAdv
            AdvectionMGLevel,
            setup_advection_hierarchy,
            Multigrid_Graph,
            solve_advection_MG!,
            Prol_int_nodes_MatrixDep,
            Prol_HEV_MatrixDep,
            Rest_int_nodes_MatrixDep,
            Rest_HVE_MatrixDep,

            #createRHS
            righthandside,

            # GraphFunktionplotten
            plot_graph_3d,
            plot_case_3d,
            plot_shishkin_3d,
            plot_case_edge_difference,
            plot_case_num_vs_exact,
            plot_edges_2d_simple,
            calculate_case_l2_error,
            calculate_case_D_error,

            # collocation
            get_edge_coeff,
            eval_f,
            get_g,
            generate_knot_vector_from_breaks,
            greville_points,
            find_span,
            ders_basis_funs,
            assemble_bspline_collocation,
            solve_bspline_collocation,
            eval_edge_u_ders,
            eval_edge_u,
            get_exact_solution_ders,
            calculate_errors_collocation,
            calculate_errors_infty,
            plot_convergence_rates_trig,

            # Helper
            normal_sign,
            fix_inflow_nodes!,
            run_MG_from_case,
            check_vertex_flux_condition_MG,
            run_star_shishkin_block_backslash,
            build_mesh,
            apply_vertex_stabilization_new!,

            # cases
            testcase_stargraph,
            testcase_barabasi,
            testcase_barabasi_with_cycle,
            exact_solution_triangle_WOx,
            testcase_triangle_exact_WOx,
            exact_solution_triangle_any,
            testcase_triangle_exact_any,
            exact_solution_star_Kumar_Leugering,
            testcase_star_exact_Kumar_Leugering,
            exact_solution_star_MMS,
            testcase_star_MMS_smooth,
            exact_solution_cycle_Lollipop,
            testcase_cycle_advection,
            solve_star_poly,
            testcase_star_poly,
            solve_star_trig,
            testcase_star_trig,

            # PCG
            sysof_equations,
            HEEInv,
            Schur_Mult,
            PCG_mod,

            # Setup_dG
            createH_DG_AdvectionDiffusion,

            # Setup_H_DiffAdv
            coth_safe,
            tau_supg,
            local_stiff,
            local_load,
            apply_dirichlet_blocks!,
            check_inflow_vertices,
            createH_AdvectionDiffusion,

            # Setup_H
            estimate_dofs,
            choose_sparse_index_type,
            createH,
            E_bar,
            E_hat,

            # Setup_Shishkin
            shishkin_grid,
            bakhvalov_nodes,
            local_stiff_std,
            local_load_std,
            assemble_system_blocks_shishkin,

            # smoothers
            get_macroscopic_sort,
            build_level_permutation,
            smoother_DGS,
            smoother_GMRES,
            smoother_ILU,

            # Examples
            evalBSpline,
            MG_example_stargraph,
            MG_example_Barabasi,
            run_MG_barabasi,
            MG_example_Barabasi_extended,
            MG_examples_varying_parameters,
            PCG_example,
            MG_example_AdvectionDiffusion,
            run_MG_stargraph,
            run_MG_DiffAdv_example,
            run_barabasi_example,
            run_barabasi_example_with_cycle,
            run_star_KumarLeugering,
            plot_convergence_rate_shishkin,
            plot_convergence_rate_SUPG,
            plot_convergence_rate_SUPG_comparison,
            run_star_KumarLeugering_dG,
            plot_convergence_rate_dG_comparison,
            calculate_case_dG_energy_error,
            plot_convergence_rate_pure_dG_comparison,
            run_star_KumarLeugering_pure_dG,
            run_MG_cycle,
            MG_advection_varying_parameters,
            MG_advection_Barabasi_extended,

            # Examples_other_stabilizers
            run_star_shishkin,
            run_star_bakhvalov





    include("AMG_Helper.jl")
    include("cases.jl")
    include("collocation.jl")
    include("createRHS.jl")
    include("Examples_other_stabilizers.jl")
    include("Examples.jl")
    include("GraphFunktionplotten.jl")
    include("Helper.jl")
    include("Multigrid_DiffAdv.jl")
    include("Multigrid.jl")
    include("PCG.jl")
    include("Setup_dG.jl")
    include("Setup_H_DiffAdv.jl")
    include("Setup_H.jl")
    include("Setup_Shishkin.jl")
    include("smoothers.jl")
end