using Test

using AdvDiffMetricGraphs

@testset "AdvectionDiffusion Metric Graph Tests" begin

    @testset "Diffusion-Reaction (DiffReac)" begin
        @test (MG_example_stargraph(3, 2; do_plots=false); true)
        @test (MG_example_Barabasi(3, 2; nodes=500, do_plots=false); true) 
        @test (PCG_example(); true)
    end

    @testset "Diffusion-Advection (DiffAdv)" begin
        @test (MG_example_AdvectionDiffusion(3, 2; do_plots=false); true)
        @test (run_MG_stargraph(do_plots=false); true)
        @test (run_MG_DiffAdv_example(do_plots=false); true)
        @test (run_barabasi_example(); true)
        @test (run_barabasi_example_with_cycle(); true)
        @test (run_star_KumarLeugering(do_plots=false); true)
        @test (run_MG_cycle(do_plots=false); true)
    end

    @testset "Stabilizers & Discontinuous Galerkin (otherStab)" begin
        # Check specific mesh formulations and error plots
        @test (run_star_shishkin(do_plots=false); true)
        @test (plot_convergence_rate_shishkin(do_plots=false); true)
        @test (plot_convergence_rate_SUPG(do_plots=false); true)
        @test (plot_convergence_rate_SUPG_comparison(do_plots=false); true)
        @test (run_star_KumarLeugering_dG(do_plots=false); true)
        @test (plot_convergence_rate_dG_comparison(do_plots=false); true)
        @test (plot_convergence_rate_pure_dG_comparison(do_plots=false); true)
        @test (run_star_KumarLeugering_pure_dG(do_plots=false); true)
    end

    @testset "Collocation" begin
        @test (plot_convergence_rates_trig(do_plots=false); true)
    end
   
end