using Test

using AdvDiffMetricGraphs

@testset "AdvectionDiffusion Metric Graph Tests" begin

    @testset "Diffusion-Reaction (DiffReac)" begin

        @test MG_example_stargraph(3, 2)
        @test MG_example_Barabasi(3, 2, nodes=500) 
        @test PCG_example()
    end

    @testset "Diffusion-Advection (DiffAdv)" begin
        @test MG_example_AdvectionDiffusion(3, 2)
        @test run_MG_stargraph()
        @test run_MG_DiffAdv_example()
        @test run_barabasi_example()
        @test run_barabasi_example_with_cycle()
        @test run_star_KumarLeugering()
        @test run_MG_cycle()
    end

    @testset "Stabilizers & Discontinuous Galerkin (otherStab)" begin
        # Check specific mesh formulations and error plots
        @test run_star_shishkin()
        @test plot_convergence_rate_shishkin()
        @test plot_convergence_rate_SUPG()
        @test plot_convergence_rate_SUPG_comparison()
        @test run_star_KumarLeugering_dG()
        @test plot_convergence_rate_dG_comparison()
        @test plot_convergence_rate_pure_dG_comparison()
        @test run_star_KumarLeugering_pure_dG()
    end

end