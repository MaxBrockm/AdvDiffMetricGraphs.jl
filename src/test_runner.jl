########################################
# This file runs all examples from 
# Examples.jl to check for dependency errors
########################################
include("Examples.jl")
"""
    test_runner(test_type::Symbol)

Test all examples from Examples.jl for depedency errors. Possible test_types are: 
    :misc
    :DiffReac
    :DiffAdv
    :otherStab
"""
function test_runner(test_type::Symbol)
    if test_type==:misc
        test_gmres_isolated();
        test_matrix();
    elseif test_type==:DiffReac
        # Diffusion reaction
        MG_example_stargraph(3,2);
        MG_example_Barabasi(3,2);
        MG_examples_varying_parameters();
        MG_example_stargraph_L2H1();

        PCG_example();
    elseif test_type==:DiffAdv
        # Diffusion Advection
        MG_example_AdvectionDiffusion();
        run_MG_stargraph();
        run_MG_DiffAdv_example();
        run_barabasi_example();
        run_star_KumarLeugering();
    elseif testtype==:otherStab
        run_star_shishkin();
    else 
        @error "Unsupported test type. Chose between: :misc, :DiffReac and :DiffAdv"
    end
end


