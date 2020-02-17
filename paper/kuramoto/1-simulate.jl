begin
    # Working in the local environment
    using Pkg
    Pkg.activate(@__DIR__)
end

begin
    # Load dependencies
    using LightGraphs
    using Random
    using DifferentialEquations
    using Distributions
    using JLD2
end

using MCBB

begin
    seed = 5386748129040267798
    Random.seed!(seed)
    # Set up the parameters for the network
    N = 30 # in this case this is the number of oscillators, the system dimension is twice this value
    g = random_regular_graph(N, 3)
    E = incidence_matrix(g, oriented=true)
    drive = [isodd(i) ? +1. : -1. for i = 1:N]
    par = second_order_kuramoto_parameters(N, 0.1, 5., E, drive)
    T = 5000.
end


begin
    # These functions define the distributions from which initial conditions and
    # parameters are drawn, as well as which parameter to vary.
    
    if length(ARGS) < 2
	k = 1
    else 
        k = parse(Int, ARGS[2])
    end

    if length(ARGS) < 3 
	dist = "uniform"
        ic_gen = () -> [rand(Uniform(-pi,pi),N); rand(Uniform(-k*pi,k*pi),N)]
    else 
	dist = ARGS[3]
        if ARGS[3] == "uniform"
                ic_gen = ()->[rand(Uniform(-pi,pi),N); rand(Uniform(-k*pi,k*pi),N)]
        elseif ARGS[3] == "normal"
                ic_gen = ()->[rand(Normal(0,pi/2),N); rand(Uniform(0,k*pi),N)]
        else
                error("unknown distribution argument handed over to script")
        end
    end 
	
    if length(ARGS) == 0
        N_ics = 500
    else
        N_ics = parse(Int, ARGS[1])
    end

    K_range = (i)-> 10. * i / N_ics
    par_vars = (:coupling, K_range)
    tail_frac = 0.9
    state_filter = collect(N+1:2*N)
end

# evaluation of the solutions, we set a state_filter to only analyse the
# frequencies and not the phases.
function eval_ode_run_inertia(sol, i)
    state_filter = collect(N+1:2*N)
    eval_funcs = [MCBB.mean, MCBB.std]
    global_eval_funcs = []
    eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs)
end

begin
    knp = ODEProblem(second_order_kuramoto, zeros(2*N), (0.,T), par)
    knp_mcp = DEMCBBProblem(knp, ic_gen, N_ics, par, par_vars, eval_ode_run_inertia, tail_frac)
end

display(knp_mcp.ic)

NAME = string("results-$N_ics-$k-",dist,".jld2")
println(NAME)

if ! isfile(NAME)
    println("Starting simulation with $N_ics runs")
    knp_sol = solve(knp_mcp)
    D_k = distance_matrix(knp_sol, knp_mcp, [1.,0.,1.])

    println("Saving results")
    jldopen(NAME, true, true, true, IOStream) do file
        file["N_ics"] = N_ics
        file["knp"] = knp
        file["knp_sol"] = knp_sol
        file["knp_mcp"] = knp_mcp
        file["D_k"] = D_k
    end
elseif length(ARGS) > 1
    println("Starting simulation with $N_ics runs")
    knp_sol = solve(knp_mcp)
    D_k = distance_matrix(knp_sol, knp_mcp, [1.,0.,1.])


    println("Saving results")
    jldopen(NAME, true, true, true, IOStream) do file
        file["N_ics"] = N_ics
        file["knp"] = knp
        file["knp_sol"] = knp_sol
        file["knp_mcp"] = knp_mcp
        file["D_k"] = D_k
    end
end

