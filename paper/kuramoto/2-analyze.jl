if length(ARGS) == 0
    N_ics = 500
else
    N_ics = parse(Int, ARGS[1])
end

if length(ARGS) < 2
        k = 1
    else
        k = parse(Int, ARGS[2])
end

if length(ARGS) < 3
        dist = ARGS[3]
else
        dist = "uniform"
end
NAME = string("results-$N_ics-$k-",dist,".jld2")


if ! isfile(NAME)
    println("File results-$N_ics.jld2 does not exist. Run 1-simulate.jl")
    exit()
end

include("1-simulate.jl")

using Clustering

begin
    # Load dependencies
    import GraphPlot
    import Compose:cm, SVG, PDF
    import Gadfly
    import Colors
    import Cairo
    import Fontconfig
    import Plots
    using LaTeXStrings
    import PyPlot
end


begin
    using LinearAlgebra
    using SparseArrays
    using Parameters
    # Minimal fake run to get all types the JLD2 needs into scope.
    knp1 = ODEProblem(second_order_kuramoto, zeros(2*N), (0.,1.), par)
    knp_mcp1 = DEMCBBProblem(knp1, ic_gen, 1, par, par_vars, eval_ode_run_inertia, tail_frac)
    knp_sol1 = solve(knp_mcp1)
    D_k1 = distance_matrix(knp_sol1, knp_mcp1, [1.,0.,1.])

    @load NAME N_ics knp knp_sol knp_mcp D_k
end

FOLDER_NAME = string("pictures-$N_ics-$k-",dist)
if !(isdir(FOLDER_NAME))
	mkdir(FOLDER_NAME)
end
begin
    fdist = k_dist(D_k,4)
    # Plots.plot(collect(1:N_ics),fdist[1:N_ics])
    # Plots.plot(collect(1000:N_ics),fdist[1000:N_ics])
end

println("Heuristic for epsilon: $(median(KNN_dist_relative(D_k)))")

begin    # The main bifurcation diagram, this sometimes needs to be run twice or it crashes...
    min_c_size = N_ics/200
    if min_c_size < 5
        min_c_size = 5
    end
    db_eps = 5
    db_res = dbscan(D_k,db_eps,round(Int, min_c_size))
    # Only differentiate clusters that have at least 0.5% of the overall probability
    cluster_members = cluster_membership(knp_mcp, db_res, 0.25, 0.1);
    # sort!(cluster_members)
    println(size(cluster_members))
    # Plots.pgfplots()
    plt = Plots.plot(cluster_members, linecolor="white", linewidth=0.1, fillalpha=0.6, xlabel="Coupling K", legend=false) # , min_members = round(Int,N_ics/200/5)
    Plots.savefig(plt, joinpath(FOLDER_NAME,"kur-membership.pdf"))
end

N_clusters = size(cluster_members)[2]

cmr = cluster_measures(knp_mcp, knp_sol, db_res, 0.25, 0.1);

for i in 1:N_clusters
    plt = Plots.plot(cmr, 1, i)
    Plots.savefig(plt, joinpath(FOLDER_NAME,"kur-cluster$i-dim-means.pdf"))
end

for i in 1:N_clusters
    plt = Plots.plot(cmr, 2, i)
    Plots.savefig(plt, joinpath(FOLDER_NAME,"kur-cluster$i-dim-vars.pdf"))
end

# m = get_trajectory(knp_mcp,knp_sol, db_res, 3 ,only_sol=true)
# Plots.plot(m, vars=state_filter)

Plots.pyplot()

# Plot histograms for means
shists = cluster_measures_sliding_histograms(knp_mcp, knp_sol, db_res, 1, 0.25, 0.1, bin_edges=-10.5:0.2:10.5);

for i in 1:N_clusters
    plt = Plots.plot(shists, i)
    Plots.savefig(plt, joinpath(FOLDER_NAME,"kur-cluster$i-histo.pdf"))
end

# Plot histograms for vars
shists = cluster_measures_sliding_histograms(knp_mcp, knp_sol, db_res, 2, 0.25, 0.1);

for i in 1:N_clusters
    plt = Plots.plot(shists, i)
    Plots.savefig(plt, joinpath(FOLDER_NAME,"kur-cluster$i-histo-var.pdf"))
end


#Plot the graphs with the means of the frequencies:

cmeans = cluster_means(knp_sol, db_res)

fixed_layout = GraphPlot.spring_layout(g)
layout_func = (x) -> fixed_layout

cg = Plots.cgrad(:RdBu, range(-13., 13., length=100))

net_plot = GraphPlot.gplot(g, nodefillc=[d > 0. ? "blue" : "red" for d in drive], layout=layout_func)
Gadfly.draw(PDF(joinpath(FOLDER_NAME, "networks", "kur-network-power.pdf"), 16cm, 16cm), net_plot)

for i in 1:size(cmeans)[1]
    net_plot = GraphPlot.gplot(g, nodefillc=[cg[cmean] for cmean in cmeans[i,1,:]], layout=layout_func)
    Gadfly.draw(PDF(joinpath(FOLDER_NAME, "networks", "kur-cluster$i-network.pdf"), 16cm, 16cm), net_plot)
end
