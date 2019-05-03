# hopefully this file will contain plot recipes for all plots at some point, so far it is not complete

# documentation for these can be found in the doc strings of the respective structs

using RecipesBase

@recipe function f(res::ClusterMembershipResult; plot_index=nothing)

    _3d = res.multidim_flag

    if typeof(plot_index)<:Number
        plot_index = [plot_index]
    end

    if plot_index==nothing
        N_cluster = _3d ? size(res.data)[3] : size(res.data)[2]

        plot_index = 1:N_cluster
    end

    labels = []
    if 1 in plot_index
        push!(labels, "1: Outlier")
    end

    for ip in plot_index
        if !(ip==1)
            push!(labels, string(ip))
        end
    end

    if !(_3d)
        x = res.par

        #xlabel --> "Parameter"
        ylabel --> "Approximate Relative Basin Volume"

        fillto --> 0
        linecolor --> :black
        seriestype --> :path
        y = cumsum(res.data[:, plot_index], dims = 2)[:,end:-1:1]

    else

        x = res.par[:,:, 1]
        y = res.par[:,:, 2]
        z = res.data[:,:, plot_index]
        seriestype --> :surface
        xlabel --> "Parameter 1"
        ylabel --> "Parameter 2"
        zlabel --> "Approximate Relative Basin Volume"
    end
    grid --> true
    label -> labels
    fillalpha --> 0.5
    gridstyle --> :dot
    gridalpha --> 1.0
    gridlinewidth --> 2
    linecolor --> "white"
    linewidth --> 2
    minorgrid --> :on

    xyz = _3d ? (x,y,z) : (x,y)

    xyz
end

@recipe function f(res::ClusterMeasureHistogramResult, plot_index::Int)

    _3d = res.multidim_flag
    if _3d
        error("3D Plot not yet supported.")
    end

    x = res.par[1]
    y = res.hist_edges
    z = collect(res.hist_vals[plot_index, :, :]')
    seriestype := :heatmap

    colorbar --> :right
    colorbar_title --> "rel. Magnitude"

    (x, y, z)
end

@recipe function f(res::ClusterMeasureResult, measure_index::Int, plot_index::Int)

    _3d = res.multidim_flag
    if _3d
        error("3D Plot not yet supported.")
    end

    x = res.par

    N_c, N_m, __, __ = size(res.cluster_measures)

    if measure_index <= N_m
        y = collect(res.cluster_measures[plot_index, measure_index, :, :]')
    else
        y = collect(res.cluster_measures_global[plot_index, measure_index, :, :]')
    end
    legend --> :none

    (x, y)
end
