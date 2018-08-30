################
## All functions that evaluate the solutions directly
###############

using DifferentialEquations, Miniball, Interpolations, Distances
import Distributions, StatsBase

# eval_ode_run, (sol, i) -> (evaluated_solution, repeat=False)
# evaluates each ODE run and computes all the statistics needed for the further calculations
# right now these are: mean, std, skewness, relative entropy / KL div. to a gaussian and curve entropy
#
# input:
# sol :: result of one of the monte carlo ode run, should have only timesteps with constant time intervals between them
# i :: Int, number of iteration
# state_filter :: array with indicies of all dimensions that should be evaluated
function eval_ode_run_old(sol, i, state_filter::Array{Int64,1})

    (N_dim, N_t) = size(sol)

    m = zeros(Float64, N_dim)
    std = zeros(Float64, N_dim)
    #skew = zeros(Float64, N_dim)
    kl = zeros(Float64, N_dim)

    for i_dim in state_filter
        sol_i = sol[i_dim,2:end] # with the saveat solver option, tspan[1] is always saved as well but is not part of the transient.
        (m[i_dim],std[i_dim]) = StatsBase.mean_and_std(sol_i; corrected=true)
        #skew[i_dim] = StatsBase.skewness(sol_i, m[i_dim])

        # if the std is zero or extremly close to zero (meaning all values are the same, thus a fix point), the KL divergence relative to a Gaussian of same mean and std is hard to numerically compute.
        # we set it to zero in this case because both the empirical and the reference distribtuion are in this case approximately delta distributions at the same locaction, thus identical, having KL=0
        if std[i_dim] < 1e-10
            kl[i_dim] = 0
        else
            ref_dist = Distributions.Normal(m[i_dim], std[i_dim]) # old
            kl[i_dim] = empirical_1D_KL_divergence(sol_i, ref_dist, 25)
            #kl[i_dim] = empirical_1D_KL_divergence(sol_i, m[i_dim], std[i_dim])
        end
    end

    # for test purposes we first write the same N-dimensonal curve entropy in every dimension
    ce = curve_entropy(sol.u[2:end])
    ((m, std, kl, ce), false)
end

function eval_ode_run(sol, i, state_filter::Array{Int64,1}, eval_funcs::Array{T,1}, global_eval_funcs::Array{T,1}) where T <: Function
    (N_dim, N_t) = size(sol)

    N_dim_measures = length(eval_funcs) + 2 # mean and var are always computed
    N_dim_global_measures = length(global_eval_funcs)
    dim_measures = [zeros(Float64, N_dim) for i=1:N_dim_measures]
    global_measures = zeros(Float64, N_dim_global_measures)

    # per dimension measures
    for i_dim in state_filter
        sol_i = sol[i_dim,2:end]
        (dim_measures[1][i_dim],dim_measures[2][i_dim]) = StatsBase.mean_and_std(sol_i; corrected=true)
        for i_meas=1:N_dim_measures
            dim_measures[3][i_dim] = eval_funcs[i_meas](sol_i, dim_measures[1][i_dim], dim_measures[2][i_dim])
        end
    end

    # measures using all dimensions
    for i_meas=1:N_dim_global_measures
        global_measures[i_meas] = global_eval_funcs[i_meas](sol.u[2:end])
    end
    (tuple(dim_measures...,global_measures...),false)
end

function eval_ode_run(sol, i)
    (N_dim, __) = size(sol)
    state_filter = collect(1:N_dim)
    eval_funcs = [empirical_1D_KL_divergence_hist]
    global_eval_funcs = [curve_entropy]
    eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs)
end


# MonteCarloProblem needs a function with only (sol, i) as inputs and this way the default of all dimensions beeing evaluated is easier to handle than with an optional/keyword argument
function eval_ode_run_old(sol, i)
    (N_dim, __) = size(sol)
    state_filter = collect(1:N_dim)
    eval_ode_run_old(sol, i, state_filter)
end

# include a return code check and repeat if integration failed. if this is not used with randomly generated initial conditions, this could leed into an endless loop!
function eval_ode_run_repeat(sol, i)
    if (sol.retcode != :Success) & (sol.retcode != :Default)
        return ((),true)
    end
    eval_ode_run(sol, i )
end

# include a return code check and set all results to Inf is the integration fails due to one or more variables exploding towards infitiy
function eval_ode_run_inf(sol, i)
    if (sol.retcode == :DtLessThanMin)
        last = sol.u[end]
        N_dim = length(last)
        inf_flag = false
        for i=1:N_dim
            if abs(last[i]) > 1e12
                inf_flag = true
            end
        end
        return ((ones(N_dim).*Inf,ones(N_dim).*Inf,ones(N_dim).*Inf,ones(N_dim).*Inf),false)
    end
    eval_ode_run(sol,i)
end

# checks if any of the results is Inf or NaN and returns the indices in a dictionary
function check_inf_nan(sol::myMCSol)
    N = sol.N_mc
    N_meas = sol.N_meas
    nan_inf = Dict("Inf" => ([]), "NaN" => ([]))
    for i=1:N
        for i_meas=1:N_meas
            if sum(isnan.(sol.sol[i][i_meas])) > 0
                push!(nan_inf["NaN"], [i,i_meas])
            end
            if sum(isinf.(sol.sol[i][i_meas])) > 0
                push!(nan_inf["Inf"], [i,i_meas])
            end
        end
    end
    nan_inf
end

# empirical_1D_KL_divergence
# NOT USED by default (instead the Perez-Cruz estimate below is used)
#
# based on histograms (might actually not be a good estimator for KL)
# u :: Input Array
# reference pdf: e.g. Normal(mean,std)
# hist_bins: number of bins of the histogram to estimate the empirical pdf of the data
function empirical_1D_KL_divergence_hist(u::AbstractArray, reference_pdf::Distributions.UnivariateDistribution, hist_bins::Int)
    hist = StatsBase.fit(StatsBase.Histogram, u; closed=:left, nbins=hist_bins)
    hist = StatsBase.normalize(hist)
    bin_centers = @. hist.edges[1] + 0.5*(hist.edges[1][2] - hist.edges[1][1])
    refpdf_discrete = Distributions.pdf.(reference_pdf, bin_centers[1:end-1])

    StatsBase.kldivergence(hist.weights, refpdf_discrete)
end
empirical_1D_KL_divergence_hist(u::AbstractArray, mu::Number, sig::Number, N_bins::Int64=25) = empirical_1D_KL_divergence_hist(u, Distributions.Normal(mu,sig), N_bins)

# KL divergence
# estimate based on Perez-Cruz (IEEE, 2008)
# estimates the KL divergence by using linearly interpolated empirical CDFs.
# TO-DO: for large time series (N>10000) there is small risk that this yields Inf because the spacing between the samples becomes so small that the precision is not high enough to yield finite numbers.
function empirical_1D_KL_divergence_pc(u::AbstractArray, mu::Number, sig::Number)

    N = length(u)
    Us = sort(u)
    Us_u = unique(Us)
    if length(Us_u)==1
        return 0.
    end

    eps = BigFloat(0.5 * minimum(diff(Us_u))) # we need very high precision for many values to not underflow to 0.

    ecdf_u = ecdf_pc(Us, Us_u, eps)
    dpc(x::BigFloat) = ecdf_u[x] - ecdf_u[x - eps]

    #ecdf_samp = ecdf_pc(samps, samps_u, eps2) # OLD VERSION
    #dqc(x::Real) =  ecdf_samp[x] - ecdf_samp[x - eps]

    normal_cdf(x::BigFloat) = BigFloat(0.5)*(BigFloat(1.)+erf((x-BigFloat(mu))/sqrt(2*BigFloat(sig)*BigFloat(sig))))
    dqc(x::BigFloat) = normal_cdf(x) - normal_cdf(x-eps)

    kld::Float64 = 0
    for i=1:N
        dp = dpc(BigFloat(Us[i]))
        dq = dqc(BigFloat(Us[i]))
        if (dp==0.) & (dq > 0.)
            kld += 0
        elseif (dp > 0.) & (dq > 0.)
            kld += log(dp/dq)
        else
            kld += Inf
        end
    end
    kld *= 1./N
    kld -= 1    # Perez-Cruz estimator converges as KL_pc - 1 -> KL
end

#
# Empirical Cumulative Densitiy function for KL divergence (w/ Heaviside(0)=1/2 and linear approx. between the points) based on Perez-Cruz, IEEE, 2008
# assumes that X is sorted and Xu is unique(X)!
function ecdf_pc(X::AbstractArray, Xu::AbstractArray, eps::BigFloat)
    N = length(X)
    Nu = length(Xu)
    ecdf = (0.5:1:(N-0.5))/N # 0.5 because Heaviside(0)=1/2 in this defintion
    eps = 0.05*eps
    dat_range = 2*(X[end] - X[1])

    # interpolation won't work with doublicate values and some extreme cases cases like only 2 or 3 unique values / delta peaks need extra care
    # the aim is to add additonal values at xi - eps, just a little bit infront of the delta peaks, to get a good cdf estimate

    # if all values are unique this is not necessary
    if Nu < N
        iu = [] # indices of unique elements, including the last element of series of non-unique elements
        i_fnu = [] # indices of the first element of series of non-unique elements

        if X[1]==X[2]
            push!(i_fnu,1)
        end
        for i=2:(N-1)
            if X[i-1]!=X[i]
                push!(iu, i-1)
                if X[i]==X[i+1]
                    push!(i_fnu, i)
                end
            end
        end
        if X[N-1]!=X[N]
            push!(iu, Nu-1)
        end
        push!(iu, N)
        x_fnu = X[i_fnu] .- eps
        ecdf_fnu = ecdf[i_fnu]

        # we also add 0 and 1 for a good asymptotic Behaviour
        # this way of array construction is quite slow, could replace with something that performs better instead

        itp = interpolate((sort([X[1]-2*dat_range,Xu...,x_fnu...,X[end]+2*dat_range]),), sort([0,ecdf[iu]...,ecdf[i_fnu]...,1]), Gridded(Linear()))
    else
        itp = interpolate((X,), ecdf, Gridded(Linear()))
    end
    itp
end
ecdf_pc(X::AbstractArray) = ecdf_pc(X, unique(X), 0.5*minimum(diff(unique(X))))
ecdf_pc(X::AbstractArray, Xu::AbstractArray, eps::Float64) = ecdf_pc(X, unique(X), BigFloat(0.5*minimum(diff(unique(X)))))

# curve entropy according to Balestrino et al, 2009, Entropy Journal
# could be used as an additional measure for the clustering
# bounded [0,1]
#
function curve_entropy(u::Array{Array{Float64,1},1}, r_eps::Float64=1e-15)
    D = mcs_diameter(u)
    if D > r_eps
        ce = log(curve_length(u)/D)/log(length(u) - 1)
    else
        ce = 0. # if the curve is just a point, its entropy is 0 (the logarithm would yield NaN)
    end
    ce
end

function curve_length(u::Array{Array{Float64,1},1})
    L::Float64 = 0.
    for it=2:size(u)[1]
        L += euclidean(u[it],u[it-1])
    end
    L
end

# Minimal Covering (Hyper)sphere of the points of the Curve
function mcs_diameter(u::Array{Array{Float64,1},1})
    # miniball routine needs a N_t x d matrix
    mcs = miniball(transpose(hcat(u...)))
    2.*sqrt(mcs.squared_radius)
end
