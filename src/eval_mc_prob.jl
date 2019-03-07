################
## All functions that evaluate the solutions directly
###############

using DifferentialEquations, Interpolations, Distances
# using Miniball
import Distributions, StatsBase

"""
    eval_ode_run

Evaluation functions for the `MonteCarloProblem`. Given a set of measures the solution `sol` is evaluated seperatly per dimension. An additional set of global measures take in the complete solution and return a single number. Handing over the function to `DEMCBBProblem` (and thus also to `MonteCarloProblem`) the expected signature is `(sol, i::Int) -> (results, repeat::Bool)`. Here, there are several more general versions that can be adjusted to the experiment.

    eval_ode_run(sol, i, state_filter::Array{Int64,1}, eval_funcs::Array{<:Function,1}, global_eval_funcs::AbstractArray; failure_handling::Symbol=:None, cyclic_setback::Bool=false, replace_inf=nothing)

* `sol`: solution of one of the MonteCarloProblem runs, should have only timesteps with constant time intervals between them
* `i`: Int, number of iteration/run
* `state_filter`: Array with indicies of all dimensions (of the solutions) that should be evaluated
* `eval_funcs`: Array of functions that should be applied to every dimension of the solution. Signature: `(sol::AbstractArray) -> measure` or `(sol::AbstractArray, previous_results::AbstractArray) -> measure` depending on the value of `flag_past_measures`.
* `global_eval_funcs`: Array of functions that should be applied to the complete N-dimensional solution, signature (N-Dim Array w/ Samples ::AbstractArray, Mean::Number, Std::Number) -> Measure
* `failure_handling`: How failure of integration is handled. Should be `:None` (do no checks), `:Inf` (If `retcode==:DtLessThanMin: return Inf`) or `:Repeat` (If no succes, repeat the trial (only works with random initial conditions))
* `cyclic_setback`: Bool, if true ``N*2\\pi`` is substracted from the solution so that the first element of the solution that is analysed is within ``[-\\pi, \\pi]``. Usefull e.g. for phase oscillators.
* `replace_inf`: Number or Nothing, if a number replaces all Infs in the solution with this number. Can be usefull if one still wants to distinguish between different solutions containing Infs, +Inf is replaced by the Number, -Inf by (-1)*Number.
* `flag_past_measures::Bool`: If true als function within `eval_funcs` also receive the previous results (of the other measures for the same dimension) as an extra arguments. Thus all functions need to have a signature `(sol::AbstractArray, previous_results::AbstractArray) -> measure`. If false the functions only receive the solution vector, thus the function should have the signature `(sol::AbstractArray) -> measure`

# Example function

In order to derive a valid evaluation function from this for the `MCBBProblem` one can define a function similar to this:

    function my_eval_ode_run(sol, i)
        N_dim = length(sol.prob.u0)
        state_filter = collect(1:N_dim)
        eval_funcs = [mean, std]
        global_eval_funcs = []
        eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs)
    end

# Utilizing previous results

If one wants to utilze the previous results (and don't compute measures twice), one has to use the `flag_past_measures=true` option. An example could read:

    function my_eval_ode_run(sol, i)
        N_dim = length(sol.prob.u0)
        state_filter = collect(1:N_dim)
        meanval(u::AbstractArray, past_measures::AbstractArray) = StatsBase.mean(u)
        standarddev(u::AbstractArray, past_measures::AbstractArray) = StatsBase.std(u; mean=dim_measures[1], corrected=false)
        eval_funcs = [meanval, standarddev, empirical_1D_KL_divergence_hist]
        global_eval_funcs = []
        eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs; flag_past_measures=true)
    end

Latter function is also already available as a default `eval_ode_run` in this library. The order of the functions is important. In this example `meanval` will always get an empty array as the second argument, `standarddev` will get an array with the result from `meanval` as the only value and ` empirical_1D_KL_divergence_hist` will get an additional array with the results from `meanval` and `standarddev`.

    eval_ode_run(sol, i)

Default `eval_ode_run`, identical to the code above.

# Continue Integration / Response Analysis

     eval_ode_run(sol, i, state_filter::Array{Int64,1}, eval_funcs::Array{<:Function,1}, global_eval_funcs::AbstractArray, par_var::OneDimParameterVar, eps::Float64, par_bounds::AbstractArray, distance_matrix_func; failure_handling::Symbol=:None, cyclic_setback::Bool=false, flag_past_measures::Bool=false, N_t::Int=200, alg=nothing, debug::Bool=false, kwargs...)

Evaluation function that continues each integration and computes the same measures for `par+eps` and `par-eps`. Returns the results of the usual `eval_ode_run` (all measures) and additionally the response of the distance function to the paramater increase/decrease.

* `par_var`: `ParameterVar` struct, same as handed over to the problem type.
* `eps`: Number, response at par+/-eps
* `distance_matrix_func`: Same distance matrix functions that will also be used for the later analysis/clustering, expected signature: `(sol::MCBBSol, prob::MCBBProblem) -> D::AbstractArray`, . Attension: if the weight vector is provided this version of the distance it needs to have one less element as the function later used before clustering because the result of the response analysis is an additional measure. 
* `N_t`: Time steps for the continued integration
* `alg`: Algorithm for `solve()`
* `debug`: If true, also returns the DifferentialEquations problem solved for the continuation.
* all further keyword arguments will be handed over to `solve(prob::DEMCBBProblem, ...)`
"""
function eval_ode_run(sol, i, state_filter::Array{Int64,1}, eval_funcs::AbstractArray, global_eval_funcs::AbstractArray; failure_handling::Symbol=:None, cyclic_setback::Bool=false, replace_inf=nothing, flag_past_measures=false)
    N_dim = length(sol.prob.u0)
    N_dim_measures = length(eval_funcs)  # mean and var are always computed

    if N_dim_measures < 1
        error("No per dimension measures")
    end

    N_dim_global_measures = length(global_eval_funcs)

    if failure_handling==:None
        failure_handling=:None # do nothing
    elseif failure_handling==:Inf
        if (sol.retcode == :DtLessThanMin) | (sol.retcode == :Unstable)
            # in case it is so unstable that the solution is empty as no results are returned
            if length(sol) == 0
                inf_flag = true
            else
                last = sol.u[end]
                inf_flag = false
                for i=1:N_dim
                    if abs(last[i]) > 1e11
                        inf_flag = true
                    end
                end
            end
            if inf_flag
                dim_measures = [Inf.*ones(N_dim) for i=1:N_dim_measures]
                global_measures = [Inf for i=1:N_dim_global_measures]
                return (tuple(dim_measures...,global_measures...),false)
            else
                @warn "Failure Handling Warning, DtLessThanMin but Solution not diverging."
            end
        end
    elseif failure_handling==:Repeat
        if (sol.retcode != :Success) & (sol.retcode != :Default)
            return ((),true)
        end
    else
        error("failure_handling symbol not known")
    end
    (N_dim, N_t) = size(sol)

    if replace_inf != nothing
        inf_ind = isinf.(sol)
        pinf_ind = inf_ind .& (sol .> 0)
        minf_ind = inf_ind .& (.~pinf_ind)

        for i_dim=1:N_dim
            for it=1:N_t
                if pinf_ind[i_dim,it]
                    sol.u[it,:][i_dim] = replace_inf
                end
                if minf_ind[i_dim,it]
                    sol.u[it,:][1][i_dim] = -1 * replace_inf
                end
            end
        end
    end

    dim_measures = [zeros(Float64, N_dim) for i=1:N_dim_measures]
    global_measures = zeros(Float64, N_dim_global_measures)
    # per dimension measures
    for i_dim in state_filter
        sol_i = sol[i_dim,2:end]
        if cyclic_setback
            _cyclic_setback!(sol_i)
        end

        if flag_past_measures
            for i_meas=1:N_dim_measures
                # collect previous measures
                past_measures = zeros(i_meas-1)
                for j_meas=1:(i_meas-1)
                    past_measures[j_meas] = dim_measures[j_meas][i_dim]
                end
                dim_measures[i_meas][i_dim] = eval_funcs[i_meas](sol_i, past_measures)
            end
        else
            for i_meas=1:N_dim_measures
                dim_measures[i_meas][i_dim] = eval_funcs[i_meas](sol_i)
            end
        end
    end

    # measures using all dimensions
    for i_meas=1:N_dim_global_measures
        global_measures[i_meas] = global_eval_funcs[i_meas](sol[:,2:end])
    end
    (tuple(dim_measures...,global_measures...),false)
end

# MonteCarloProblem needs a function with only (sol, i) as inputs and this way the default of all dimensions beeing evaluated is easier to handle than with an optional/keyword argument
function eval_ode_run(sol, i)
    N_dim = length(sol.prob.u0)
    state_filter = collect(1:N_dim)
    meanval(u::AbstractArray, past_measures::AbstractArray) = StatsBase.mean(u)
    standarddev(u::AbstractArray, past_measures::AbstractArray) = StatsBase.std(u; mean=past_measures[1], corrected=true)
    eval_funcs = [meanval, standarddev, empirical_1D_KL_divergence_hist]
    global_eval_funcs = []
    eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs; flag_past_measures=true)
end

# evaluation function that also includes a response analysis with a contiuation of the integration
# it records the differnce in Distance of the measures chosen thus each time recording a pair = 1(dp,dD)
function eval_ode_run(sol, i, state_filter::Array{Int64,1}, eval_funcs::Array{<:Function,1}, global_eval_funcs::AbstractArray, par_var::OneDimParameterVar, eps::Float64, par_bounds::AbstractArray, distance_matrix_func; failure_handling::Symbol=:None, cyclic_setback::Bool=false, flag_past_measures::Bool=false, N_t::Int=200, alg=nothing, debug::Bool=false, kwargs...)

    N_dim = length(sol.prob.u0)
    probi = sol.prob

    meas_1 = eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs; failure_handling=failure_handling, cyclic_setback=cyclic_setback)

    # we define a new MonteCarloProblem, so that we can reuse all the other old code. it has only three problems to solve, (p - eps, p, p+eps) for a relativly short integration time.
    ic = zeros(eltype(sol),3,N_dim)
    par = zeros(eltype(sol),3,1)
    # new IC is end point of last solution
    ic[1,:] = sol[end]
    ic[2,:] = sol[end]
    ic[3,:] = sol[end]

    # need par_var tuple hear as well -> maybe do it as a field of the problem structure
    par[1,1] = getfield(probi.p, par_var.name) - eps
    par[2,1] = getfield(probi.p, par_var.name)
    par[3,1] = getfield(probi.p, par_var.name) + eps
    if par[1,1] < par_bounds[1]
        par[1,1] = par_bounds[1]
    end
    if par[3,1] > par_bounds[2]
        par[3,1] = par_bounds[2]
    end

    new_tspan = (probi.tspan[1],probi.tspan[1]+0.25*(probi.tspan[2]-probi.tspan[1]))
    #new_tspan = probi.tspan
    function new_prob(baseprob, i, repeat)
        n_prob = remake(baseprob, u0=ic[i,:])
        n_prob = remake(n_prob, tspan=new_tspan)
        custom_problem_new_parameters(n_prob, par_var.new_par(probi.p; Dict(par_var.name => par[i,1])...))
    end

    mcp = MonteCarloProblem(probi, prob_func=new_prob, output_func=(sol,i)->eval_ode_run(sol, i, state_filter, eval_funcs, global_eval_funcs; failure_handling=failure_handling, cyclic_setback=cyclic_setback, flag_past_measures=flag_past_measures))
    bamcp = DEMCBBProblem(mcp, 3, 0., ic, par, par_var) # 3 problems and no transient time
    if alg==nothing
        mcpsol = solve(bamcp, N_t=N_t, parallel_type=:none, kwargs...)
    else
        mcpsol = solve(bamcp, alg, N_t=N_t, parallel_type=:none, kwargs...)
    end

    D = distance_matrix_func(mcpsol, bamcp)

    # symmetric difference
    dD = (D[1,2] + D[2,3])/2.
    if debug
        (tuple(meas_1[1]...,dD,[bamcp]),false)
    else
        (tuple(meas_1[1]...,dD),false)
    end
end


"""
    check_inf_nan(sol::myMCSol)

Checks if any of the results is `inf` or `nan` and returns the indices in a dictionary with keys `Inf` and `NaN`
"""
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


"""
    _cyclic_setback!(arr_in::AbstractArray)

Function helping to evaluate phase oscillators / other cyclic systems. Often these systems produce growing values. This routine substracts ``N\\cdot 2\\cdot \\pi`` from the values of the array so that `arr_in[1]`` is in ``[-\\pi,\\pi]``. Works inplace.
"""
function _cyclic_setback!(arr_in::AbstractArray)
    arr1_tmp = arr_in[1]

    # transform arr_in[1] back to [-pi,pi]
    arr_in[1] = mod(arr_in[1],2pi)
    arr_in[1] > pi ? arr_in[1] -= 2pi : arr_in[1]

    subtr = arr1_tmp - arr_in[1]

    arr_in[2:end] = arr_in[2:end] .- subtr;
end

"""
    empirical_1D_KL_divergence_hist(u::AbstractArray, mu::Number, sig::Number, hist_bins::Int=31, n_stds::Number=3, sig_tol=1e-4::Number)

One measure that can be used with `eval_ode_run`. Computes the empirical Kullback-Leibler Divergence of the input to a normal distribution with the same mean and std as the input, thus it is a measure of normality. This version does this with histograms.

* `u`: Input Array
* `mu`: Mean of `u`
* `sig`: Std of `u`
* `hist_bins`: number of bins of the histogram to estimate the empirical pdf of the data
* `n_stds`: Interval that the histogram covers in numbers of stds (it covers  mean +/- n_stds*std)
* `sig_tol`: At times, all KL div methods run into difficulties when `sig` gets really small, for `sig<sig_tol` 0 is returned as a result because in the limit of `sig` -> 0 the reference distribution is a delta distribution and the data is constant thus also a delta distribution. hence the distributions are identical and the KL div should be zero.
"""
function empirical_1D_KL_divergence_hist(u::AbstractArray, mu_sig::AbstractArray, hist_bins::Int=31, n_stds::Number=3, sig_tol=1e-4::Number)

   mu = mu_sig[1]
   sig = mu_sig[2]
   if sig < sig_tol # very small sigmas lead to numerical problems.
       return 0. # In the limit sig -> 0, the reference distribution is a delta distribution and the data is constant thus also a delta distribution. hence the distributions are identical and the KL div should be zero.
   end

   # first we calculate the bins (automatic bin calculation leads to errors)
   # they range from mean-3*sigma to mean+3*sigma
   reference_pdf =  Distributions.Normal(mu,sig)
   if iseven(hist_bins)
       hist_bins += 1
   end
   k = (hist_bins - 1)/2. # number of bins on each side of the histogram
   bin_width = (n_stds*sig) / k
   bin_centers = mu-k*bin_width:bin_width:mu+k*bin_width
   bin_edges = collect(bin_centers .- bin_width/2.)
   push!(bin_edges, k*bin_width + bin_width/2.)

   hist = StatsBase.fit(StatsBase.Histogram, u, StatsBase.AnalyticWeights(ones(length(u))), bin_edges; closed=:left)
   StatsBase.normalize!(hist, mode=:probability)

   reference_pdf = Distributions.Normal(mu,sig)
   refpdf_discrete = Distributions.pdf.(reference_pdf, bin_centers)
   refpdf_discrete ./= sum(refpdf_discrete)
   StatsBase.kldivergence(hist.weights, refpdf_discrete)
end

"""
    empirical_1D_KL_divergence_pc(u::AbstractArray, mu::Number, sig::Number)

One measure that can be used with `eval_ode_run`. Computes the empirical Kullback-Leibler Divergence of the input to a normal distribution with the same mean and std as the input, thus it is a measure of normality. This version does this based on a linearly interpolated emperical CDF, see Perez-Cruz (IEEE, 2008). This version can run into numerical difficulties for discrete systems with alternating inputs like [a,-a,a,a] and large 2a. For reasonable continous input it is a better and parameter free approximation to the KL divergence than the histogram estimator.

* `u`: Input Array
* `mu`: Mean of `u`
* `sig`: Std of `u`
"""
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
    kld *= 1. /N
    kld -= 1    # Perez-Cruz estimator converges as KL_pc - 1 -> KL
end

"""
    ecdf_pc(X::AbstractArray, Xu::AbstractArray, eps::BigFloat)

Empirical Cumulative Densitiy function for KL divergence (w/ Heaviside(0)=1/2 and linear approx. between the points) based on Perez-Cruz, IEEE, 2008. Assumes that `X` is sorted and `Xu` is `unique(X)`!

* `eps`: Only needed if not all elements of `X` are unique for interpolation as interpolation won't work with doublicate values and some extreme cases cases like only 2 or 3 unique values / delta peaks need extra care. The aim is to add additonal values at `xi - eps`, just a little bit infront of the delta peaks, to get a good cdf estimate. Default argument is `eps=0.5*minimum(diff(unique(X))))`.
"""
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
ecdf_pc(X::AbstractArray) = ecdf_pc(X, unique(X), BigFloat(0.5*minimum(diff(unique(X)))))

"""
    wasserstein_ecdf(u::AbstractArray, mu::Number, sig::Number)

One measure that can be used with `eval_ode_run`. Computes the 1-wasserstein distance based on ECDFs.

* `u`: Input Array
* `mu`: Mean of `u`
* `sig`: Std of `u`
"""
function wasserstein_ecdf(u::AbstractArray, mu::Number, sig::Number)
    if sig < 1e-10
        return 0.
    end

    N = length(u)
    us = sort(u)

    deltas = diff(us)
    normal_cdf(x) = 0.5*(1. + erf((x-mu)/sqrt(2. * sig * sig)))

    u_ecdf = (1:1:N)./N
    u_ecdf = u_ecdf[1:end-1]

    ref_cdf = normal_cdf.(us[1:end-1])
    sum(abs.(u_ecdf .- ref_cdf) .* deltas)
end

# curve entropy according to Balestrino et al, 2009, Entropy Journal
# could be used as an additional measure for the clustering
# bounded [0,1]
#
"""
function curve_entropy(sol::AbstractArray, r_eps::Float64=1e-15)
    D = mcs_diameter(sol)
    if D > r_eps
        ce = log(curve_length(sol)/D)/log(size(sol)[1] - 1)
    else
        ce = 0. # if the curve is just a point, its entropy is 0 (the logarithm would yield NaN)
    end
    ce
end

function curve_length(u::AbstractArray)
    L::Float64 = 0.
    for it=2:size(u)[1]
        L += euclidean(u[it,:],u[it-1,:])
    end
    L
end

# Minimal Covering (Hyper)sphere of the points of the Curve
function mcs_diameter(u::AbstractArray)
    # miniball routine needs a N_t x d matrix
    mcs = miniball(u)
    2. * sqrt(mcs.squared_radius)
end
"""
