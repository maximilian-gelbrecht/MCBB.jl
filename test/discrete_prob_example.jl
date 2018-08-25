using HighBifLib
using DifferentialEquations
using Distributions

# ranges
r = 3:0.05:3.3
pars = logistic_parameters(r[1])
ic_ranges = [0.1:0.3:0.9]
dp = DiscreteProblem(logistic, ic_ranges[1][1], (0.,5000.), pars)
(ic_r_prob, ic_par, N_mc) = setup_ic_par_mc_problem(dp, ic_ranges, pars, (:r, r))
log_mcp = MonteCarloProblem(dp, prob_func=ic_r_prob, output_func=eval_ode_run)
tail_frac = 0.8
log_emcp = EqMCProblem(log_mcp, N_mc, tail_frac)
log_sol = solve(log_emcp)

# random+range
r =  3:0.05:3.3
pars = logistic_parameters(r[1])
icdist = Uniform(0.1,0.9)
ic_ranges = [()->rand(icdist)]
N_ic = 20
dp = DiscreteProblem(logistic, ic_ranges[1][1], (0.,5000.), pars)
(ic_r_prob, ic_par, N_mc) = setup_ic_par_mc_problem(dp, ic_ranges, N_ic, pars, (:r, r))
log_mcp = MonteCarloProblem(dp, prob_func=ic_r_prob, output_func=eval_ode_run)
tail_frac = 0.8
log_emcp = EqMCProblem(log_mcp, N_mc, tail_frac)
log_sol = solve(log_emcp)

# random+random
rdist = Uniform(3,3.3)
r = ()->rand(rdist)
pars = logistic_parameters(r[1])
icdist = Uniform(0.1,0.9)
ic_ranges = [()->rand(icdist)]
N_ic = 20
dp = DiscreteProblem(logistic, ic_ranges[1][1], (0.,5000.), pars)
(ic_r_prob, ic_par, N_mc) = setup_ic_par_mc_problem(dp, ic_ranges, N_ic, pars, (:r, r))
log_mcp = MonteCarloProblem(dp, prob_func=ic_r_prob, output_func=eval_ode_run)
tail_frac = 0.8
log_emcp = EqMCProblem(log_mcp, N_mc, tail_frac)
log_sol = solve(log_emcp)
