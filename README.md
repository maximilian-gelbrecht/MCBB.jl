Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

As of now this is only tested with Julia 0.6.2, but will be probably ported and used with Julia 1.0 as soon as all Libraries used work with it.

# Install

* Pkg.clone("git@gitlab.pik-potsdam.de:maxgelbr/HighBifLib.jl.git")
* Pkg.test("HighBifLib") to test the installation

# Updates

* Pkg.update("HighBifLib")

# Development Notes

* It is possible to checkout/use branches with Julias Pkg as well, but have not tried this yet. So far I only used the master branch
* Pkg.update/clone(..., fetch_all=true) and Pkg.checkout("HighBifLib","branch-name") should do the job (I hope so)

# To-Do

* new KL divergence needed?
* there is now a more flexible eval_ode_run that can use all kinds of metrics and measures, e.g  1-wasserstein metric is implemented as well. Tests from HypothesisTests.jl can also easily be used. Example see LogisticMap example from run_mc.ipynb
  * lets play around with it!
* write better Readme and start Doc
