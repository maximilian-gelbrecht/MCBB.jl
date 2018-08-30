Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

As of now this is only tested with Julia 0.6.2, but will be probably ported and used with Julia 1.0 as soon as all Libraries used work with it.

# Install

* Pkg.clone("git@gitlab.pik-potsdam.de:maxgelbr/HighBifLib.jl.git")
* Pkg.test("HighBifLib") to test the installation

# Updates

* Pkg.update("HighBifLib")

# To-Do

* new KL divergence needed
* introduce eval_ode_run with additional input function argument
  * test various other divergences like Wasserstein divergence or different estimates of KL divergence
* write better Readme and start Doc
