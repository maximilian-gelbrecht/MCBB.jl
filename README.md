Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

As of now this is only tested with Julia 0.6.2, but will be probably ported and used with Julia 1.0 as soon as all Libraries used work with it.

# NEWS

* now ported to Julia 1.0

# Install

* Pkg.clone("git@gitlab.pik-potsdam.de:maxgelbr/HighBifLib.jl.git")
* Pkg.test("HighBifLib") to test the installation

# Updates

* Pkg.update("HighBifLib")

# Development Notes

* It is possible to checkout/use branches with Julias Pkg as well, but have not tried this yet. So far I only used the master branch
* Pkg.update/clone(..., fetch_all=true) and Pkg.checkout("HighBifLib","branch-name") should do the job (I hope so)

# To-Do

* more than 1 parameter dimension
* update the Readme  
