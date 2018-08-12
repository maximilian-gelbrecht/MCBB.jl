Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

As of now this is only tested with Julia 0.6.2, but will be probably ported and used with Julia 1.0 as soons as all Libraries used work with it.

* HighBifLib.jl: Library file containing all relevant functions
* run_mc.ipynb: Jupyter notebook with examples
* nbconv.tpl: Template heling to convert the jupyter notebook to a regular Julia script for usage on a hpc
* run_mc.jl: Converted notebook (not meant for direct editing)
* submit.sh: SLURM submit file
* test_sub_functions.ipynb: Test of some functions (e.g. KL-divergence)
