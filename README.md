Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

It uses Julia 1.0 (and is not backwards compatible to Julia < v0.7)

# NEWS

The name of the project has changed from `HighBifLib` to `MCBB`. In order to transition smoothly between the old and the new version, it is easiest to deinstall the old version first

```
pkg> remove HighBifLib
```

and then install the new one with the instructions below.

Additionally, there have been other breaking changes:
* `BifAnaMCProblem` is now called `DEMCBBProblem`
* `CustomMCBBProblem` introduced for systems that don't use DifferentialEquations.jl as a backend
* `eval_ode_run` does not have `mean_and_std` as a default evaluation function anymore.

Check out the documentation for detailed info.

# Install

In `Pkg` shell doing:
```
pkg> add git@gitlab.pik-potsdam.de:maxgelbr/MCBB.git
```
should work, if you encounter any problem updating all packages with
```
pkg> up
```
can help, otherwise (and if you want to contribute) you can manually clone it:
```
$ git clone git@gitlab.pik-potsdam.de:maxgelbr/MCBB.git
$ cd MCBB
pkg> dev /path/to/package
```
The `dev` will make Julia load the module always from the local files.

Then, you can test the installation with
```
pkg> test MCBB
```

# Updates

```
pkg> update MCBB
```

# Built the documentation

There's a nice documentation that's not yet hosted (due to GitLab pages not being available at PIK-Gitlab). You can build it yourself, though! Assuming you downloaded the package (with e.g. `git clone ...`):

```
$ /path/to/julia1.0/binary /path/to/package/MCBB/docs/make.jl
```
The documentation then gets built in `/path/to/package/MCBB/docs/build`. You need a local webserver for hyperlinks to work:
```
$ cd /path/to/package/MCBB/docs/build/
$ python3 -m http.server --bind localhost
```
Afterwards in your browser go to `localhost:8000`.

# TO-DO

* Plot Recipes: So far, there are no plot recipes and all examples are for PyPlot. This is suboptimal and for a proper release, they should be plot recipes to make the plotting simpler 
