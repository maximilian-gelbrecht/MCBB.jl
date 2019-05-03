Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

It uses Julia 1.0/1.1 (and is not backwards compatible to Julia < v0.7)

# NEWS

Most plots and cluster evaluation function have been rewritten to include plot recipes for easier plotting.

Check out the documentation for detailed info, at `?cluster_membership`, `?MCBB.ClusterMembershipResult`, `?cluster_measures_sliding_histograms`, `?MCBB.ClusterMeasuresHistogramResult`, `?cluster_measures`, `?MCBB.ClusterMeasureResult`. 

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

* Plot Recipes: So far, there are not yet plot recipes for everything
