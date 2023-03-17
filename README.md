# NEWS

The method is now implemented in [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl), i.e. its sub-libary [Attractors.jl](https://juliadynamics.github.io/Attractors.jl/dev/). See its [docs for more information](https://juliadynamics.github.io/Attractors.jl/dev/examples/#Featurizing-and-grouping-across-parameters-(MCBB)). This repository will thus not be maintained or updated and only serve as an archive


# Package Archive: 

Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

It uses Julia 1.x (CI/CD test with Julia 1.3).
It is not backwards compatible to Julia < v0.7

[![Build Status](https://travis-ci.com/maximilian-gelbrecht/MCBB.jl.svg?branch=master)](https://travis-ci.com/maximilian-gelbrecht/MCBB.jl/branches)
[![doc dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maximilian-gelbrecht.github.io/MCBB.jl/dev)



## Paper

A paper describing the article was is published in New Journal of Physics (Open Access) http://doi.org/10.1088/1367-2630/ab7a05

## Documentation

The documentation can be found at [![doc dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://maximilian-gelbrecht.github.io/MCBB.jl/dev)


## Install

In `Pkg` shell doing:
```
pkg> add https://github.com/maximilian-gelbrecht/MCBB.jl.git

```
should work, if you encounter any problem updating all packages with
```
pkg> up
```
can help, otherwise (and if you want to contribute) you can manually clone it:
```
$ git clone https://github.com/maximilian-gelbrecht/MCBB.jl.git
$ cd MCBB
pkg> dev /path/to/package
```
The `dev` will make Julia load the module always from the local files.

Then, you can test the installation with
```
pkg> test MCBB
```

The package will be added to the Julia registry soon.

# Updates

```
pkg> update MCBB
```
