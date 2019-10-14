Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

It uses Julia 1.x (and is not backwards compatible to Julia < v0.7)

# NEWS

Moved to GitHub. Some things still need to be updated accordingly.

# Paper

A paper describing the article can found in the arXiv. If you use this package, please always cite this paper.

# Documentation

The documentation can be found at

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

The package will be added to the Julia registry soon.

# Updates

```
pkg> update MCBB
```
