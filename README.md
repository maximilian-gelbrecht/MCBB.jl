Library and Script with Example to explore High Dimensional Bifurcations with sample and clustering based methods using Julia language.

It uses Julia 1.0 (and is not backwards compatible to Julia < v0.7)

# Install

In `Pkg` shell doing:
```
pkg> add git@gitlab.pik-potsdam.de:maxgelbr/HighBifLib.git
```
should work, if you encounter any problem updating all packages with
```
pkg> up
```
can help, otherwise (and if you want to contribute) you can manually clone it:
```
$ git clone git@gitlab.pik-potsdam.de:maxgelbr/HighBifLib.git
$ cd HighBifLib
pkg> dev /path/to/package
```
The `dev` will make Julia load the module always from the local files.

Then, you can test the installation with
```
pkg> test HighBifLib
```

# Updates

```
pkg> update HighBifLib
```

# Built the documentation

There's a nice documentation that's not yet hosted (due to GitLab pages not being available at PIK-Gitlab). You can build it yourself, though! Assuming you downloaded the package (with e.g. `git clone ...`):

```
$ /path/to/julia1.0/binary /path/to/package/HighBifLib/docs/make.jl
```
The documentation then gets built in `/path/to/package/HighBifLib/docs/build`. You need a local webserver for hyperlinks to work:
```
$ cd /path/to/package/HighBifLib/docs/build/
$ python3 -m http.server --bind localhost
```
Afterwards in your browser go to `localhost:8000`.
