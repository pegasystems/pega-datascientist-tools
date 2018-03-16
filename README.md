# Customer Decision Hub Data Scientist Tools

Utilities, tools and demo scripts for data scientists to work with Pega DSM/CDH.

Currently only supporting an R package with several utiliites. Python utils and demos will follow.

## R

[![Build Status](https://travis-ci.org/pegasystems/cdh-datascientist-tools.svg?branch=master)](https://travis-ci.org/pegasystems/cdh-datascientist-tools)
[![codecov](https://codecov.io/gh/pegasystems/cdh-datascientist-tools/branch/master/graph/badge.svg)](https://codecov.io/gh/pegasystems/cdh-datascientist-tools)

There are two way to use the R scripts. The first option is to use it like you use any package: install it from GitHub, browse the vignettes and copy/paste the code snippets of interest.

To install the package use the **devtools** package. If you don't have that installed yet, do that first:

```r
install.packages("devtools")
```

Then load the **devtools** library and install the **cdhtools** package. Note the `build_vignettes` flag. By default packages built from GitHub do not include the vignettes, but these are essential for these demo scripts, so you should include them.

```r
library(devtools)
install_github("pegasystems/cdh-datascientist-tools/r", build_vignettes=TRUE)
```

If all is well, this will then install an R package called **cdhtools** that you can then use just like any other R package.

We use vignettes as the primary vehicle for demo scripts that show how to make use of the package. The source of the vignettes itself is typically useful as this can be customized to specific needs and situations.

For those less familiar with R vignettes: you can get the list of vignettes with `browseVignettes("cdhtools")` (as a web page) or `vignette(package="cdhtools")`. A vignette provides the original source as well as a readable HTML or PDF page and a file with the R code. Read a specific one with `vignette(x)` and see its code with `edit(vignette(x))`.

The other option is to download the source (clone from the GitHub repository) and use the functions and demo scripts directly. Just clone the repository and explore the package contents. The R code, tests, vignettes etc are in the **r** subdirectory.

Both are supported. We release this under the Apache 2.0 license and welcome contributing back, preferably through pull requests, but just submitting an Issue or sending a note to the authors is fine too.

## Python

No Python libraries or demo scripts available yet.

