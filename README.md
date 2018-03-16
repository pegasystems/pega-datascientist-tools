# Customer Decision Hub Data Scientist Tools

Utilities, tools and demo scripts for data scientists to work with Pega DSM/CDH.

Currently only supporting an R package with several utiliites. Python utils and demos will follow.

## R

There are two way to use the R scripts. The first option is to use it like you use any package: install it from GitHub, browse the vignettes and copy/paste the code snippets of interest.

To install the package use the devtools package. If you don't have that yet, do that first:

```r
install.packages("devtools")
```

Then load the devtools library and install the cdh package

```r
library(devtools)
install_github("pegasystems/cdh-datascientist-tools/r")
```

The other option is to download the source (clone from the GitHub repository) and use the functions and demo scripts directly.

Both are supported. We release this under the Apache 2.0 license and welcome contributing back, preferably through pull requests, but just submitting an Issue or sending a note to the authors is fine too.

## Python

No Python libraries or demo scripts available yet.

