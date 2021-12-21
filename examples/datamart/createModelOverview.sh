#!/bin/bash

# Run an R notebook on the given inputfile with just the model data. This
# example shows how to run the notebook when there is no predictor data.

# Location of the GIT checkout of the CDH tools
cdhtools="~/Documents/pega/cdh-datascientist-tools"
modeloverviewnotebook="$cdhtools/examples/datamart/healthcheck.Rmd"

# Predictor data. This can be a CSV or any other of the supported formats.
source="$cdhtools/data/pr_data_dm_admmart_mdl_fact.csv"

# Generated file
output="`pwd`/DMSample\ Adaptive\ Models\ Overview.html"

R -e "rmarkdown::render('$modeloverviewnotebook',params = list(modelfile='$source', predictordatafile=''), output_file='$output')"

