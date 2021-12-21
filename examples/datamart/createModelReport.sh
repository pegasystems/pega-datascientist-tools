#!/bin/bash

# Run an R notebook on the given inputfile with predictor binning data

# Location of the GIT checkout of the CDH tools
cdhtools="~/Documents/pega/cdh-datascientist-tools"
modelreportnotebook="$cdhtools/examples/datamart/modelreport.Rmd"

# Predictor data. This can be a CSV or any other of the supported formats.
source="$cdhtools/data/pr_data_dm_admmart_pred.csv"

# Model ID to use
modelid="277a2c48-8888-5b71-911e-443d52c9b50f"
modeldescription="Banner Model - BMOBILEAPP"

# Generated file
output="`pwd`/$modeldescription.html"

R -e "rmarkdown::render('$modelreportnotebook',params = list(predictordatafile='$source', modeldescription='$modeldescription', modelid='$modelid'), output_file='$output')"

