#!/bin/bash

# Run an R notebook on the given inputfile with predictor binning data

# Source needs to be a CSV file from the predictor binning table with the details of one particular model
# or the modelid needs to be passed in. The model description is optional and gives a title to the model
# as this info is not present in the predictor binning data

notebook="./modelreport.Rmd"
source="../extra/pr_data_dm_admmart_pred.csv"
modelid="7bf31ac7-8562-522f-8f95-7d35e3c7a96f"
modeldescription="Banner Model - BMOBILEAPP"
output="`pwd`/$modeldescription.html"

R -e "rmarkdown::render('$notebook',params = list(predictordatafile='$source', modeldescription='$modeldescription', modelid='$modelid'), output_file='$output')"

