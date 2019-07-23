#!/bin/bash

# Run an R notebook on the given inputfile with model data from the ADM datamart

# Source needs to be a CSV file from the model table

notebook="./adaptivemodeloverview.Rmd"
#source="../extra/pr_data_dm_admmart_pred.csv"
source="~/Box Sync/Customers/RABO/datamart july 2019/PR_DATA_DM_ADMMART_MDL_FACT.csv.zip"
modelconfig="pipodeclown"
output="`pwd`/$modelconfig.html"

R -e "rmarkdown::render('$notebook',params = list(modelfile='$source', modelconfig='$modelconfig'), output_file='$output')"

