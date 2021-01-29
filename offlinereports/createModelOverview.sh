#!/bin/bash

# Run an R notebook on the given inputfile with model data from the ADM datamart

# Source needs to be a CSV file from the model table

notebook="./adaptivemodeloverview.Rmd"
source="../data/pr_data_dm_admmart_mdl_fact.csv"
output="`pwd`/DMSample\ Adaptive\ Models\ Overview.html"

R -e "rmarkdown::render('$notebook',params = list(modelfile='$source'), output_file='$output')"

