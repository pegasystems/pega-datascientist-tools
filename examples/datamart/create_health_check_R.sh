#!/bin/bash

# Example of how to run the R based ADM Health Check from the command line.

# Requirements:
#
# - PDSTools R library installed
#   See https://github.com/pegasystems/pega-datascientist-tools#getting-started
#
# - A clone of the pega-datascientist-tools repository. We need this in addition
#   to the library because we directly access sample files and notebook files
#   from the repository
#   https://github.com/pegasystems/pega-datascientist-tools
#
# - Pandoc (installed automatically when you run this script from RStudio)
#   https://pandoc.org/

# Folder with a clone of the PDS tools pega-datascientist-tools repository.
pdstools_root_folder="~/Documents/pega/pega-datascientist-tools"
healthcheck_notebook_R="${pdstools_root_folder}/examples/datamart/healthcheck.Rmd"

# ADM datamart files exported from Pega. Swap these for files exported from
# your own project. See PDS Tools Wiki for guidance on the export steps.
modeldata="${pdstools_root_folder}/data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
predictordata="${pdstools_root_folder}/data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"

# Generated file
output="`pwd`/Sample Adaptive Models Overview.html"

R -e "rmarkdown::render('${healthcheck_notebook_R}', params = list(modelfile='${modeldata}', predictordatafile='${predictordata}', title='ADM Health Check', subtitle='Sample Customer'), output_file='${output}')"

echo "Created ADM health check: ${output}"
