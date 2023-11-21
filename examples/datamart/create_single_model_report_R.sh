#!/bin/bash

# Example of how to create a single R based off-line model report from the command line.

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
offlinemodelreport_notebook_R="${pdstools_root_folder}/examples/datamart/modelreport.Rmd"

# ADM datamart files exported from Pega. Swap these for files exported from
# your own project. See PDS Tools Wiki for guidance on the export steps.
modeldata="${pdstools_root_folder}/data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
predictordata="${pdstools_root_folder}/data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"

# The ID of model that you are interested in. You can find this model ID in
# Pega Prediction studio or load the model data in Python or R and select the
# model ID(s) based on criteria like issue, group, number of positives etc.
modelid="c3f445b2-6037-5560-9f16-71a7259d2c2c"

# Generated file
output="`pwd`/ADM Model Report.html"

R -e "rmarkdown::render('${offlinemodelreport_notebook_R}', params = list(predictordatafile='${predictordata}', title='ADM Single Model Report', modelid='${modelid}', modeldescription='Sample Model'), output_file='${output}')"

echo "Created model report: ${output}"

