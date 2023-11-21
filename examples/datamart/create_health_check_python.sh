#!/bin/bash

# Example of how to run the Python based ADM Health Check from the command line.

# Requirements:
#
# - PDSTools python library installed
#   See https://github.com/pegasystems/pega-datascientist-tools#getting-started
#
# - A clone of the pega-datascientist-tools repository. We need this in addition
#   to the library because we directly access sample files and notebook files
#   from the repository
#   https://github.com/pegasystems/pega-datascientist-tools
#
# - Quarto (this is the notebook format we use)
#   https://quarto.org/
#
# - Pandoc
#   https://pandoc.org/

# Folder with a clone of the PDS tools pega-datascientist-tools repository.
pdstools_root_folder="$HOME/Documents/pega/pega-datascientist-tools"
healthcheck_notebook_python="${pdstools_root_folder}/python/pdstools/reports/HealthCheck.qmd"

# ADM datamart files exported from Pega. Swap these for files exported from
# your own project. See PDS Tools Wiki for guidance on the export steps.
modeldata="${pdstools_root_folder}/data/Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip"
predictordata="${pdstools_root_folder}/data/Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20210526T133622_GMT.zip"

# Generated file
output="`pwd`/Sample Adaptive Models Overview.html"
quarto render ${healthcheck_notebook_python} -P title:'ADM Health Check' -P subtitle:'Sample Customer' -P datafolder:"$(dirname -- ${modeldata})" -P modelfilename:"$(basename -- ${modeldata})"  -P predictorfilename:"$(basename -- ${predictordata})"

# The generated file will be in the same folder as the notebook. Copy it here.
cp "${pdstools_root_folder}/python/pdstools/reports/HealthCheck.html" "${output}"

echo "Created ADM health check: ${output}"

