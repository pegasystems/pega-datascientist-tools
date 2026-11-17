#!/bin/bash

# Example of how to run the Python based ADM Health Check from the command line.

# Requirements:
#
# - PDSTools python library installed
#   See https://github.com/pegasystems/pega-datascientist-tools#getting-started
#
# - A clone of the pega-datascientist-tools repository. We use its report
#   template to generate the report.
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

# Generated file
output="$(pwd)/Sample Adaptive Models Overview.html"
quarto render "$healthcheck_notebook_python" -P "title:ADM Health Check" -P "subtitle:Synthetic Demo Data"

# The generated file will be in the same folder as the notebook. Copy it here.
cp "${pdstools_root_folder}/python/pdstools/reports/HealthCheck.html" "$output"

echo "Created ADM health check: ${output}"
