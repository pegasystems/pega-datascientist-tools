#!/bin/bash

# Example of how to create a single python based off-line model report from the command line.

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
offlinemodelreport_notebook_python="${pdstools_root_folder}/python/pdstools/reports/ModelReport.qmd"

# The ID of model that you are interested in. You can find this model ID in
# Pega Prediction Studio or select another model ID from your own data.
modelid="05085857-c52c-5daa-b51b-0af576a038af"

# Generated file
output="$(pwd)/ADM Model Report.html"

quarto render "$offlinemodelreport_notebook_python" -P "title:ADM Single Model Report" -P "subtitle:Synthetic Demo Data" -P "model_id:${modelid}"

# The generated file will be in the same folder as the notebook. Copy it here.
cp "${pdstools_root_folder}/python/pdstools/reports/ModelReport.html" "$output"

echo "Created model report: ${output}"
