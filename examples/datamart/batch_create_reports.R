# Example script to create the Health Check and individual Model Reports
# from ADM datamart data in batch.
#
# This script is in R but the reports are Quarto files (a form of notebook)
# for the newer Python based versions, and in Markdown for the R versions.
#
# The choice of language for this script is arbitrary, it can be ported
# to python or even in just bash.

# Requirements:
#
# - PDSTools library installed: python version if you are planning to run
#   the python version of the reports, the R version if you are planning to
#   run the (older) R versions - or both if you want to compare or are missing
#   some elements that are in one but not yet in the other.
#   See https://github.com/pegasystems/pega-datascientist-tools#getting-started
#
# - A clone of the pega-datascientist-tools repository. We need this in addition
#   to the library because we directly access sample files and notebook files
#   from the repository
#   https://github.com/pegasystems/pega-datascientist-tools
#
# - Quarto (this is the notebook format we use) (only for the python versions)
#   https://quarto.org/
#
# - Pandoc (needed for both)
#   https://pandoc.org/
#
# - R packages, in so far not already installed via PDS tools:
#   data.table, markdown, quarto, arrow, jsonlite, lubridate
#   You can bulk-install them with:
#   install.packages(c("data.table", "markdown", "quarto", "arrow", "jsonlite", "lubridate"))
#   As with PDS tools we recommend only using compiled CRAN versions of the libraries, so not from sources.

customer <- "Sample Customer"

# ADM datamart source files. This is where you save the "zip" datamart export
# files with names like:
#
# Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20231102T152909_GMT.zip
# Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_20231102T153036_GMT.zip
#
# If you repeatedly save new files, you can just add them to the folder, the
# utility will pick up the latest versions automatically. For more control,
# see the doc of the ADMDatamart function in PDSTools.

source_folder <- "~/Documents/pega/pega-datascientist-tools/data"

# Folder with a clone of the PDS tools. You'll need to clone the repo because
# we acces the Quarto markdown files directly.
pdstools_root_folder <- "~/Documents/pega/pega-datascientist-tools"

# Destination folder for the HTML reports. This is where the health check
# and model reports will be copied to. You may also see small ".hash" files
# along with them. These are for internal purposes, used so we don't
# unnecessarily re-generate the HTML reports if not needed.
results_folder <- file.path(getwd(), "reports")

# Folder for temp files and cached files
temp_folder <- file.path(getwd(), "tmp")

# Some simple utilities that help with caching the datamart data, and
# adding a "hash" file to the generated files to prevent unnecessary
# recreation - only will be recreated if the data changes or the quarto
# files get updated.
source(file.path(pdstools_root_folder, "examples/datamart/report_utils.R"))
init_report_utils(pdstool_root_folder = pdstools_root_folder,
                  results_folder = results_folder,
                  intermediates_folder = temp_folder)

# Pandoc is needed by RMarkdown and part of RStudio. If you run this
# script outside of RStudio you'll need to make sure pandoc is installed
# and known to R / markdown. For now this is the best I could think of. To
# make it slightly more generic, dir can be a character vector of paths:
if (!rmarkdown::pandoc_available()) {
  rmarkdown::find_pandoc(dir = c("/opt/anaconda3/bin"))
}
if (!rmarkdown::pandoc_available()) {
  stop("Pandoc not available. Please check your installation.")
}

# ADM Datamart

# Optional, custom, function to tag certain predictors as - for example -
# external model scores. The default categorization just tags predictors
# according to the text before the first dot, for example "Customer", "Account"
# or "IH".
samplePredictorCategorization <- function(name) {

  if (startsWith(name, "Param.Ext")) return("External Model")

  return(defaultPredictorCategorization(name))
}

### Read datamart

dm <- ADMDatamart(
  folder = source_folder
  ,predictorCategorization = samplePredictorCategorization
  #
  # You can optionally filter out models right here, or alternative do it
  # after the reading. Doing it here saves time. See the ADMDatamart function
  # documentation for details.
  # https://pegasystems.github.io/pega-datascientist-tools/R/reference/ADMDatamart.html
  #
  # ,filterModelData = function(m) {
  #   m[ConfigurationName != "OmniAdaptiveModel"]
  # }
)

### Health Check overview
### both R (legacy) and Python (newer) versions created here
### but you can of course choose to comment-out the ones you don't need

run_r_healthcheck(customer, dm)
run_python_healthcheck(customer, dm)

### Individual Model Reports
### both R (legacy) and Python (newer) versions created here
### but you can of course choose to comment-out the ones you don't need

# Optionally select your models of interest
models_of_interest <- unique(filterLatestSnapshotOnly(dm$modeldata[ModelID %in% dm$predictordata$ModelID])[Issue=="Sales" & Positives > 0, ModelID])

run_r_model_reports(customer, dm, modelids = models_of_interest, max = 5)
run_python_model_reports(customer, dm, modelids = models_of_interest, max = 5)


### clean up

report_utils_cleanup_cache()

print("Done!")

