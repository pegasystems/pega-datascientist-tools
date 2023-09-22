# Example R script to create off-line model reports. There is a similar
# but much less complex bash script to do the same. Any language can really
# be used here - the fact that the off-line model reports are in an R notebook
# does not mean the preprocessing and batch processing needs to be in R as well.

# You can run this script from R, from R Studio, VS Code or really any editor
# of your choice. You will need to change some of the paths defined here.

library(pdstools)
library(data.table)
library(rmarkdown)
library(arrow)

# Pandoc is needed by RMarkdown and part of RStudio. If you run this
# script outside of RStudio you'll need to make sure pandoc is installed
# and known to R / markdown. For now this is the best I could think of. To
# make it slightly more generic, dir can be a character vector of paths:
if (!rmarkdown::pandoc_available()) {
  rmarkdown::find_pandoc(dir = c("/opt/anaconda3/bin"))
  cat("Pandoc:", rmarkdown::pandoc_available(), fill = T)
}

customer <- "SampleCustomer" # just for titles, change to your customer name
datamart_datasets_folder <- "~/Downloads" # will pick the latest from there

# Path to the checked out versions of the notebooks. You'll need them locally
# so make sure to to a "clone" of the PDS Tools GitHub repository at
# https://github.com/pegasystems/pega-datascientist-tools. Update the path
# below to reflect the folder where you cloned the repo.

pdstools_repo_folder <- "~/Documents/pega/pega-datascientist-tools"

healthcheck_notebook_R <- file.path(pdstools_repo_folder, "examples/datamart/healthcheck.Rmd")
offlinemodelreport_notebook_R <- file.path(pdstools_repo_folder, "examples/datamart/modelreport.Rmd")

working_folder <- tempdir(TRUE)
output_folder <- file.path(getwd(), "reports")
if (!dir.exists(output_folder)) dir.create(output_folder)

# Read ADM Datamart from the folder specified above. You can also give
# explicit paths to both dataset. See help on ADMDatamart, in R Studio
# with ?pdstools::ADMDatamart or online at
# https://pegasystems.github.io/pega-datascientist-tools/R/reference/ADMDatamart.html

# Example of a function you can implement to highlight certain 
# types of predictors based on their names. By default the system will
# highlight IH.* and Param.* predictors - simply splitting on the first dot,
# but you can customize this as shown below:

myPredictorCategorization <- function(name)
{
  if (startsWith(name, "Param.ExtGroup")) return("External Model")
  if (endsWith(name, "Score")) return("External Model")
  if (endsWith(name, "RiskCode")) return("External Model")
  
  return(defaultPredictorCategorization(name))
}

dm <- ADMDatamart(datamart_datasets_folder, 
                  # optional predictor categorization, see above
                  predictorCategorization = myPredictorCategorization,
                  
                  # filtering the data to be used
                  filterModelData = function(mdls) {
                    return(mdls[ConfigurationName %in% c("OmniAdaptiveModel") & 
                                  Group == "CreditCards" & 
                                  Direction == "Outbound"])
                  })

# Write back temp files with the filtered data - not strictly necessary, you
# can also refer to the full files in the call to the notebooks.

tempModelFile <- tempfile(fileext = "_mdls.arrow", tmpdir = working_folder)
arrow::write_ipc_file(dm$modeldata, sink = tempModelFile)
tempPredictorFile <- tempfile(fileext = "_preds.arrow", tmpdir = working_folder)
arrow::write_ipc_file(dm$predictordata, sink = tempPredictorFile)

# Create Health Check (legacy R version - now superseded by the new Python version)

rmarkdown::render(healthcheck_notebook_R,
                  params = list(
                    modelfile = tempModelFile,
                    predictordatafile = tempPredictorFile,
                    title = paste("ADM Health Check", customer, sep = " - "),
                    subtitle = "legacy R version"
                  ),
                  output_dir = working_folder,
                  output_file = paste("ADM Health Check ", customer, ".html", sep = ""),
                  quiet = FALSE, intermediates_dir = working_folder
)

# Individual Model reports

# In real life situations you probably want to select a subset of the 
# models, not run a model report for every possible ADM instance, which
# would typically be in the 100's or 1000's.

# Below we select 5 of the models from every channel with the largest
# response counts. This is just a simple example that can easily be
# extended.

recentModels <- filterLatestSnapshotOnly(dm$modeldata)[Positives > 10]
recentModels[, PosRank := frank(-Positives, ties.method="random"), by=c("Direction", "Channel", "ConfigurationName")]
ids <- recentModels[PosRank <= 5, "ModelID"]

# Associate a name with the model IDs
modelNames <- sapply(ids, function(id) {
  make.names(paste(
    sapply(unique(dm$modeldata[
      ModelID == id,
      c("ConfigurationName", "Channel", "Direction", "Issue", "Group", "Name", "Treatment")
    ]), as.character),
    collapse = "_"
  ))
})

# Create a report for every of these models
for (n in seq_along(ids)) {
  id <- ids[order(modelNames)][n]
  modelName <- modelNames[id]
  
  cat("Model:", modelName, n, "of", length(ids), fill = T)
  
  localModelReportHTMLFile <- paste0(customer, "_", modelName, ".html")
  
  rmarkdown::render(offlinemodelreport_notebook_R,
                    params = list(
                      predictordatafile = tempPredictorFile,
                      modeldescription = modelName,
                      modelid = id
                    ),
                    output_dir = output_folder,
                    output_file = localModelReportHTMLFile,
                    quiet = F, intermediates_dir = working_folder
  )
}

cat("Done. Output is in", output_folder, fill=T)


