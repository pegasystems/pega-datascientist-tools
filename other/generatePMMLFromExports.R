# Export ADM Datamart models to PMML from a dataset export

library(tidyverse)
library(data.table)
library(lubridate)

# get directory of current script, see https://stackoverflow.com/questions/1815606/rscript-determine-path-of-the-executing-script
this.file <- sys.frame(tail(grep('source',sys.calls()),n=1))$ofile
this.dir <- dirname(this.file)

source(paste(this.dir,"..","utils","pmdsm_utils.R",sep="/"))
source(paste(this.dir,"exportUtilsDatamart.R",sep="/"))
source(paste(this.dir,"adm2pmml.R",sep="/"))

dest.dir <- paste(this.dir, "generated_pmml", sep="/")
tmp.dir <- "tmp"
if (!dir.exists(tmp.dir)) { dir.create(tmp.dir) }

# In Pega, export the data from Data-Decision-ADM-ModelSnapshot through a data set (in DA utils/PMDSM there is an AllModelSnapshots
# dataset already, otherwise create a DB dataset yourself). Run the actions --> export and download the generated file. Do the same
# for Data-Decision-ADM-PredictorBinningSnapshot (a default AllPredictorSnapshots dataset is available).

models <- readDSExport("Data-Decision-ADM-ModelSnapshot", srcFolder="~/Downloads", tmpFolder="tmp")
predictors <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot", srcFolder="~/Downloads", tmpFolder="tmp")

# Iterate over unique model rules (model configurations). Every single one of these results in a seperate PMML file.

modelRules <- unique(select(models, pyConfigurationName, pyAppliesToClass, pxApplication))
if(nrow(modelRules)>0) {
  for (m in seq(nrow(modelRules))) {
    modelPartitionName <- modelRules$pyConfigurationName[m]
    
    modelPartitionFullName <- paste(modelRules$pxApplication[m], modelRules$pyAppliesToClass[m], modelRules$pyConfigurationName[m], sep="_")  # get all model "partitions" for this model rule
    pmmlFileName <- paste(dest.dir, paste(modelRules$pyConfigurationName[m], "ds", "pmml", "xml", sep="."), sep="/")
    
    cat("Processing", modelPartitionFullName, fill=T)
    
    # Each of the model instances is an instantiation of a model rule for a specific set of context key values
    modelInstances <- unique(models[pyConfigurationName==modelRules$pyConfigurationName[m] &
                                      pyAppliesToClass==modelRules$pyAppliesToClass[m] &
                                      pxApplication==modelRules$pxApplication[m]])

    modelList <- createListFromDatamart(predictors[which(pyModelID %in% modelInstances$pyModelID),], modelPartitionFullName, tmp.dir, modelsForPartition=modelInstances)
    
    pmml <- createPMML(modelList, modelPartitionFullName)
    
    sink(pmmlFileName)
    print(pmml)
    sink()
    
    cat(paste("Generated",pmmlFileName,paste("(",length(modelList)," models)", sep="")), fill = T)
  }
} else {
  cat("No models available", fill=T)
}
