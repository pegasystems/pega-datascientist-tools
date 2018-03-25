# Export ADM Datamart models to PMML directly from DB

library(tidyverse)
library(data.table)
library(XML)
library(lubridate)
library(RJDBC)

# get directory of current script, see https://stackoverflow.com/questions/1815606/rscript-determine-path-of-the-executing-script
this.file <- sys.frame(tail(grep('source',sys.calls()),n=1))$ofile
this.dir <- dirname(this.file)

source(paste(this.dir,"..","utils","pmdsm_utils.R",sep="/"))
source(paste(this.dir,"exportUtilsDatamart.R",sep="/"))
source(paste(this.dir,"adm2pmml.R",sep="/"))

dest.dir <- paste(this.dir, "generated_pmml", sep="/")
tmp.dir <- "tmp"
if (!dir.exists(tmp.dir)) { dir.create(tmp.dir) }

# Read data from the ADM Datamart. 
# Change the driver path, host, db name, username and password to run on different environments.

drv <- JDBC("org.postgresql.Driver", "~/Documents/tools/jdbc-drivers/postgresql-9.2-1002.jdbc4.jar")
pg_host <- "localhost:5432"
pg_db <- "pega731"
pg_user <- "pega"
pg_pwd <- "pega"

conn <- dbConnect(drv, paste("jdbc:postgresql://", pg_host,  "/", pg_db, sep=""), pg_user, pg_pwd)

models <- getModelsForClassFromDatamart(conn, "Exam-PMDSM-Data-Customer")
predictors <- getPredictorsForModelsFromDatamart(conn, models)

dbDisconnect(conn)

# Iterate over unique model rules (model configurations). Every single one of these results in a seperate PMML file.
modelRules <- unique(select(models, pyconfigurationname, pyappliestoclass, pxapplication))
if(nrow(modelRules)>0) {
  for (m in seq(nrow(modelRules))) {
    modelName <- paste(modelRules$pxapplication[m], modelRules$pyappliestoclass[m], modelRules$pyconfigurationname[m], sep="_")
    cat("Processing", modelName, fill=T)
    pmmlFileName <- paste(dest.dir, paste(modelRules$pyconfigurationname[m], "dm", "xml", sep="."), sep="/")
    
    # Each of the model instances is an instantiation of a model rule for a specific set of context key values
    modelInstances <- unique(models[pyconfigurationname==modelRules$pyconfigurationname[m] &
                                      pyappliestoclass==modelRules$pyappliestoclass[m] &
                                      pxapplication==modelRules$pxapplication[m]])
    modelList <- createListFromDatamart(predictors[pymodelid %in% modelInstances$pymodelid], modelName, tmp.dir, modelsForPartition=modelInstances)
    
    pmml <- createPMML(modelList, modelName)

    sink(pmmlFileName)
    print(pmml)
    sink()
    
    cat(paste("Generated",pmmlFileName,paste("(",nrow(modelInstances)," models)", sep="")), fill = T)
  }
}
