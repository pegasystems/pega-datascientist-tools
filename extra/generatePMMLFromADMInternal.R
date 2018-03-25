# Read ADM factory JSON and turn into PMML

library(tidyverse)
library(data.table)
library(RJDBC)
library(jsonlite)

# get directory of current script, see https://stackoverflow.com/questions/1815606/rscript-determine-path-of-the-executing-script
this.file <- sys.frame(tail(grep('source',sys.calls()),n=1))$ofile
this.dir <- dirname(this.file)

source(paste(this.dir,"..","utils", "pmdsm_utils.R",sep="/"))
source(paste(this.dir,"exportUtilsJSON.R",sep="/"))
source(paste(this.dir,"adm2pmml.R",sep="/"))

dest.dir <- paste(this.dir, "generated_pmml", sep="/")
tmp.dir <- "tmp"
if (!dir.exists(tmp.dir)) { dir.create(tmp.dir) }

# Change the driver path, host, db name, username and password to run on different environments.

drv <- JDBC("org.postgresql.Driver", "~/Documents/tools/jdbc-drivers/postgresql-9.2-1002.jdbc4.jar")
pg_host <- "localhost:5432"
pg_db <- "pega731"
pg_user <- "pega"
pg_pwd <- "pega"

conn <- dbConnect(drv, paste("jdbc:postgresql://", pg_host,  "/", pg_db, sep=""), pg_user, pg_pwd)
modelFactory <- getModelsFromJSONTable(conn)
dbDisconnect(conn)

# each "config partition" is an ADM model rule

# returns a list of binning, context pairs - we just want to pass this list to the PMML exporter
for (p in unique(modelFactory$pyconfigpartitionid)) {
  # get all model "partitions" for this model rule
  modelPartitions <- modelFactory[pyconfigpartitionid==p]
  modelPartitionInfo <- fromJSON(modelPartitions[1]$pyconfigpartition)
  modelPartitionName <- modelPartitionInfo$partition$pyPurpose
  modelPartitionFullName <- paste(modelPartitionInfo$partition$pyClassName, modelPartitionInfo$partition$pyPurpose, sep="_")

  cat("Processing", modelPartitionFullName, fill=T)
  pmmlFileName <- paste(dest.dir, paste(modelPartitionName, "json", "pmml", "xml", sep="."), sep="/")

  modelList <- createListFromADMFactory(modelPartitions, modelPartitionFullName, tmp.dir)
  pmml <- createPMML(modelList, modelPartitionFullName)

  sink(pmmlFileName)
  print(pmml)
  sink()

  cat(paste("Generated",pmmlFileName,paste("(",nrow(modelPartitions)," models)", sep="")), fill = T)

}
