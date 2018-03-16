#' cdhtools: Data Science add-ons for Pega.
#'
#' Various utilities to access and manipulate data from Pega for purposes
#' of data analysis, reporting and monitoring.
#'
#' @section Style:
#'
#' The utilities in cdhtools heavily rely on \code{data.table} and often return values of this type. \code{data.table}
#' provides a high-performance version of base R's \code{data.frame}, see \url{https://github.com/Rdatatable/data.table/wiki}.
#' The transition is not complete however, so in parts of the code we also still use the \code{tidyverse} style with
#' processing pipelines based on \code{tidyr} and \code{dplyr} etc.
#'
#' @docType package
#' @name cdhtools
#' @import data.table
#' @importFrom rlang is_list
NULL

#' Read a Pega dataset export file.
#'
#' \code{readDSExport} reads a dataset export file as exported and downloaded from Pega. The file
#' is formatted as a zipped mulit-line JSON file (\url{http://jsonlines.org/}). \code{readDSExport} will find the
#' most recent file (Pega appends a datetime stamp), unzip it into a temp folder
#' and read the data into a \code{data.table}.
#'
#' @param instancename Name of the file w/o the timestamp, in Pega format <Applies To>_<Instance Name>
#' @param srcFolder Optional folder to look for the file (defaults to the current folder)
#' @param tmpFolder Optional folder to store the unzipped data (defaults to the source folder)
#'
#' @return A \code{data.table} with the contents
#' @export
#'
#' @examples
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots")}
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots", "~/Downloads")}
readDSExport <- function(instancename, srcFolder=".", tmpFolder=srcFolder)
{
  mostRecentZip <- rev(sort(list.files(path=srcFolder, pattern=paste("^", instancename, "_.*\\.zip$", sep=""))))[1]
  utils::unzip(paste(srcFolder,mostRecentZip,sep="/"), exdir=tmpFolder)
  multiLineJSON <- readLines(paste(tmpFolder,"data.json", sep="/"))
  ds <- data.table(jsonlite::fromJSON(paste("[",paste(multiLineJSON,sep="",collapse = ","),"]")))
  return(ds [, names(ds)[!sapply(ds, is_list)], with=F])
}

#' Convert AUC performance metric to GINI.
#'
#' @param auc The AUC (number between 0.5 and 1)
#'
#' @return GINI metric, a number beteen 0 and 1
#' @export
#'
#' @examples
#' auc2GINI(0.8232)
auc2GINI <- function(auc)
{
  return (2*auc-1)
}

