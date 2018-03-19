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
  jsonFile <- paste(tmpFolder,"data.json", sep="/")
  if(file.exists(jsonFile)) file.remove(jsonFile)
  utils::unzip(paste(srcFolder,mostRecentZip,sep="/"), exdir=tmpFolder)
  multiLineJSON <- readLines(jsonFile)
  ds <- data.table(jsonlite::fromJSON(paste("[",paste(multiLineJSON,sep="",collapse = ","),"]")))
  return(ds [, names(ds)[!sapply(ds, is_list)], with=F])
}

#' Calculates AUC from an array of truth values and predictions.
#'
#' Calculates the area under the ROC curve from an array of truth values and predictions, making sure to
#' always return a value between 0.5 and 1.0 and will return 0.5 in case of any issues.
#'
#' @param groundtruth The 'true' values, with just 2 different values, passed on to the pROC::roc function.
#' @param probs The predictions, as a numeric vector as the same length as \code{groundtruth}
#'
#' @return The AUC as a value between 0.5 and 1, return 0.5 if there are any issues with the data.
#' @export
#'
#' @examples
#' safe_auc( c("yes", "yes", "no"), c(0.6, 0.2, 0.2))
safe_auc <- function(groundtruth, probs)
{
  if (length(unique(groundtruth)) < 2) { return (0.5) }
  auc <- as.numeric(pROC::auc(groundtruth, probs))
  return (ifelse(is.na(auc),0.5,(0.5+abs(0.5-auc))))
}

#' Calculates AUC from counts of positives and negatives directly.
#'
#' This is an efficient calculation of the AUC directly from an array of positives and negatives. It makes
#' sure to always return a value between 0.5 and 1.0 and will return 0.5 in case of any issues.
#'
#' @param pos Vector with counts of the positive responses
#' @param neg Vector with counts of the negative responses
#'
#' @return The AUC as a value between 0.5 and 1, return 0.5 if there are any issues with the data.
#' @export
#'
#' @examples
#' dsm_auc( c(3,1,0), c(2,0,1))
dsm_auc <- function(pos, neg)
{
  o <- order(pos/(pos+neg), decreasing = T)
  FPR <- rev(cumsum(neg[o]) / sum(neg))
  TPR <- rev(cumsum(pos[o]) / sum(pos))
  Area <- (FPR-c(FPR[-1],0))*(TPR+c(TPR[-1],0))/2
  auc <- sum(Area)
  return (ifelse(is.na(auc),0.5,(0.5+abs(0.5-auc))))
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

#' Convert from a Pega date-time string.
#'
#' @param x A vector of Pega date-time strings
#'
#' @return An object of class \linkS4class{POSIXlt}
#' @export
#'
#' @examples
#' fromPRPCDateTime("20180316T134127.847 GMT")
fromPRPCDateTime <- function(x)
{
  op <- options(digits.secs=3) # for the milliseconds; the %OS3 is not universally supported so doing via options
  t <- strptime(x, format="%Y%m%dT%H%M%OS", tz="GMT")
  options(op)
  return(t)
}

#' Convert to a Pega date-time string.
#'
#' @param x A vector of valid date-time objects (valid for \code{\link[base]{strftime}})
#'
#' @return A vector of string representations in the format used by Pega.
#' @export
#'
#' @examples
#' toPRPCDateTime( lubridate::now() )
toPRPCDateTime <- function(x)
{
  return(strftime(x, format="%Y%m%dT%H%M%OS3", tz="GMT", usetz=T))
}


