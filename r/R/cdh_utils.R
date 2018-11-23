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
#' \code{readDSExport} reads a dataset export file as exported and downloaded from Pega. The export file
#' is formatted as a zipped mulit-line JSON file (\url{http://jsonlines.org/}). \code{readDSExport} will find the
#' most recent file (Pega appends a datetime stamp), unzip it into a temp folder and read the data into a
#' \code{data.table}.
#'
#' @param instancename Name of the file w/o the timestamp, in Pega format <Applies To>_<Instance Name>, or the
#' complete filename including timestamp and zip extension as exported from Pega.
#' @param srcFolder Optional folder to look for the file (defaults to the current folder)
#' @param tmpFolder Optional folder to store the unzipped data (defaults to the source folder)
#' @param excludeComplexTypes Flag to not return complex embedded types, defaults to T so not including nested lists/data frames
#'
#' @return A \code{data.table} with the contents
#' @export
#'
#' @examples
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_All")}
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_All", "~/Downloads")}
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_All_20180316T135038_GMT.zip", "~/Downloads")}
#' \dontrun{readDSExport("~/Downloads/Data-Decision-ADM-ModelSnapshot_All_20180316T135038_GMT.zip")}
readDSExport <- function(instancename, srcFolder=".", tmpFolder=srcFolder, excludeComplexTypes=T)
{
  if(endsWith(instancename, ".zip")) {
    if (file.exists(instancename)) {
      zipFile <- instancename
    } else if (file.exists(paste(srcFolder,instancename,sep="/"))) {
      zipFile <- paste(srcFolder,instancename,sep="/")
    } else {
      stop("File not found")
    }
  } else {
    zipFile <- paste(srcFolder,
                     rev(sort(list.files(path=srcFolder, pattern=paste("^", instancename, "_.*\\.zip$", sep=""))))[1],
                     sep="/")
    if(!file.exists(zipFile)) stop("File not found")
  }
  jsonFile <- paste(tmpFolder,"data.json", sep="/")
  if(file.exists(jsonFile)) file.remove(jsonFile)
  utils::unzip(zipFile, exdir=tmpFolder)
  multiLineJSON <- readLines(jsonFile)
  file.remove(jsonFile)
  ds <- data.table(jsonlite::fromJSON(paste("[",paste(multiLineJSON,sep="",collapse = ","),"]")))
  if (excludeComplexTypes) {
    return(ds [, names(ds)[!sapply(ds, is_list)], with=F])
  } else {
    return(ds)
  }
}

# Internal helper to keep auc a safe number between 0.5 and 1.0 always
safe_range_auc <- function(auc)
{
  return (ifelse(is.na(auc),0.5,(0.5+abs(0.5-auc))))
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
#' auc_from_probs( c("yes", "yes", "no"), c(0.6, 0.2, 0.2))
auc_from_probs <- function(groundtruth, probs)
{
  if (length(unique(groundtruth)) < 2) { return (0.5) }
  auc <- as.numeric(pROC::auc(groundtruth, probs))
  return (safe_range_auc(auc))
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
#' auc_from_bincounts( c(3,1,0), c(2,0,1))
auc_from_bincounts <- function(pos, neg)
{
  o <- order(pos/(pos+neg), decreasing = T)
  FPR <- rev(cumsum(neg[o]) / sum(neg))
  TPR <- rev(cumsum(pos[o]) / sum(pos))
  Area <- (FPR-c(FPR[-1],0))*(TPR+c(TPR[-1],0))/2
  return (safe_range_auc(sum(Area)))
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
  return (2*safe_range_auc(auc)-1)
}

#' Convert from a Pega date-time string.
#'
#' The timezone string is taken into account but it assumed that they are the same for all the strings. If
#' this is not a valid assumption, \code{apply} this function to your data instead of passing in a vector.
#'
#' @param x A vector of Pega date-time strings
#'
#' @return An object of class \code{POSIXct} (this the compact POSIXt class, the lt is the list variation)
#' @export
#'
#' @examples
#' fromPRPCDateTime("20180316T134127.847 GMT")
#' fromPRPCDateTime("20180316T184127.846")
fromPRPCDateTime <- function(x)
{
  op <- options(digits.secs=3) # for the milliseconds; the %OS3 is not universally supported so doing via options
  timezonesplits <- data.table::tstrsplit(x, " ")
  if(length(timezonesplits)>1) {
    t <- strptime(x, format="%Y%m%dT%H%M%OS", tz=timezonesplits[[2]][1])
  } else {
    t <- strptime(x, format="%Y%m%dT%H%M%OS")
  }
  options(op)
  return(as.POSIXct(t))
}

#' Convert to a Pega date-time string.
#'
#' @param x A vector of valid date-time objects (valid for \code{\link[base]{strftime}})
#'
#' @return A vector of string representations in the format used by Pega.
#' @export
#'
#' @examples
#' toPRPCDateTime(Sys.time())
toPRPCDateTime <- function(x)
{
  return(strftime(x, format="%Y%m%dT%H%M%OS3", tz="GMT", usetz=T))
}


