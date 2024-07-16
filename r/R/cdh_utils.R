#' pdstools: Open-sourced Data Science tools for Pega.
#'
#' Open source tooling to help Data Scientists working with Pega analyze Pega
#' models, responses and predictors using Python and R. There are utility
#' functions to read data, easily create plots and more. See the Wiki of the
#' GitHub repository at \url{https://github.com/pegasystems/pega-datascientist-tools/wiki}
#' for a more detailed description and example notebooks.
#'
#' @section A note on style:
#'
#' The utilities in pdstools heavily rely on \code{data.table} and often expect
#' arguments or return values as a \code{data.table}. This table library provides
#' a high-performance version of base R's \code{data.frame}, see \url{https://github.com/Rdatatable/data.table/wiki}.

#' @docType package
#' @name pdstools
#' @import data.table jsonlite ggplot2 stats
#' @importFrom rlang is_list
#' @importFrom utils zip head
#' @importFrom arrow read_parquet read_feather
#' @importFrom DBI dbGetQuery
#' @importFrom scales percent muted
#' @importFrom base64enc base64decode
#' @importFrom lubridate is.POSIXt parse_date_time
#' @importFrom XML xmlNode xmlTextNode
NULL

# TODO explicitly use lubridate week/month etc instead of data.table versions?

#' Fix field casing
#'
#' Uniformly sets the field names for data.tables from DB and dataset export,
#' capitalizing first letter, fixing common endings.
#'
#' @param dt Data table to apply to
#'
#' @return The names of the input are modified by reference
#' @export
#'
#' @examples
#' DT <- data.table::data.table(id = "abc", pyresponsecount = 1234)
#' standardizeFieldCasing(DT)
standardizeFieldCasing <- function(dt)
{
  fields <- names(dt)

  # field name endings we want to see capitalized
  capitializeEndWords <- c("ID", "Key", "Name", "Count", "Category", "Data",
                           "Time", "DateTime", "UpdateTime",
                           "ToClass", "Version", "Predictor", "Predictors", "Rate", "Ratio",
                           "Negatives", "Positives", "Threshold", "Error", "Importance",
                           "Type", "Percentage", "Index", "Symbol",
                           "LowerBound", "UpperBound", "Bins", "GroupIndex",
                           "ResponseCount", "NegativesPercentage", "PositivesPercentage",
                           "BinPositives", "BinNegatives", "BinResponseCount",
                           "ResponseCountPercentage", "AUC")

  # remove px/py/pz prefixes
  fields <- gsub(pattern = "^p(x|y|z)", replacement = "", tolower(fields))

  # capitalize word endings
  for (w in capitializeEndWords) {
    # exact match at the end
    fields <- gsub(paste0(w,"$"), w, fields, ignore.case=T)
  }

  # capitalize first letter
  fields <- gsub("^(.)", "\\U\\1", fields, perl = T)

  setnames(dt, fields)
}

#' Read a Pega dataset export file.
#'
#' \code{readDSExport} reads a dataset export file as exported and downloaded
#' from Pega datasets. This export file is a zipped multi-line JSON file
#' (\url{http://jsonlines.org/}). \code{readDSExport} will find the most recent
#' file (Pega appends a datetime stamp) if given a folder, then unzips it into a
#' temp folder and read the data into a \code{data.table}. You can also specify
#' a full file name to the export or to a JSON file that you unpacked yourself.
#'
#' @param instancename Name of the file w/o the timestamp, in Pega format
#'   <Applies To>_<Instance Name>, or the complete filename of the dataset
#'   export file, or a specific JSON file.
#' @param srcFolder Optional folder to look for the file (defaults to the
#'   current folder)
#' @param tmpFolder Optional folder to store the unzipped data (defaults to a temp folder)
#' @param excludeComplexTypes Flag to not return complex embedded types,
#'   defaults to T so not including nested lists/data frames
#' @param acceptJSONLines Optional filter function. If given, this is applied
#'   to the lines with JSON data and only those lines for which the function
#'   returns TRUE will be parsed. This is just for efficiency when reading
#'   really big files so you can filter rows early.
#' @param stringsAsFactors Logical (default is FALSE). Convert all character columns to factors?
#' @param verbose Flag for verbose logging, defaults to FALSE.
#'
#' @return A \code{data.table} with the contents
#' @export
#'
#' @examples
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_All")}
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_All", "~/Downloads")}
#' \dontrun{readDSExport("Data-Decision-ADM-ModelSnapshot_All_20180316T135038_GMT.zip",
#' "~/Downloads")}
#' \dontrun{readDSExport("~/Downloads/Data-Decision-ADM-ModelSnapshot_All_20180316T135038_GMT.zip")}
#' \dontrun{readDSExport("SampleApp-SR_DecisionResults.json", "data")}
#' \dontrun{readDSExport("json$", "create_hds/data")}
readDSExport <- function(instancename, srcFolder=".",
                         tmpFolder=tempdir(check = T),
                         excludeComplexTypes=T,
                         acceptJSONLines=NULL,
                         stringsAsFactors=F,
                         verbose=F)
{
  jsonFile <- NULL
  errorReason <- ""

  if(endsWith(instancename, ".json")) {
    errorReason <- "reading existing JSON file"

    # See if it is a single, existing, JSON file
    if (file.exists(instancename)) {
      jsonFile <- instancename
      if (verbose) cat(errorReason, jsonFile, fill=T)
      multiLineJSON <- readLines(jsonFile)
    } else if (file.exists(file.path(srcFolder,instancename))) {
      jsonFile <- file.path(srcFolder,instancename)
      if (verbose) cat(errorReason, jsonFile, fill=T)
      multiLineJSON <- readLines(jsonFile)
    }
  } else {
    # See if it is a fully specified ZIP file then assume this is a Pega
    # export with a "data.json" file inside.
    zipFile <- instancename
    if (verbose) cat("Reading", zipFile, fill=T)

    if(endsWith(instancename, ".zip")) {
      errorReason <- "reading existing ZIP file"

      if (file.exists(instancename)) {
        zipFile <- instancename
      } else if (file.exists(file.path(srcFolder,instancename))) {
        zipFile <- file.path(srcFolder,instancename)
      }
      if (verbose) cat(errorReason, zipFile, fill=T)
    } else {
      errorReason <- "reading most recent dataset export ZIP file"

      # See if it is just a base name of the instance, then find the most
      # recent file by appending the date and "zip" extension according to
      # the Pega dataset export conventions.

      zipFile <- paste(srcFolder,
                       rev(sort(list.files(path=srcFolder, pattern=paste("^", instancename, "_.*\\.zip$", sep=""))))[1],
                       sep="/")
      if (verbose) cat(errorReason, zipFile, fill=T)
    }

    if (file.exists(zipFile)) {
      errorReason <- "unzipping data.json from standard dataset export"
      if (verbose) cat(errorReason, zipFile, fill=T)

      jsonFile <- file.path(tmpFolder,"data.json")
      if(file.exists(jsonFile)) file.remove(jsonFile)

      allZipEntries <- utils::unzip(zipFile, list=T)$Name
      jsonFileInZip <- allZipEntries[grepl("[/]+data.json$", allZipEntries)]
      if (verbose) cat("data.json in zip file:",jsonFileInZip, zipFile, fill=T)
      utils::unzip(zipFile, exdir=tmpFolder, junkpaths = T, files=jsonFileInZip)

      if (!file.exists(jsonFile)) stop(paste("Expected Pega Dataset Export zipfile but",
                                             zipFile,
                                             'does not contain "data.json"'))
      multiLineJSON <- readLines(jsonFile)
      file.remove(jsonFile)
    }
  }
  #
  #   Deprecated option. Users should just use arrow directly to read JSON.
  #
  #   if (is.null(jsonFile)) {
  #     errorReason <- paste("looking for pattern to match", instancename)
  #     if (verbose) cat(errorReason, fill=T)
  #
  #     # Try to interpret the filename as a pattern.
  #     jsonFile <- list.files(pattern = instancename, path = srcFolder, full.names = T)
  #     if (length(jsonFile) > 0) {
  #       multiLineJSON <- unlist(sapply(jsonFile, readLines, warn=F))
  #     } else {
  #       jsonFile <- NULL
  #     }
  #  }

  if (is.null(jsonFile)) {
    stop(paste("Dataset JSON file not found", errorReason))
  }

  if(!is.null(acceptJSONLines)) {
    multiLineJSON <- multiLineJSON[acceptJSONLines(multiLineJSON)]
  }

  # To prevent OOM for very large files we chunk up the input for conversion to tabular. The average
  # line length is obtained from the first 1000 entries, and we assume a max total of 500M.
  # This is conservative at least on my laptop.
  chunkSize <- trunc(500e6 / mean(sapply(utils::head(multiLineJSON,1000), nchar)))
  chunkList <- list()
  for (n in seq(ceiling(length(multiLineJSON)/chunkSize))) {
    from <- (n-1)*chunkSize+1
    to <- min(n*chunkSize, length(multiLineJSON))
    # cat("From", from, "to", to, fill = T)
    # TODO is this really faster than reading the lines one by one and
    # rbindlist-ing all the many tables?
    ds <- data.table(jsonlite::fromJSON(paste("[",paste(multiLineJSON[from:to],sep="",collapse = ","),"]")), stringsAsFactors = stringsAsFactors)
    if (excludeComplexTypes) {
      chunkList[[n]] <- ds [, names(ds)[!sapply(ds, rlang::is_list)], with=F]
    } else {
      chunkList[[n]] <- ds
    }
  }
  return(rbindlist(chunkList, use.names = T, fill = T))
}


#' Write table to a file in the format of the dataset export files.
#'
#' \code{writeDSExport} writes a \code{data.table} in the same zipped,
#' multi-line JSON format (\url{http://jsonlines.org/}) as the Pega datasets.
#'
#' @param x Data table/frame to export.
#' @param filename Filename to export to.
#' @param tmpFolder Optional folder to store the multi-line JSON file
#'
#' @export
#'
#' @examples
#' \dontrun{writeDSExport(myData, "~/Downloads/mydata.zip")}
writeDSExport <- function(x, filename, tmpFolder=tempdir())
{
  jsonFile <- file.path(tmpFolder, "data.json")
  writeLines(sapply(1:nrow(x), function(r) {
    jsonlite::toJSON(as.list(x[r,]), auto_unbox = T, na = "null", digits = NA)}),
    jsonFile)
  zip(filename, jsonFile, flags = "-9Xj")
  file.remove(jsonFile)
}

# Internal helper to keep auc a safe number between 0.5 and 1.0 always
safe_range_auc <- function(auc)
{
  return (ifelse(is.na(auc),0.5,(0.5+abs(0.5-auc))))
}

#' Calculates AUC from an array of truth values and predictions.
#'
#' Calculates the area under the ROC curve from an array of truth values and predictions, making sure to
#' always return a value between 0.5 and 1.0 and returns 0.5 when there is just one
#' groundtruth label.
#'
#' @param groundtruth The 'true' values, with just 2 different values
#' @param probs The predictions, as a numeric vector of the same length as \code{groundtruth}
#'
#' @return The AUC as a value between 0.5 and 1.
#' @export
#'
#' @examples
#' auc_from_probs( c("yes", "yes", "no"), c(0.6, 0.2, 0.2))
auc_from_probs <- function(groundtruth, probs)
{
  nlabels <- length(unique(groundtruth))
  if (nlabels < 2) { return (0.5) }
  if (nlabels > 2) { stop("'Groundtruth' has more than two levels.") }

  # with external lib it would be something like this:
  # auc <- MLmetrics::AUC(y_pred = probs, y_true = as.integer(as.factor(groundtruth))-1)
  # return (safe_range_auc(auc))
  # or:
  # auc <- as.numeric(pROC::auc(groundtruth, probs, quiet=T))
  # return (safe_range_auc(auc))

  probabilities <- data.frame(groundtruth = as.integer(as.factor(groundtruth)) == 2,
                              probs = probs)
  groupedProbabilities <- by(probabilities, # base group by, avoiding need for frameworks
                             probabilities$probs,
                             function(x) {return(c(p=first(x[2]),
                                                   pos=sum(x$groundtruth),
                                                   neg=sum(!x$groundtruth)))})
  binProbs <- unlist(lapply(groupedProbabilities, function(x) {x[[1]]}))
  binPos <- unlist(lapply(groupedProbabilities, function(x) {x[[2]]}))
  binNeg <- unlist(lapply(groupedProbabilities, function(x) {x[[3]]}))
  return(auc_from_bincounts(binPos, binNeg, binProbs))
}

#' Calculates PR AUC (precision-recall) from an array of truth values and predictions.
#'
#' Calculates the area under the PR curve from an array of truth values and predictions.
#' Returns 0.0 when there is just one groundtruth label.
#'
#' @param groundtruth The 'true' values, with just 2 different values
#' @param probs The predictions, as a numeric vector as the same length as \code{groundtruth}
#'
#' @return The AUC as a value between 0.0 and 1.0
#' @export
#'
#' @examples
#' aucpr_from_probs( c("yes", "yes", "no"), c(0.6, 0.2, 0.2))
aucpr_from_probs <- function(groundtruth, probs)
{
  nlabels <- length(unique(groundtruth))
  if (nlabels < 2) { return (0.0) }
  if (nlabels > 2) { stop("'Groundtruth' has more than two levels.") }

  # with external lib it would be something like this:
  # auc <- MLmetrics::PRAUC(y_pred = probs, y_true = as.integer(as.factor(groundtruth))-1)
  # return (safe_range_auc(auc))

  probabilities <- data.frame(groundtruth = as.integer(as.factor(groundtruth)) == 2,
                              probs = probs)
  groupedProbabilities <- by(probabilities, # base group by, avoiding need for frameworks
                             probabilities$probs,
                             function(x) {return(c(p=first(x[2]),
                                                   pos=sum(x$groundtruth),
                                                   neg=sum(!x$groundtruth)))})
  binProbs <- unlist(lapply(groupedProbabilities, function(x) {x[[1]]}))
  binPos <- unlist(lapply(groupedProbabilities, function(x) {x[[2]]}))
  binNeg <- unlist(lapply(groupedProbabilities, function(x) {x[[3]]}))
  return(aucpr_from_bincounts(binPos, binNeg, binProbs))
}

#' Calculates AUC from counts of positives and negatives directly.
#'
#' This is an efficient calculation of the area under the ROC curve directly from an array of positives and negatives. It makes
#' sure to always return a value between 0.5 and 1.0 and will return 0.5 when there is just one
#' groundtruth label.
#'
#' @param pos Vector with counts of the positive responses
#' @param neg Vector with counts of the negative responses
#' @param probs Optional probabilities, used to set the binorder, defaults to pos/(pos+neg)
#' @param binorder Optional order of the bins, defaults to decreasing order of propensities.
#'
#' @return The ROC AUC as a value between 0.5 and 1.
#' @export
#'
#' @examples
#' auc_from_bincounts( c(3,1,0), c(2,0,1))
auc_from_bincounts <- function(pos, neg, probs = pos/(pos+neg), binorder = NULL)
{
  if (is.null(binorder)) {
    binorder <- order(probs, decreasing = T)
  }

  FPR <- cumsum(neg[binorder]) / sum(neg)
  TPR <- cumsum(pos[binorder]) / sum(pos)
  Area <- (FPR - shift(FPR, 1, fill=0)) * (TPR + shift(TPR, 1, fill=0)) / 2
  return (safe_range_auc(sum(Area)))
}

#' Calculates PR AUC (precision-recall) from counts of positives and negatives directly.
#'
#' This is an efficient calculation of the area under the PR curve directly from an
#' array of positives and negatives. Returns 0.5 when there is just one
#' groundtruth label.
#'
#' @param pos Vector with counts of the positive responses
#' @param neg Vector with counts of the negative responses
#' @param probs Optional probabilities, used to set the binorder, defaults to pos/(pos+neg)
#' @param binorder Optional order of the bins, defaults to decreasing order of propensities.
#'
#' @return The PR AUC as a value between 0.0 and 1.0
#' @export
#'
#' @examples
#' auc_from_bincounts( c(3,1,0), c(2,0,1))
aucpr_from_bincounts <- function(pos, neg, probs = pos/(pos+neg), binorder = NULL)
{
  if (is.null(binorder)) {
    binorder <- order(probs, decreasing = T)
  }

  recall <- cumsum(pos[binorder]) / sum(pos)
  precision <- cumsum(pos[binorder]) / cumsum(pos[binorder] + neg[binorder])
  Area <- (recall - shift(recall, 1, fill=0)) * (precision + shift(precision, 1, fill=0)) / 2
  return (sum(Area[-1])) # not mapping to 0.5 - 1.0
}

#' Convert AUC performance metric to GINI.
#'
#' @param auc The ROC AUC (number between 0.5 and 1)
#'
#' @return GINI metric, a number between 0 and 1
#' @export
#'
#' @examples
#' auc2GINI(0.8232)
auc2GINI <- function(auc)
{
  return (2*safe_range_auc(auc)-1)
}

#' Calculates lift from counts of positives and negatives.
#'
#' Lift is defined as the success rate in a single bin divided by the overall success rate.
#'
#' @param pos Vector with counts of the positive responses
#' @param neg Vector with counts of the negative responses
#'
#' @return A vector with lift values. Lift can be any positive number. A return value of 2 means 200% lift.
#' @export
#'
#' @examples
#' 100*lift(c(0,119,59,69,0), c(50,387,105,40,37))
lift <- function(pos, neg)
{
  l <- (pos/(pos+neg)) / (sum(pos)/sum(pos+neg))
  return(l)
}

#' Calculates the Z-Ratio for the success rate from counts of positives and negatives.
#'
#' Z-Ratio is the difference between the propensity in the bin with the average propensity, expressed in standard deviations from the average.
#'
#' @param pos Vector with counts of the positive responses
#' @param neg Vector with counts of the negative responses
#'
#' @return A vector with z-ratio values. Z-ratio can be both positive and negative.
#' @export
#'
#' @examples
#' zratio(c(0,119,59,69,0), c(50,387,105,40,37))
zratio <- function(pos, neg)
{
  posFraction <- pos / sum(pos)
  negFraction <- neg / sum(neg)
  ratio <- (posFraction - negFraction) / sqrt(posFraction*(1-posFraction)/sum(pos) + negFraction*(1-negFraction)/sum(neg))
  return(ratio)
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
#' @param x A vector of valid date-time objects (valid for \code{\link[base:strptime]{strftime}})
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

# robust version of which.max that can deal with NA's
safe_which_max <- function(x)
{
  if (all(is.na(x))) return (length(x))
  return (which.max(x))
}

safe_is_max <- function(x)
{
  if (all(is.na(x))) return (rep(T,length(x)))
  return (x == max(x, na.rm=T))
}

