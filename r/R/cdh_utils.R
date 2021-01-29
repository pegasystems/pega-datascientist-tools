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

#' @docType package
#' @name cdhtools
#' @import data.table jsonlite
#' @importFrom rlang is_list
#' @importFrom utils zip
NULL

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
#' applyUniformPegaFieldCasing(DT)
applyUniformPegaFieldCasing <- function(dt)
{
  fields <- names(dt)

  # field name endings we want to see capitalized
  capitializeEndWords <- c("ID", "Key", "Name", "Count", "Time", "DateTime", "UpdateTime",
                           "ToClass", "Version", "Predictor", "Predictors", "Rate", "Ratio",
                           "Negatives", "Positives", "Threshold", "Error", "Importance",
                           "Type", "Percentage", "Index", "Symbol",
                           "LowerBound", "UpperBound", "Bins", "GroupIndex",
                           "ResponseCount", "NegativesPercentage", "PositivesPercentage",
                           "BinPositives", "BinNegatives", "BinResponseCount", "BinResponseCount",
                           "ResponseCountPercentage")

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

# Drop internal fields, fix types and more
standardizeDatamartModelData <- function(dt, latestOnly)
{
  # drop internal fields
  dt[, names(dt)[grepl("^p[x|z]", names(dt))] := NULL]

  # standardized camel casing of fields
  applyUniformPegaFieldCasing(dt)

  # filter to only take latest snapshot
  if (latestOnly) {
    dt <- dt[, .SD[which.max(SnapshotTime)], by=ModelID]
  }

  # some fields notoriously returned as char but are numeric
  dt[, Performance := as.numeric(as.character(Performance))]
  dt[, Positives := as.numeric(as.character(Positives))]
  dt[, Negatives := as.numeric(as.character(Negatives))]

  # convert date/time fields if present and not already converted prior
  if (is.factor(dt$SnapshotTime) || is.character(dt$SnapshotTime)) {
    dt[, SnapshotTime := fromPRPCDateTime(SnapshotTime)]
  }
  if ("FactoryUpdateTime" %in% names(dt)) {
    if (is.factor(dt$FactoryUpdateTime) || is.character(dt$FactoryUpdateTime)) {
      dt[, FactoryUpdateTime := fromPRPCDateTime(FactoryUpdateTime)]
    }
  }

  return(dt)
}

# If context of ADM models has been customized, additional context keys will
# be represented as a JSON string in the pyName field. Here we try to detect
# that and create proper fields instead of a JSON string.
expandJSONContextInNameField <- function(dt)
{
  mapping <- data.table( OriginalName = levels(dt$Name) )
  mapping[, isJSON := startsWith(OriginalName, "{") & endsWith(OriginalName, "}")]
  if (!any(mapping$isJSON)) return(dt)

  jsonFields <- rbindlist(lapply(mapping$OriginalName[mapping$isJSON], jsonlite::fromJSON, flatten=T), fill = T)
  mapping[(isJSON), names(jsonFields) := jsonFields]
  rbindlist(lapply(mapping$OriginalName[mapping$isJSON], jsonlite::fromJSON, flatten=T), fill = T)
  mapping[, isJSON := NULL]
  for (newName in names(mapping)) {
    if (is.character(mapping[[newName]])) {
      mapping[[newName]] <- factor(mapping[[newName]]) # retain as factors
    }
  }
  applyUniformPegaFieldCasing(mapping) # apply uniform naming to new fields as well

  dt <- merge(dt, mapping, by.x="Name", by.y="OriginalName")
  if ("Name.y" %in% names(dt)) {
    # use name from JSON strings but only if present there
    dt[, Name := factor(ifelse(is.na(Name.y), as.character(Name), as.character(Name.y)))]
    dt[, Name.y := NULL]
  }
  return(dt)
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
#'   <Applies To>_<Instance Name>, or the complete filename including timestamp
#'   and zip extension as exported from Pega.
#' @param srcFolder Optional folder to look for the file (defaults to the
#'   current folder)
#' @param tmpFolder Optional folder to store the unzipped data (defaults to a temp folder)
#' @param excludeComplexTypes Flag to not return complex embedded types,
#'   defaults to T so not including nested lists/data frames
#' @param acceptJSONLines Optional filter function. If given, this is applied
#'   to the lines with JSON data and only those lines for which the function
#'   returns TRUE will be parsed. This is just for efficiency when reading
#'   really big files so you can filter early.
#' @param stringsAsFactors Logical (default is FALSE). Convert all character columns to factors?
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
readDSExport <- function(instancename, srcFolder=".", tmpFolder=tempdir(check = T), excludeComplexTypes=T, acceptJSONLines=NULL, stringsAsFactors=F)
{
  if(endsWith(instancename, ".json")) {
    if (file.exists(instancename)) {
      jsonFile <- instancename
      multiLineJSON <- readLines(jsonFile)
    } else if (file.exists(file.path(srcFolder,instancename))) {
      jsonFile <- file.path(srcFolder,instancename)
      multiLineJSON <- readLines(jsonFile)
    } else {
      stop("File not found (specified a JSON file)")
    }
  } else {
    if(endsWith(instancename, ".zip")) {
      if (file.exists(instancename)) {
        zipFile <- instancename
      } else if (file.exists(file.path(srcFolder,instancename))) {
        zipFile <- file.path(srcFolder,instancename)
      } else {
        stop("File not found (specified a ZIP file)")
      }
    } else {
      zipFile <- paste(srcFolder,
                       rev(sort(list.files(path=srcFolder, pattern=paste("^", instancename, "_.*\\.zip$", sep=""))))[1],
                       sep="/")
      if(!file.exists(zipFile)) stop(paste("File not found (looking for most recent file matching", instancename, "in", srcFolder, ")"))
    }
    jsonFile <- file.path(tmpFolder,"data.json")
    if(file.exists(jsonFile)) file.remove(jsonFile)
    utils::unzip(zipFile, exdir=tmpFolder)
    multiLineJSON <- readLines(jsonFile)
    file.remove(jsonFile)
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
    # rbindlisting all the many tables?
    ds <- data.table(jsonlite::fromJSON(paste("[",paste(multiLineJSON[from:to],sep="",collapse = ","),"]")), stringsAsFactors = stringsAsFactors)
    if (excludeComplexTypes) {
      chunkList[[n]] <- ds [, names(ds)[!sapply(ds, rlang::is_list)], with=F]
    } else {
      chunkList[[n]] <- ds
    }
  }
  return(rbindlist(chunkList, use.names = T, fill = T))
}

# TODO provide similar func for IH data - even only for demo scenarios that strips off internal fields

# TODO BinType probably does not belong in predictor data w/o bins

#' Read export of ADM model data.
#'
#' This is a specialized version of \code{readDSExport}
#' that defaults the dataset name, leaves out the detailed model data (if present) and
#' other internal fields, returns the properties without the py prefixes and converts
#' date fields, and makes sure numeric fields are returned as numerics.
#'
#' @param srcFolder Optional folder to look for the file (defaults to the
#'   current folder)
#' @param instancename Name of the file w/o the timestamp, in Pega format
#'   <Applies To>_<Instance Name>, or the complete filename including timestamp
#'   and zip extension as exported from Pega. Defaults to the Pega generated
#'   name of the dataset: \code{Data-Decision-ADM-ModelSnapshot_pyModelSnapshots}.
#' @param latestOnly If TRUE only the most recent snapshot for every model
#'   is read. Defaults to FALSE, so all model data over time is returned.
#' @param tmpFolder Optional folder to store the unzipped data (defaults to a
#' temp folder)
#'
#' @return A \code{data.table} with the ADM model data
#' @export
#'
#' @examples
#' \dontrun{readADMDatamartModelExport("~/Downloads")}
readADMDatamartModelExport <- function(srcFolder=".",
                                       instancename = "Data-Decision-ADM-ModelSnapshot_pyModelSnapshots",
                                       latestOnly = F,
                                       tmpFolder=tempdir(check = T))
{
  if (file.exists(srcFolder) & !dir.exists(srcFolder)) {
    # if just one argument was passed and it happens to be an existing file, try use that
    instancename = srcFolder
    srcFolder = "."
  }

  modelz <- readDSExport(instancename, srcFolder, tmpFolder=tmpFolder, stringsAsFactors=T)
  if ("pyModelData" %in% names(modelz)) {
    modelz[,pyModelData := NULL]
  }

  modelz <- standardizeDatamartModelData(modelz, latestOnly=latestOnly)
  modelz <- expandJSONContextInNameField(modelz)

  return(modelz)
}

#' Read export of ADM predictor data.
#'
#' This is a specialized version of \code{readDSExport}
#' that defaults the dataset name, leaves out internal fields and by default omits the
#' predictor binning data. In addition it converts date fields and makes sure
#' the returned fields have the correct type. All string values are returned
#' as factors.
#'
#' @param srcFolder Optional folder to look for the file (defaults to the
#'   current folder)
#' @param instancename Name of the file w/o the timestamp, in Pega format
#'   <Applies To>_<Instance Name>, or the complete filename including timestamp
#'   and zip extension as exported from Pega. Defaults to the Pega generated
#'   name of the dataset: \code{Data-Decision-ADM-PredictorBinningSnapshot_PredictorBinningSnapshot}.
#' @param noBinning If TRUE (the default), skip reading the predictor binning data and just
#'   return predictor level data. For a detailed view with all binning data, set
#'   to FALSE.
#' @param latestOnly If TRUE (the default) only the most recent snapshot for every model
#'   is read. To return all data over time (if available), set to FALSE.
#' @param tmpFolder Optional folder to store the unzipped data (defaults to a
#' temp folder)
#'
#' @return A \code{data.table} with the ADM model data
#' @export
#'
#' @examples
#' \dontrun{readADMDatamartPredictorExport("~/Downloads")}
readADMDatamartPredictorExport <- function(srcFolder=".",
                                           instancename = "Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots",
                                           noBinning = T,
                                           latestOnly = T,
                                           tmpFolder=tempdir(check = T))
{
  noBinningSkipFields <- c("pyBinSymbol","pyBinNegativesPercentage","pyBinPositivesPercentage",
                           "pyBinNegatives", "pyBinPositives", "pyRelativeBinNegatives", "pyRelativeBinPositives",
                           "pyBinResponseCount", "pyRelativeBinResponseCount", "pyBinResponseCountPercentage",
                           "pyBinLowerBound", "pyBinUpperBound", "pyZRatio", "pyLift", "pyBinIndex")

  if (noBinning) {
    predz <- readDSExport(instancename, srcFolder, tmpFolder=tmpFolder,
                          acceptJSONLines=function(linez) {
                            # TODO make non fixed and a bit safer, allowing for
                            # extra spaces and so on
                            return(grepl('"pyBinIndex":1,', linez, fixed=T))
                            },
                          stringsAsFactors=T)
    # just to be very defensive, double check there really are no other bins left
    if (any(predz$pyBinIndex > 1)) {
      predz <- predz[pyBinIndex = 1]
    }
    predz[, pyBinIndex := NULL]

    predz[, intersect(names(predz), noBinningSkipFields) := NULL]
  } else {
    predz <- readDSExport(instancename, srcFolder, tmpFolder=tmpFolder)
  }

  predz[, names(predz)[grepl("^p[x|z]", names(predz))] := NULL]
  applyUniformPegaFieldCasing(predz)
  predz[, Performance := as.numeric(as.character(Performance))] # some fields notoriously returned as char but are numeric
  predz[, Positives := as.numeric(as.character(Positives))]
  predz[, Negatives := as.numeric(as.character(Negatives))]

  predz[, SnapshotTime := fromPRPCDateTime(SnapshotTime)]

  if (latestOnly) {
    return(predz[, .SD[which.max(SnapshotTime)], by=ModelID])
  } else {
    return(predz)
  }

  return(predz)
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
  auc <- as.numeric(pROC::auc(groundtruth, probs, quiet=T))
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

# TODO below function is tricky and depends on names. It was used in the
# modelreport.Rmd but possibly also internally.

#' Helper function to return the actual and reported performance of ADM models.
#'
#' In the actual performance the actual score range of the model is
#' taken into account, while for the stored (reported) performance that is not always the case. This function is mainly in support of validating
#' a fix to the reported performance.
#'
#' @param dmModels A \code{data.table} with (possibly a subset of) the models exported from the ADM Datamart (\code{Data-Decision-ADM-ModelSnapshot}).
#' @param dmPredictors A \code{data.table} with the predictors exported from the ADM Datamart (\code{Data-Decision-ADM-PredictorBinningSnapshot}).
#' @param jsonPartitions A list with the (possibly subset) partitions data from the ADM Factory table
#'
#' @return A \code{data.table} with model name, stored performance, actual performance, number of responses and the score range.
#' @export
getModelPerformanceOverview <- function(dmModels = NULL, dmPredictors = NULL, jsonPartitions = NULL)
{
  if (!is.null(dmModels) & !is.null(dmPredictors)) {
    modelList <- createListFromDatamart(dmPredictors[ModelID %in% dmModels$ModelID],
                                        fullName="Dummy",
                                        modelsForPartition=dmModels)
  } else if (!is.null(dmPredictors) & is.null(dmModels)) {

    modelList <- createListFromDatamart(dmPredictors, fullName="Dummy")
    modelList[[1]][["context"]] <- list("Name" =  "Dummy") # should arguably be part of the createList but that breaks some tests, didnt want to bother
  } else if (!is.null(jsonPartitions)) {
    modelList <- createListFromADMFactory(jsonPartitions, overallModelName="Dummy")
  } else {
    stop("Needs either datamart or JSON Factory specifications")
  }

  # Name, response count and reported performance obtained from the data directly. For the score min/max using the utility function
  # that summarizes the predictor bins into a table ("ScaledBins") with the min and max weight per predictor. These weights are the
  # normalized log odds and score min/max is found by adding them up.

  perfOverview <- data.table( name = sapply(modelList, function(m) {return(m$context$pyName)}),
                              reported_performance = sapply(modelList, function(m) {return(m$binning$Performance[1])}),
                              actual_performance = NA, # placeholder
                              responses = sapply(modelList, function(m) {return(m$binning$TotalPos[1] + m$binning$TotalNeg[1])}),
                              score_min = sapply(modelList, function(m) {scaled <- setBinWeights(copy(m$binning)); return(sum(scaled$binning[PredictorType != "CLASSIFIER"]$minWeight))}),
                              score_max = sapply(modelList, function(m) {scaled <- setBinWeights(copy(m$binning)); return(sum(scaled$binning[PredictorType != "CLASSIFIER"]$maxWeight))}))


  classifiers <- lapply(modelList, function(m) { return (m$binning[PredictorType == "CLASSIFIER"])})

  findClassifierBin <- function( classifierBins, score )
  {
    if (nrow(classifierBins) == 1) {
      return(-999)
    }

    return (1 + findInterval(score, classifierBins$BinUpperBound[1 : (nrow(classifierBins) - 1)]))
  }

  perfOverview$nbins <- sapply(classifiers, nrow)
  perfOverview$actual_score_bin_min <- sapply(seq(nrow(perfOverview)),
                                              function(n) { return( findClassifierBin( classifiers[[n]], perfOverview$score_min[n]))} )
  perfOverview$actual_score_bin_max <- sapply(seq(nrow(perfOverview)),
                                              function(n) { return( findClassifierBin( classifiers[[n]], perfOverview$score_max[n]))} )
  perfOverview$actual_performance <- sapply(seq(nrow(perfOverview)),
                                           function(n) { return( auc_from_bincounts(
                                             classifiers[[n]]$BinPos[perfOverview$actual_score_bin_min[n]:perfOverview$actual_score_bin_max[n]],
                                             classifiers[[n]]$BinNeg[perfOverview$actual_score_bin_min[n]:perfOverview$actual_score_bin_max[n]] )) })
  # print(perfOverview)
  # print(sapply(perfOverview, class))
  # print(modelList)
  # stop("Booh")
  setorder(perfOverview, name)


  return(perfOverview)
}

