# Flexible time conversion - necessary to support different databases as
# this is not always Pega format in some of the database exports
standardizedParseTime <- function(t)
{
  if (!is.POSIXt(t) && (is.factor(t) || is.character(t))) {
    suppressWarnings(timez <- fromPRPCDateTime(t))
    if (sum(is.na(timez))/length(timez) > 0.2) {
      suppressWarnings(timez <- parse_date_time(t, orders=c("%Y-%m-%d %H:%M:%S", "%y-%b-%d") )) # TODO: more formats here
      if (sum(is.na(timez))/length(timez) > 0.2) {
        warning(paste("Assumed Pega date-time string but resulting in over 20% NA's in snapshot time after conversion.",
                      "Check that this is valid or update the code that deals with date/time conversion.",
                      "Head: ", paste(head(t), collapse=";")))
      }
    }
    return(timez)
  }
  return(t)
}

# Drop internal fields, fix types and more
standardizeDatamartModelData <- function(dt, latestOnly)
{
  # drop internal fields
  if (any(grepl("^p[x|z]", names(dt), ignore.case = T))) {
    dt[, names(dt)[grepl("^p[x|z]", names(dt), ignore.case = T)] := NULL]
  }

  # standardized camel casing of fields
  applyUniformPegaFieldCasing(dt)

  # filter to only take latest snapshot
  if (latestOnly) {
    if (is.factor(dt$SnapshotTime)) {
      dt <- dt[, .SD[SnapshotTime == max(as.character(SnapshotTime))], by=ModelID] # careful, which.max returns only 1
    } else {
      dt <- dt[, .SD[SnapshotTime == max(SnapshotTime)], by=ModelID] # careful, which.max returns only 1
    }
  }

  # some fields notoriously returned as char but are numeric
  for (fld in c("Performance", "Positives", "Negatives")) {
    if (fld %in% names(dt)) {
      if (!is.numeric(dt[[fld]])) dt[[fld]] <- as.numeric(as.character(dt[[fld]]))
    }
  }

  # convert date/time fields if present and not already converted prior
  if (!is.POSIXt(dt$SnapshotTime)) {
    dt[, SnapshotTime := standardizedParseTime(SnapshotTime)]
  }
  if ("FactoryUpdateTime" %in% names(dt)) {
    if (!is.POSIXt(dt$FactoryUpdateTime)) {
      dt[, FactoryUpdateTime := standardizedParseTime(FactoryUpdateTime)]
    }
  }

  return(dt)
}

#' Expand embedded JSON in a column of a data.table into separate columns.
#'
#' Mostly
#' used to expand Name + Treatment that are embedded in the Name field as JSON
#' strings. Other ADM context key modifications will also show as JSON in the
#' Name fields, this utility will peel it apart.
#'
#' Called as part of the standard model read, but exposed so it can also be
#' applied when reading data from different sources like a CSV file.
#'
#' @param dt The data.table to use
#' @param fieldName The field name to look for embedded JSON. Defaults to Name.
#'
#' @return A data.table with the extra columns. If there is no JSON in the
#' field returns the original data.table directly.
#' @export
#'
#' @examples
#' \dontrun{models <- expandEmbeddedJSONContext(models)}
expandEmbeddedJSONContext <- function(dt, fieldName = "Name")
{
  if (!is.factor(dt[[fieldName]])) {
    dt[[fieldName]] <- as.factor(dt[[fieldName]])
  }
  mapping <- data.table( OriginalName = levels(dt[[fieldName]]) )
  mapping[, isJSON := startsWith(OriginalName, "{") & endsWith(OriginalName, "}")]

  # Exit if it doesn't seem to be a JSON string
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

  dt <- merge(dt, mapping, by.x=fieldName, by.y="OriginalName")
  if (paste0(fieldName, ".y") %in% names(dt)) {
    # use name from JSON strings but only if present there
    dt[[fieldName]] <- factor(ifelse(is.na(dt[[paste0(fieldName, ".y")]]), as.character(dt[[fieldName]]), as.character(dt[[paste0(fieldName, ".y")]])))
    dt[[paste0(fieldName, ".y")]] <- NULL
  }

  return(dt)
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
  if ("ModelData" %in% names(modelz)) {
    modelz[,ModelData := NULL]
  }

  modelz <- standardizeDatamartModelData(modelz, latestOnly=latestOnly)
  modelz <- expandEmbeddedJSONContext(modelz)

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
#' @param classifierOnly if TRUE only returns the data for the model classifiers.
#'   Default is FALSE, so all predictors are returned. When setting this flag
#'   to TRUE, \code{noBinning} would typically be set to FALSE as the typical
#'   use case is to analyse the score distributions.
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
                                           classifierOnly = F,
                                           latestOnly = T,
                                           tmpFolder=tempdir(check = T))
{
  noBinningSkipFields <- c("BinSymbol","BinNegativesPercentage","BinPositivesPercentage",
                           "BinNegatives", "BinPositives", "RelativeBinNegatives", "RelativeBinPositives",
                           "BinResponseCount", "RelativeBinResponseCount", "BinResponseCountPercentage",
                           "BinLowerBound", "BinUpperBound", "ZRatio", "Lift", "BinIndex")

  if (file.exists(srcFolder) & !dir.exists(srcFolder)) {
    # if just one argument was passed and it happens to be an existing file, try use that
    instancename = srcFolder
    srcFolder = "."
  }

  if (classifierOnly) {
    stop("Not implmemented yet.")
  }

  if (noBinning) {
    predz <- readDSExport(instancename, srcFolder, tmpFolder=tmpFolder,
                          acceptJSONLines=function(linez) {
                            # require binindex = 1 to be in there,
                            # match both with and without py, upper and lower case
                            return(grepl('"[[:alpha:]]*BinIndex"[[:space:]]*:[[:space:]]*1[^[:digit:]]', linez, ignore.case = T))
                          },
                          stringsAsFactors=T)

    if (any(grepl("^p[x|z]", names(predz)))) {
      predz[, names(predz)[grepl("^p[x|z]", names(predz))] := NULL] # drop px/pz fields
    }
    applyUniformPegaFieldCasing(predz)

    # double check there really are no other bins left
    if (any(predz$BinIndex > 1)) {
      predz <- predz[BinIndex == 1]
    }
    predz[, BinIndex := NULL] # drop the column

    predz[, intersect(names(predz), noBinningSkipFields) := NULL]

  } else {
    predz <- readDSExport(instancename, srcFolder, tmpFolder=tmpFolder)

    if (any(grepl("^p[x|z]", names(predz)))) {
      predz[, names(predz)[grepl("^p[x|z]", names(predz))] := NULL] # drop px/pz fields
    }
    applyUniformPegaFieldCasing(predz)
  }

  # some fields notoriously returned as char but are numeric
  for (fld in c("Performance", "Positives", "Negatives", "BinLowerBound", "BinUpperBound")) {
    if (fld %in% names(predz)) {
      if (!is.numeric(predz[[fld]])) predz[[fld]] <- as.numeric(as.character(predz[[fld]]))
    }
  }

  predz[, SnapshotTime := fromPRPCDateTime(SnapshotTime)]

  if (latestOnly) {
    return(predz[, .SD[SnapshotTime == max(SnapshotTime)], by=ModelID]) # careful: which.max would only return 1 row
  } else {
    return(predz)
  }

  return(predz)
}

#' Generic method to read ADM Datamart data into a list structure.
#'
#' Method is very flexible in the arguments. It can take a \code{data.table}
#' for the model data and the predictor data, but the arguments can also be
#' pointing to files in CSV, zipped CSV, (LD)JSON, parquet or dataset export formats.
#'
#' @param modeldata Location, reference or the actual model table from the
#' ADM datamart.
#' @param predictordata Location, reference or the actual predictor binning
#' table from the ADM datamart.
#' @param folder Optional path for the folder in which to look for the model
#' and predictor data.
#' @param keepSerializedModelData By default the serialized model data is left
#' out from the model data as this typically is large and only needed for
#' specific use cases.
#'
#' @return
#' @export
#'
#' @examples
ADMDatamart <- function(modeldata = NULL, predictordata = NULL, folder = NULL,
                        keepModelData = FALSE)
{
  # modeldata can be data.table
  # modeldata can be a file, pointing to CSV or parquet file or export zip
  # modeldata can also be just the base name of the export zip but then there should be a folder
  # modeldata can even be nul then if you specify folder it should pick up the latest

  # TODO this is rather incomplete - to be implemented and used in all plot
  # methods

  return(list(modeldata = modeldata,
              predictordata = predictordata,
              hasMultipleModelSnapshots = T,
              hasMultiplePredictorSnapshots = F,
              hasPredictorBinning = F))
}

#' Calculate variable importance from ADM Datamart data.
#'
#' Uses the predictor binning from the ADM Datamart data to find the
#' relative importance. The importance is given by the distance to the
#' average log odds, per predictor. Next to importance it also returns
#' the univariate predictor performance (AUC).
#'
#' @param dmModels Data frame (or anything else that converts to
#'   \code{data.table}) with the ADM datamart model data. Only used
#'   when there are facets, so can be NULL.
#' @param dmPredictors Data frame (or anything else that converts to
#'   \code{data.table}) with the ADM datamart predictor data.
#' @param facets Optional list of names from the model data to aggregate
#' up to.
#'
#' @return A \code{data.table}) with \code{PredictorName}, \code{Importance}
#' and \code{Rank} plus columns for each of the \code{facets} if supplied.
#' @export
#'
#' @examples
#' \dontrun{
#'   models <- readADMDatamartModelExport("~/Downloads")
#'   preds <- readADMDatamartPredictorExport("~/Downloads", noBinning = F)
#'
#'   varimp <- admVarImp(models, preds)
#' }
admVarImp <- function(dmModels, dmPredictors, facets = NULL)
{
  # Normalize field names
  applyUniformPegaFieldCasing(dmPredictors)

  # Log odds per bin, then the bin weight is the distance to the weighted mean
  bins <- dmPredictors[EntryType=="Active", c("BinPositives", "BinNegatives", "BinResponseCount", "Performance", "PredictorName", "ModelID"), with=F]
  bins[, BinLogOdds := log(BinPositives+1) - log(BinNegatives+1)]
  bins[, AvgLogOdds := weighted.mean(BinLogOdds, BinResponseCount), by=c("PredictorName", "ModelID")]
  bins[, BinDiffLogOdds := abs(BinLogOdds - AvgLogOdds)]

  # The feature importance per predictor then is just the weighted average of these distances to the mean
  featureImportance <-
    bins[, .(Importance = weighted.mean(BinDiffLogOdds, BinResponseCount, na.rm=T),
             Performance = first(as.numeric(Performance)), # numeric cast should not be necessary but sometimes data is odd
             ResponseCount = sum(BinResponseCount)), by=c("PredictorName", "ModelID")]
  featureImportance[, Importance := ifelse(is.na(Importance), 0, Importance)]


  # Now join in any facets - if there are any
  if (!is.null(facets)) {
    applyUniformPegaFieldCasing(dmModels)

    featureImportance <- merge(featureImportance,
                               unique(dmModels[, c(facets, "ModelID"), with=F]), by="ModelID")

    # Then aggregate up to the predictor names x facets
    aggregateFeatureImportance <- featureImportance[, .(Importance = weighted.mean(Importance, ResponseCount),
                                                        Performance = weighted.mean(Performance, ResponseCount),
                                                        ResponseCount = sum(ResponseCount)),
                                                    by=c("PredictorName", facets)]

    aggregateFeatureImportance[, Importance := Importance*100.0/max(Importance), by=facets]
    aggregateFeatureImportance[, ImportanceRank := frank(-Importance, ties.method = "dense"), by=facets]
    aggregateFeatureImportance[, PerformanceRank := frank(-Performance, ties.method = "dense"), by=facets]
    setorder(aggregateFeatureImportance, ImportanceRank)

    return(aggregateFeatureImportance)
  } else {
    featureImportance[, Importance := Importance*100.0/max(Importance), by="ModelID"]
    featureImportance[, ImportanceRank := frank(-Importance, ties.method = "dense"), by="ModelID"]
    featureImportance[, PerformanceRank := frank(-Performance, ties.method = "dense"), by="ModelID"]
    setorder(featureImportance, ImportanceRank)

    return(featureImportance)
  }
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
    modelList <- normalizedBinningFromDatamart(dmPredictors[ModelID %in% dmModels$ModelID],
                                               fullName="Dummy",
                                               modelsForPartition=dmModels)
  } else if (!is.null(dmPredictors) & is.null(dmModels)) {

    modelList <- normalizedBinningFromDatamart(dmPredictors, fullName="Dummy")
    modelList[[1]][["context"]] <- list("Name" =  "Dummy") # should arguably be part of the normalizedBinning but that breaks some tests, didnt want to bother
  } else if (!is.null(jsonPartitions)) {
    modelList <- normalizedBinningFromADMFactory(jsonPartitions, overallModelName="Dummy")
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
                              score_min = sapply(modelList, function(m) {scaled <- getNBFormulaWeights(copy(m$binning)); return(sum(scaled$binning[PredictorType != "CLASSIFIER"]$minWeight))}),
                              score_max = sapply(modelList, function(m) {scaled <- getNBFormulaWeights(copy(m$binning)); return(sum(scaled$binning[PredictorType != "CLASSIFIER"]$maxWeight))}))


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

