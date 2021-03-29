# Drop internal fields, fix types and more
standardizeDatamartModelData <- function(dt, latestOnly)
{
  # drop internal fields
  dt[, names(dt)[grepl("^p[x|z]", names(dt))] := NULL]

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

  if (file.exists(srcFolder) & !dir.exists(srcFolder)) {
    # if just one argument was passed and it happens to be an existing file, try use that
    instancename = srcFolder
    srcFolder = "."
  }

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
    return(predz[, .SD[SnapshotTime == max(SnapshotTime)], by=ModelID]) # careful: which.max would only return 1 row
  } else {
    return(predz)
  }

  return(predz)
}

#' Calculate variable importance from ADM Datamart data.
#'
#' Uses the predictor binning from the ADM Datamart data to find the
#' relative predictor importance. Uses the exact same cumulative log odds
#' scores and weighs these by the evidence in the bins.
#'
#' @param dmModels Data frame (or anything else that converts to
#'   \code{data.table}) with the ADM datamart model data.
#' @param dmPredictors Data frame (or anything else that converts to
#'   \code{data.table}) with the ADM datamart predictor data.
#' @param facets Optional list of names from the model data to split the
#'   predictor importance by.
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
  applyUniformPegaFieldCasing(dmModels)
  applyUniformPegaFieldCasing(dmPredictors)

  # Iterate over all models, (TODO: this is not efficient)
  allNormalizedBins <- lapply(unique(dmModels$ModelID), function(m) {
    if (nrow(dmPredictors[ModelID == m & EntryType == "Active"]) < 1) return (NULL)
    dmmodelList <- normalizedBinningFromDatamart(dmPredictors[ModelID == m],
                                                 modelsForPartition = dmModels[ModelID == m])

    if (length(dmmodelList) > 1) {
      # xxx <<- dmmodelList
      stop(paste("Code assumes model list per AppliesTo/ConfigurationName/ModelID always has 1 element but contains", length(dmmodelList)))
    }

    # To pull in score card scaling factors use the other properties
    # of the returned list - this just uses the binning component of
    # the returned list.
    predictorWeightsPerBin <- getNBFormulaWeights(dmmodelList[[1]]$binning[(IsActive)])$binning[, c("ModelID", "PredictorName", "PredictorType", "BinLabel", "BinPos", "BinNeg", "BinWeight", "BinIndex")]

    # Symbols that were combined into a single bin may have been split into
    # multiple rows, so make sure to return that count so we can factor that
    # in when weighing the bins.
    predictorWeightsPerBin[, BinIndexOccurences := .N, by=c("ModelID", "PredictorName", "BinIndex")]

    return(predictorWeightsPerBin)
  })

  allNormalizedBins <- rbindlist(allNormalizedBins, fill=T)

  # Merge in any fields that are requested for facetting the result
  if (!is.null(facets)) {
    allNormalizedBins <- merge(allNormalizedBins, unique(dmModels[, c("ModelID", facets), with=F]), by="ModelID")
  }

  # Calculate and scale the predictor weights
  # NB: Classifier uppercase is correct, as allNormalizedBins is a normalized binning structure(!)
  varImp <- allNormalizedBins[PredictorType != "CLASSIFIER", .(Importance = stats::weighted.mean(BinWeight/BinIndexOccurences, BinPos+BinNeg)), by=c("PredictorName",facets)][order(-Importance)]
  varImp[, Rank := frank(-Importance, ties.method = "first"), by=facets]
  varImp[, Importance := 100.0*Importance/max(Importance), by=facets]
  setorder(varImp, Rank)

  return(varImp)
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

