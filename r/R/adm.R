
# Flexible time conversion - necessary to support different databases as
# this is not always Pega format in some of the database exports
standardizedParseTime <- function(t)
{
  if (!lubridate::is.POSIXt(t) && (is.factor(t) || is.character(t))) {
    suppressWarnings(timez <- fromPRPCDateTime(t))
    if (sum(is.na(timez))/length(timez) > 0.2) {
      # TODO can do more formats but careful with interpretation of days as years
      # we could be smart prefering a format that results in dates closer to
      # eachother
      suppressWarnings(timez <- parse_date_time(t, orders=c("%Y-%m-%d %H:%M:%S", "%Y-%b-%d", "%d-%b-%y") ))
      if (sum(is.na(timez))/length(timez) > 0.2) {
        warning(paste("Assumed Pega date-time string but resulting in over 20% NA's in snapshot time after conversion.",
                      "Check that this is valid or update the code that deals with date/time conversion.",
                      "Sample values: ", paste(sample(t, min(5, length(t))), collapse="; ")))
      }
    }
    return(timez)
  }
  return(t)
}

# Drop internal fields
dropInternalDatamartFields <- function(dt)
{
  if (any(grepl("^p[x|z]", names(dt), ignore.case = T))) {
    dt[, names(dt)[grepl("^p[x|z]", names(dt), ignore.case = T)] := NULL]
  }

  return(dt)
}

# Drop internal fields
fixDatamartFieldTypes <- function(dt)
{
  SnapshotTime <- NULL # Trick to silence R CMD Check warnings

  # some fields notoriously returned as char but are numeric
  for (fld in c("Performance", "Positives", "Negatives", "BinLowerBound", "BinUpperBound")) {
    if (fld %in% names(dt)) {
      if (!is.numeric(dt[[fld]])) dt[[fld]] <- as.numeric(as.character(dt[[fld]]))
    }
  }

  # convert date/time fields if present and not already converted prior
  if (!lubridate::is.POSIXt(dt$SnapshotTime)) {
    dt[, SnapshotTime := standardizedParseTime(SnapshotTime)]
  }

  return(dt)
}

# Expand embedded JSON in a column of a \code{data.table} into separate columns.
#
# Mostly used to expand Name + Treatment that are embedded in the Name field as JSON
# strings. Other ADM context key modifications will also show as JSON in the
# Name fields, this utility will peel it apart.
expandEmbeddedJSONContext <- function(dt, fieldName = "Name")
{
  isJSON <- OriginalName <- NULL # Trick to silence R CMD Check warnings

  if(nrow(dt)==0) return(dt)

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
  standardizeFieldCasing(mapping) # apply uniform naming to new fields as well

  dt <- merge(dt, mapping, by.x=fieldName, by.y="OriginalName")
  if (paste0(fieldName, ".y") %in% names(dt)) {
    # use name from JSON strings but only if present there
    dt[[fieldName]] <- factor(ifelse(is.na(dt[[paste0(fieldName, ".y")]]), as.character(dt[[fieldName]]), as.character(dt[[paste0(fieldName, ".y")]])))
    dt[[paste0(fieldName, ".y")]] <- NULL
  }

  return(dt)
}


#' Subset the provided datamart data to just the latest snapshot per model.
#'
#' If there is just one snapshot, nothing will change. It works for both
#' model and predictor tables. If there is no snapshottime field, it will
#' not do anything.
#'
#' @param dt The \code{data.table} with the datamart data.
#'
#' @return A \code{data.table} with just the latest snapshots per model.
#' @export
#'
#' @examples
#' data(adm_datamart)
#' filterLatestSnapshotOnly(adm_datamart$modeldata)
filterLatestSnapshotOnly <- function(dt)
{
  ModelID <- NULL # Trick to silence R CMD Check warnings

  l <- function(grp, fld) { grp[grp[[fld]] == max(grp[[fld]])] }

  snapshottimeField <- names(dt)[which(tolower(names(dt)) %in% c("snapshottime", "pysnapshottime"))[1]] # be careful with the names after all manipulation we do
  if (!is.na(snapshottimeField)) {
    return(dt[, l(.SD, snapshottimeField), by=ModelID])
  } else {
    return(dt)
  }
}

#' Subset the datamart predictor data to only the classifier binning.
#'
#' @param dt Predictor datamart table
#' @param reverse Boolean to flip the operation, so setting
#' this to TRUE results in all rows except the classifier.
#'
#' @return A \code{data.table}.
#' @export
#'
#' @examples
#' data(adm_datamart)
#' filterClassifierOnly(adm_datamart$preditordata)
filterClassifierOnly <- function(dt, reverse = F)
{
  EntryType <- NULL # Trick to silence R CMD Check warnings

  if (reverse) {
    return(dt[EntryType!="Classifier"])
  } else {
    return(dt[EntryType=="Classifier"])
  }
}

#' Subset the datamart predictor data to only the active predictors.
#'
#' @param dt Predictor datamart table
#' @param reverse Boolean to flip the operation.
#'
#' @return A \code{data.table}.
#' @export
#'
#' @examples
#' data(adm_datamart)
#' filterActiveOnly(adm_datamart$predictordata)
filterActiveOnly <- function(dt, reverse = F)
{
  EntryType <- NULL # Trick to silence R CMD Check warnings

  if (reverse) {
    return(dt[EntryType!="Active"])
  } else {
    return(dt[EntryType=="Active"])
  }
}

#' Subset the datamart predictor data to only the inactive predictors.
#'
#' @param dt Predictor datamart table
#' @param reverse Boolean to flip the operation.
#'
#' @return A \code{data.table}.
#' @export
#'
#' @examples
#' data(adm_datamart)
#' filterInactiveOnly(adm_datamart$predictordata)
filterInactiveOnly <- function(dt, reverse = F)
{
  EntryType <- NULL # Trick to silence R CMD Check warnings

  if (reverse) {
    return(dt[EntryType!="Inactive"])
  } else {
    return(dt[EntryType=="Inactive"])
  }
}

#' Remove the binning information from the provided datamart predictor data so
#' it only is predictor level, not more detailed. This is commonly used in
#' plotting functions.
#'
#' @param dt The \code{data.table} with the datamart predictor data.
#'
#' @return A \code{data.table} without the predictor binning.
#' @export
#'
#' @examples
#' data(adm_datamart)
#' filterPredictorBinning(adm_datamart$predictordata)
filterPredictorBinning <- function(dt)
{
  BinIndex <- NULL # Trick to silence R CMD Check warnings

  noBinningSkipFields <- c("BinSymbol","BinNegativesPercentage","BinPositivesPercentage",
                           "BinNegatives", "BinPositives", "RelativeBinNegatives", "RelativeBinPositives",
                           "BinResponseCount", "RelativeBinResponseCount", "BinResponseCountPercentage",
                           "BinLowerBound", "BinUpperBound", "ZRatio", "Lift", "BinIndex")

  if (any(dt$BinIndex > 1)) {
    dt <- dt[BinIndex == 1]
  }
  dt[, BinIndex := NULL] # drop the column

  dt[, intersect(names(dt), noBinningSkipFields) := NULL]

  return(dt)
}

#' Check if data has multiple snapshots
#'
#' @param dt Model or predictor table to check.
#'
#' @return True if there is more than one snapshot
#' @export
#'
#' @examples
#' data(adm_datamart)
#' hasMultipleSnapshots(adm_datamart$modeldata)
hasMultipleSnapshots <- function(dt)
{
  if (is.null(dt)) return(F)
  return ((uniqueN(dt$SnapshotTime) > 1))
}

# Returns predictor category as the element before the first dot, or "TopLevel" if there is no dot.
defaultPredictorCategorization <- function(p)
{
  hasDot <- grepl(".", p, fixed = T)
  return (ifelse(hasDot, gsub("^([^.]*)\\..*$", "\\1", p), rep("TopLevel", length(p))))
}

# Read CSV, JSON or parquet file from source
# TODO: consider reading only specific columns, or dropping specific ones
readFromSource <- function(file, folder, tmpFolder)
{
  # Use folder only if file itself doesnt exist
  if (!file.exists(file) & file.exists(file.path(folder, file))) {
    file <- file.path(folder, file)
  }

  if (file.exists(file)) {
    if (endsWith(file, ".csv")) {
      return(fread(file))
    }

    if (endsWith(file, ".json")) {
      # Speedy JSON read through arrow
      return(as.data.table(arrow::read_json_arrow(file)))
    }

    if (endsWith(file, ".parquet")) {
      return(as.data.table(arrow::read_parquet(file)))
    }
  }

  # Generic dataset read
  dt <- readDSExport(file, folder, tmpFolder=tmpFolder, stringsAsFactors=T)

  return(dt)
}

#' Generic method to read ADM Datamart data.
#'
#' Method is very flexible in the arguments. It can take a \code{data.table}
#' for the model data and the predictor data, but the arguments can also be
#' pointing to files in CSV, zipped CSV, (LD)JSON, parquet or dataset export formats.
#'
#' @param modeldata Location, reference or the actual model table from the
#' ADM datamart. If not given defaults to the name of the dataset export file
#' for the datamart model table. To not use it at all, set to FALSE.
#' @param predictordata Location, reference or the actual predictor binning
#' table from the ADM datamart. If not given defaults to the name of the dataset export file
#' for the datamart predictor binning table. To not use at all, set to FALSE.
#' @param folder Optional path for the folder in which to look for the model
#' and predictor data. If first two arguments are not given will try to
#' interpret the modeldata argument as the folder name and use the default
#' dataset export file names.
#' @param cleanupHookModelData Optional cleanup function to call directly after the
#' raw model data has been read from the source file. This is especially helpful
#' to clean up data that has been custom exported from a database into an
#' Excel or CSV file.
#' @param cleanupHookPredictorData Optional cleanup function to call directly after the
#' raw predictor data has been read from the source file. This is especially helpful
#' to clean up data that has been custom exported from a database into an
#' Excel or CSV file.
#' @param keepSerializedModelData By default the serialized model data is left
#' out from the model data as this typically is large and only needed for
#' specific use cases.
#' @param filterModelData Post processing filter on the models, defaults to no
#' filtering but a custom function can be provided to drop e.g. certain
#' channels or issues or perform other clean up activities.
#' @param filterPredictorData Post processing filter on the predictor data. Defaults
#' to no filtering. Filtering on model ID's present in the model data will be
#' performed regardless. Potentially useful is to supply \code{filterLatestSnapshotOnly},
#' so to keep only the predictor data of the last snapshot - if multiple are
#' present.
#' @param predictorCategorization When passed in, a function that determines
#' the category of a predictor. This is then used to set an extra field in
#' the predictor data: "PredictorCategory" that holds the predictor category.
#' This is useful to globally set the predictor categories instead of having
#' to pass it to every individual plot function. The function does not need
#' to be vectorized. It is just applied to the levels of a factor and should
#' take just a character as an argument. Defaults to a function that returns the text
#' before the first dot.
#' @param tmpFolder Optional folder to store unzipped data (defaults to a temp folder)
#' @param verbose Flag for verbose logging, defaults to FALSE.
#'
#' @return A \code{list}) with two elements: "modeldata" contains a clean
#' \code{data.table} with the ADM model data, and "predictordata" contains a
#' clean \code{data.table} with the ADM predictor snapshots.
#' @export
#'
#' @examples
#' \dontrun{
#'   datamart <- ADMDatamart("~/Downloads")
#'
#'   datamart <- ADMDatamart("models.csv", "predictors.csv", folder="adm")
#' }
ADMDatamart <- function(modeldata = NULL,
                        predictordata = NULL,
                        folder = ".",
                        cleanupHookModelData = identity,
                        cleanupHookPredictorData = identity,
                        keepSerializedModelData = F,
                        filterModelData = identity,
                        filterPredictorData = identity,
                        predictorCategorization = defaultPredictorCategorization,
                        tmpFolder=tempdir(check = T),
                        verbose = F)
{
  ModelData <- SuccessRate <- Positives <- ResponseCount <- AUC <-
    Performance <- PredictorName <- ModelID <- EntryType <- Propensity <-
    BinPositives <- BinResponseCount <- BinIndex <- GroupIndex <- NULL # Trick to silence warnings from R CMD Check

  # If first arg is a folder then ignore the folder arg and reset modeldata to default.
  if (!is.data.table(modeldata) && !is.data.frame(modeldata) && !is.logical(modeldata)) {
    if (!is.null(modeldata) && dir.exists(modeldata)) {
      folder = modeldata
      modeldata <- NULL
    }
  }

  # If second arg is a folder assume there is no predictor data
  if (!is.data.table(predictordata) && !is.data.frame(predictordata) && !is.logical(predictordata)) {
    if (!is.null(predictordata) && dir.exists(predictordata)) {
      folder = predictordata
      predictordata <- F
    }
  }

  # Defaults assume model and predictor data are a dataset export
  if (is.null(modeldata)) {
    modeldata <- "Data-Decision-ADM-ModelSnapshot_(py)?ModelSnapshots"
  }
  if (is.null(predictordata)) {
    predictordata <- "Data-Decision-ADM-PredictorBinningSnapshot_(py)?ADMPredictorSnapshots"
  }

  # Define fields for model and predictor tables that we require, fields that are
  # added, and fields that are optional - typically added in later releases.
  # This is used to subset the data to just those fields and verify that all
  # required fields are present. This is especially helpful when dealing with
  # custom dataset exports from the database, via CSV etc.
  requiredModelFields <- c("ModelID", "ConfigurationName","SnapshotTime",
                           "Issue", "Group", "Name", "Direction", "Channel",
                           "TotalPredictors", "ActivePredictors", "Negatives", "Positives", "ResponseCount",
                           "Performance", "ResponseCount")
  optionalModelFields <- c("ModelData", "ModelVersion") # added in later releases
  additionalModelFields <- c("AUC", "SuccessRate") # additional, not in Datamart
  dropModelFields <- function(x, dropSerializedModelData) {
    # Hard-coded list because the names of (x) may contain just about anything
    # after the JSON expansion from the potentially customized context fields
    # in Name.
    dropFields <- c("AppliesToClass", "Application", "PerformanceError", "FactoryUpdateTime",
                    "InsName", "SaveDateTime", "InsKey", "CommitDateTime",
                    "RelativeNegatives", "RelativePositives", "RelativeResponseCount", # non-standard but crept in into our datasets
                    "Memory", "PerformanceThreshold", "CorrelationThreshold") # non-standard but crept in into our datasets
    if (dropSerializedModelData) {
      dropFields <- c(dropFields, c("ModelData", "ModelVersion"))
    }
    return(intersect(names(x), dropFields))
  }

  requiredPredictorFields <- c("ModelID", "PredictorName", "Performance", "SnapshotTime",
                               "Type", "EntryType",
                               "ResponseCount", "Positives", "Negatives", "TotalBins")
  optionalPredictorFields <- c(
    # bin level:
    "BinType", "BinSymbol", "BinIndex", "BinNegatives", "BinPositives", "BinResponseCount", "BinLowerBound", "BinUpperBound", "BinSymbol", "Lift", "ZRatio",
    # added in later versions:
    "GroupIndex", "FeatureImportance",
    # additional, not in Datamart:
    "Propensity")
  additionalPredictorFields <- c("AUC", "PredictorCategory") # additional, not in Datamart
  dropPredictorFields <- function(x) {
    return(setdiff(names(x), c(requiredPredictorFields, optionalPredictorFields, additionalPredictorFields)))
  }

  # Read models

  if (isFALSE(modeldata)) {
    modelz <- NULL
  } else {
    if (is.data.table(modeldata) | is.data.frame(modeldata)) {
      if (is.data.table(modeldata)) {
        modelz <- modeldata
      } else {
        modelz <- as.data.table(modeldata)
      }
    } else {
      modelz <- readFromSource(modeldata, folder, tmpFolder)
    }

    modelz <- cleanupHookModelData(modelz)
    modelz <- dropInternalDatamartFields(modelz)
    standardizeFieldCasing(modelz)
    modelz <- fixDatamartFieldTypes(modelz)
    modelz <- filterModelData(modelz)
    modelz <- expandEmbeddedJSONContext(modelz)

    if (nrow(modelz) > 0) {
      # TODO: generalize below
      doNotFactorizeFields <- c("ModelID")
      for (f in setdiff(names(modelz)[sapply(modelz, is.character)], doNotFactorizeFields)) {
        modelz[[f]] <- factor(modelz[[f]])
      }
      for (f in doNotFactorizeFields) {
        modelz[[f]] <- as.character(modelz[[f]])
      }

      # Add fields for common calculations

      modelz[["SuccessRate"]] <- modelz$Positives / modelz$ResponseCount # Using set syntax to avoid data.table copy warning
      modelz[["AUC"]] <- 100*modelz$Performance

      # Remove columns not used in CDH Tools
      # TODO: consider doing much earlier but then be careful with inconsistent naming
      if (length(dropModelFields(modelz, !keepSerializedModelData)) > 0) {
        if (verbose) warning("Dropping model fields:", paste(dropModelFields(modelz, !keepSerializedModelData), collapse=", "))

        modelz[, dropModelFields(modelz, !keepSerializedModelData) := NULL]
      }

      # Assert presence of required + additional fields
      notPresentFields <- setdiff(c(requiredModelFields, additionalModelFields), names(modelz))
      if (length(notPresentFields) > 0) {
        stop(paste("Not all required model fields present. Missing:", paste(notPresentFields, collapse=", ")))
      }
      if (length(names(modelz)) != length(unique(names(modelz)))) {
        stop(paste("Duplicate model fields found in:", paste(names(modelz), collapse=", ")))
      }
    } else {
      modelz <- NULL
    }
  }

  # Read Predictor data

  if (isFALSE(predictordata)) {
    predz <- NULL
  } else {
    if (is.data.table(predictordata) | is.data.frame(predictordata)) {
      if (is.data.table(predictordata)) {
        predz <- predictordata
      } else {
        predz <- as.data.table(predictordata)
      }
    } else {
      predz <- readFromSource(predictordata, folder, tmpFolder)
    }

    predz <- cleanupHookPredictorData(predz)
    predz <- dropInternalDatamartFields(predz)
    standardizeFieldCasing(predz)
    predz <- fixDatamartFieldTypes(predz)

    if (!is.null(modelz)) {
      predz <- predz[ModelID %in% modelz$ModelID]
    }
    predz <- filterPredictorData(predz)

    if (nrow(predz) > 0) {
      # TODO: generalize below
      doNotFactorizeFields <- intersect(c("ModelID", "BinSymbol"), names(predz))
      for (f in setdiff(names(predz)[sapply(predz, is.character)], doNotFactorizeFields)) {
        predz[[f]] <- factor(predz[[f]])
      }
      for (f in doNotFactorizeFields) {
        predz[[f]] <- as.character(predz[[f]])
      }

      # Apply predictor categorization if not present already
      if (!("PredictorCategory" %in% names(predz))) {
        predCategories <- data.table( PredictorName = factor(levels(predz$PredictorName), levels = levels(predz$PredictorName)),
                                      PredictorCategory = factor(sapply(levels(predz$PredictorName), predictorCategorization)))
        predz <- merge(predz, predCategories, by="PredictorName")
      }

      # Add fields for common calculations

      if ("BinPositives" %in% names(predz) && "BinResponseCount" %in% names(predz)) {
        # Bin fields could have been filtered out when looking at the data w/o the binning for exampls
        predz[EntryType != "Classifier", Propensity := BinPositives / BinResponseCount]
        predz[EntryType == "Classifier", Propensity := (BinPositives+0.5) / (BinResponseCount+1)]
      }
      predz[, AUC := 100*Performance]

      # predictor grouping index was not always there, add it as just a sequence number when absent
      if (!("GroupIndex" %in% names(predz))) {
        predz[, GroupIndex := .GRP, by=PredictorName]
      }

      if ("BinIndex" %in% names(predz)) {
        # Bin fields could have been filtered out when looking at the data w/o the binning for exampls
        setorder(predz, -Performance, BinIndex)
      } else {
        setorder(predz, -Performance)
      }

      # Remove columns not used in CDH Tools
      # TODO: consider doing much earlier but then be careful with inconsistent naming
      if (length(dropPredictorFields(predz)) > 0) {
        if (verbose) warning("Dropping predictor fields:", paste(dropPredictorFields(predz), collapse=", "))

        predz[, dropPredictorFields(predz) := NULL]
      }

      # Assert presence of required + additional fields
      notPresentFields <- setdiff(c(requiredPredictorFields, additionalPredictorFields), names(predz))
      if (length(notPresentFields) > 0) {
        print(head(predz))
        stop(paste("Not all required predictor fields present. Missing:", paste(notPresentFields, collapse=", ")))
      }
      if (length(names(predz)) != length(unique(names(predz)))) {
        stop(paste("Duplicate predictor fields found in:", paste(names(predz), collapse=", ")))
      }
    } else {
      predz <- NULL
    }
  }


  return(list(modeldata = modelz,
              predictordata = predz))
}

#' Calculate variable importance from ADM Datamart data.
#'
#' Uses the predictor binning from the ADM Datamart data to find the
#' relative importance. The importance is given by the distance to the
#' average log odds, per predictor. Next to importance it also returns
#' the univariate predictor performance (AUC).
#'
#' @param datamart Data frame with the ADM datamart data. Model data is only
#'   used when there are facets, so the model part could in principle be NULL.
#' @param facets Optional list of names from the model data to aggregate
#' up to.
#' @param filter Optional filter for the predictor data. Defaults to filter
#' out the classifiers, so operates on both active and inactive predictors.
#' See examples.
#'
#' @return A \code{data.table}) with \code{PredictorName}, \code{Importance}
#' and \code{Rank} plus columns for each of the \code{facets} if supplied.
#' @export
#'
#' @examples
#' \dontrun{
#'   dm <- ADMDatamart("~/Downloads")
#'
#'   varimp <- admVarImp(dm)
#'
#'   varimp <- admVarImp(dm, filter = filterActiveOnly)
#' }
admVarImp <- function(datamart, facets = NULL, filter = function(x) {filterClassifierOnly(x, reverse=T)})
{
  EntryType <- BinLogOdds <- BinPositives <- BinNegatives <- AvgLogOdds <- NULL # Trick to silence warnings from R CMD Check
  BinResponseCount <- BinDiffLogOdds <- Performance <- Importance <- NULL
  ResponseCount <- ImportanceRank <- PerformanceRank <- NULL

  dmModels <- datamart$modeldata
  dmPredictors <- filter(datamart$predictordata)

  # Normalize field names
  standardizeFieldCasing(dmPredictors)

  # Log odds per bin, then the bin weight is the distance to the weighted mean
  bins <- dmPredictors[, c("BinPositives", "BinNegatives", "BinResponseCount", "Performance", "PredictorName", "PredictorCategory", "ModelID"), with=F]
  bins[, BinLogOdds := log(BinPositives+1) - log(BinNegatives+1)]
  bins[, AvgLogOdds := weighted.mean(BinLogOdds, BinResponseCount), by=c("PredictorName", "ModelID")]
  bins[, BinDiffLogOdds := abs(BinLogOdds - AvgLogOdds)]

  # The feature importance per predictor then is just the weighted average of these distances to the mean
  featureImportance <-
    bins[, list(Importance = weighted.mean(BinDiffLogOdds, BinResponseCount, na.rm=T),
                Performance = first(as.numeric(Performance)), # numeric cast should not be necessary but sometimes data is odd
                ResponseCount = sum(BinResponseCount)), by=c("PredictorName", "PredictorCategory", "ModelID")]
  featureImportance[, Importance := ifelse(is.na(Importance), 0, Importance)]


  # Now join in any facets - if there are any
  if (!is.null(facets)) {
    standardizeFieldCasing(dmModels)

    featureImportance <- merge(featureImportance,
                               unique(dmModels[, c(facets, "ModelID"), with=F]), by="ModelID")

    # Then aggregate up to the predictor names x facets
    aggregateFeatureImportance <- featureImportance[, list(Importance = weighted.mean(Importance, ResponseCount),
                                                           Performance = weighted.mean(Performance, ResponseCount),
                                                           ResponseCount = sum(ResponseCount)),
                                                    by=c("PredictorName", "PredictorCategory", facets)]

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

# TODO consider removing now

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
  ModelID <- name <- PredictorType <- NULL # Trick to silence R CMD Check warnings

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

