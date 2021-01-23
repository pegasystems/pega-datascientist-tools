# Utils to read the ADM Factory JSON into proper binning tables

ADMFACTORY_TABLE_SPLIT_SCHEMA <- "pegadata.pr_data_adm_factory"
ADMFACTORY_TABLE_SINGLE_SCHEMA <- "pr_data_adm_factory"

#' Converts a model partition (context) from a JSON representation to a string
#'
#' @param p Model partition as a JSON formatted string
#'
#' @return String representation of the context
#' @export
#'
#' @examples
#' library(jsonlite)
#' getJSONModelContextAsString( "{\"partition\":{\"Color\":\"Green\", \"Size\":\"120\"}}")
getJSONModelContextAsString <- function(p)
{
  partition <- (fromJSON(p))$partition
  flat <- paste(sapply(names(partition)[order(tolower(names(partition)))],
                       function(x){return(paste(tolower(x), partition[[x]], sep="_"))}), collapse = "_")
  return(flat)
}

buildIntervalNotation <- function(lower, upper) {
  return(ifelse(is.na(lower) & is.na(upper), "MISSING",
                ifelse(is.na(lower),paste0("<", upper),
                       ifelse(is.na(upper),paste0("\u2265", lower),
                              paste0("[",lower,", ",upper,">")))))
}

#' Retrieves ADM models from the internal "Factory" table.
#'
#' The models in the internal "Factory" table contain all the raw data that the
#' system needs to build the models from. It includes the raw binning, the
#' current data analysis, the score mapping and much more. The data can be used
#' to reconstruct the models from, similar to the ADM datamart data. The model
#' data here is however complete as it is all stored in JSOn while the datamart
#' may contain truncated values.
#'
#' @param conn Connection to the database
#' @param appliesto Optional filter on the \code{AppliesTo} class. Currently
#'   only filters using exact string match, not a regexp.
#' @param configurationname Optional filter on the ADM rule name. Currently only
#'   filters using exact string match, not a regexp.
#' @param verbose Set to \code{True} to show database queries.
#'
#' @return Returns a \code{data.table} with all the models. When inspecting
#'   this, be careful with the \code{pyfactory} column as this contains a
#'   possibly very large string with the model JSON data. You would typically
#'   want to exclude this when looking at an overview of all models. A single
#'   model instance from the table can be further inspected by calling the
#'   \code{fromJSON} function on the relevant \code{pyfactory} value.
#'
#' @export
#'
#' @examples
#' \dontrun{models <- getModelsFromJSONTable(conn)
#' myModel <- fromJSON(models$pyfactory[1])}
getModelsFromJSONTable <- function(conn, appliesto=NULL, configurationname=NULL, verbose=F)
{
  models <- NULL
  tryCatch( {
    factoryTable <- ADMFACTORY_TABLE_SPLIT_SCHEMA
    query <- paste("select pyconfigpartitionid,pyconfigpartition from", factoryTable)
    if(verbose) {
      print(query)
    }
    models <- as.data.table(dbGetQuery(conn, query))
  }, error=function(x){})
  if(is.null(models)) {
    tryCatch( {
      factoryTable <- ADMFACTORY_TABLE_SINGLE_SCHEMA
      query <- paste("select pyconfigpartitionid,pyconfigpartition from", factoryTable)
      if(verbose) {
        print(query)
      }
      models <- as.data.table(dbGetQuery(conn, query))
    }, error=function(x){print(x)})
  }
  if (is.null(models)) { return(data.table())}

  models[, pyClassName := sapply(models$pyconfigpartition, function(x) { return((fromJSON(x))$partition$pyClassName) })]
  models[, pyPurpose := sapply(models$pyconfigpartition, function(x) { return((fromJSON(x))$partition$pyPurpose) })]

  if (!is.null(appliesto)) {
    models <- models[pyClassName == appliesto ]
  }
  if (!is.null(configurationname)) {
    models <- models[pyPurpose == configurationname ]
  }

  query <- paste("select * from",
                 factoryTable,
                 "where pyconfigpartitionid in (",
                 paste(paste("'",unique(models$pyconfigpartitionid),"'",sep=""), collapse = ","),
                 ")")
  if(verbose) {
    print(query)
  }
  factories <- as.data.table(dbGetQuery(conn, query))

  return(factories)
}

getNumBinningFromJSON <- function(numField, id)
{
  labels <- unlist(numField$mapping$labels)
  intervals <- unlist(numField$mapping$intervals)
  if (labels[1] != "Missing") {
    print(numField)
    stop(paste("Assumption invalidated that first numeric bin is MISSING, field=", numField$name, "id=", id))
  }
  data.table( ModelID = id,
              PredictorName = numField$name,
              PredictorType = numField$type,
              BinLabel = NA, # binlabel not relevant for numerics
              BinLowerBound = c(NA, NA, intervals[rlang::seq2(2,length(intervals))]), # bin[1] is for missing values
              BinUpperBound = c(NA, intervals[rlang::seq2(2,length(intervals))], NA),
              BinType = c("MISSING", rep("INTERVAL", length(intervals))),
              BinPos = unlist(numField$positives),
              BinNeg = unlist(numField$negatives),
              BinIndex = c(0, seq(length(intervals))),
              TotalPos = numField$totalPositives,
              TotalNeg = numField$totalNegatives,
              Smoothing = numField$laplaceSmoothingValue)
}

getSymBinningFromJSON <- function(symField, id)
{
  labels <- unlist(symField$mapping$labels)
  mapping <- unlist(symField$mapping$values)
  symkeys <- unlist(symField$mapping$symbolicKeys)
  if (labels[1] != "Missing") {
    print(symField)
    stop(paste("Assumption invalidated that first symbolic bin is MISSING, field=", symField$name, "id=", id))
  }
  if (symkeys[1] != "$RemainingSymbols$" |
      symkeys[2] != "$ResidualGroup$") {
    print(symField)
    stop(paste("Assumptions invalidated about symbolic bin structure, field=", symField$name, "id=", id))
  }
  remainingSymbolsIdx <- mapping[1]
  # get the symbol list indices but exclude the first 2, and exclude any symbol that maps to the missing bin, or to
  # the remaining symbols bin
  normalValues <- setdiff(which(mapping != 0 & mapping != remainingSymbolsIdx), c(1,2))

  data.table( ModelID = id,
              PredictorName = symField$name,
              PredictorType = symField$type,
              BinLabel = c("Missing", symkeys[normalValues], "Other"),
              BinLowerBound = NA,
              BinUpperBound = NA,
              BinType = c("MISSING", rep("SYMBOL", length(normalValues)), "REMAININGSYMBOLS"),
              BinPos = unlist(symField$positives) [c(1, 1 + mapping[normalValues], 1 + remainingSymbolsIdx)],
              BinNeg = unlist(symField$negatives) [c(1, 1 + mapping[normalValues], 1 + remainingSymbolsIdx)],
              BinIndex = c(0, mapping[normalValues], remainingSymbolsIdx),
              TotalPos = symField$totalPositives,
              TotalNeg = symField$totalNegatives,
              Smoothing = symField$laplaceSmoothingValue)
}

internalBinningFromJSONFactory <- function(binningJSON, id, activePredsOnly)
{
  # assuming first of each active group is the active predictor
  activePreds <- list()
  if (length(binningJSON$predictorGroups$active) > 0) {
    activePreds <- sapply( binningJSON$predictorGroups$groups[binningJSON$predictorGroups$active,], function(x) {return(x[1])} )
  }

  # build binning for num and sym predictors, being defensive in case of no predictors
  predBinningTableNum <- list()
  if (length(binningJSON$groupedPredictors) > 0) {
    numPredictorIdx <- which(apply(binningJSON$groupedPredictors, 1, function(x) {return(x$type == "NUMERIC" & (x$name %in% activePreds | !activePredsOnly))}))
    if (length(numPredictorIdx) > 0) {
      predBinningTableNum <- rbindlist(lapply(numPredictorIdx, function(i) {
        b <- getNumBinningFromJSON(binningJSON$groupedPredictors[i,], id = id)
        b[, IsActive := PredictorName[1] %in% activePreds]
        return(b)}))
    }
  }

  predBinningTableSym <- list()
  if (length(binningJSON$groupedPredictors) > 0) {
    symPredictorIdx <- which(apply(binningJSON$groupedPredictors, 1, function(x) {return(x$type == "SYMBOLIC" & (x$name %in% activePreds | !activePredsOnly))}))
    if (length(symPredictorIdx) > 0) {
      predBinningTableSym <- rbindlist(lapply(symPredictorIdx, function(i) {
        b <- getSymBinningFromJSON(binningJSON$groupedPredictors[i,], id = id)
        b[, IsActive := PredictorName[1] %in% activePreds]
        return(b)}))
    }
  }

  lengthClassifierArrays <- length(binningJSON$outcomeProfile$positives) # Element [1] will not be used, would be for missing but contains no data

  classifierTable <- data.table( ModelID = id,
                                 PredictorName = "Classifier",
                                 PredictorType = "CLASSIFIER",
                                 BinLabel = NA, # label not relevant for classifier
                                 BinLowerBound = c(NA, unlist(binningJSON$outcomeProfile$mapping$intervals[seq_len(lengthClassifierArrays-2)+1])),
                                 BinUpperBound = c(unlist(binningJSON$outcomeProfile$mapping$intervals[seq_len(lengthClassifierArrays-2)+1]), NA),
                                 BinType = rep("INTERVAL", lengthClassifierArrays-1),
                                 BinPos = unlist(binningJSON$outcomeProfile$positives[2:lengthClassifierArrays]),
                                 BinNeg = unlist(binningJSON$outcomeProfile$negatives[2:lengthClassifierArrays]),
                                 BinIndex = 1:(lengthClassifierArrays-1),
                                 TotalPos = binningJSON$outcomeProfile$totalPositives,
                                 TotalNeg = binningJSON$outcomeProfile$totalNegatives,
                                 Smoothing = NA, # not using binningJSON$outcomeProfile$laplaceSmoothingValue
                                 IsActive = T)

  predBinningList <- list(predBinningTableNum, predBinningTableSym, classifierTable)
  predBinningTable <- rbindlist(predBinningList[lengths(predBinningList)>0]) # exclude the NULL or empty lists

  # Performance calculated from outcome profile to be compatible with DM tables where there is no performance profile
  predBinningTable$Performance <- auc_from_bincounts(binningJSON$outcomeProfile$positives,
                                                     binningJSON$outcomeProfile$negatives)

  predBinningTable$BinType <- factor(predBinningTable$BinType, levels=BINTYPES)
  setorder(predBinningTable, PredictorName, BinType, BinUpperBound, BinLabel, na.last = T)

  return(predBinningTable)
}

#' Build a scoring model from the "analyzedData" JSON string that is part of the
#' JSON Factory.
#'
#' This part of the ADM Factory string is the only piece that is necessary for
#' scoring. The scoring model itself is currently just a binning table in
#' internal format. We should see if this needs to be augmented to make it more
#' of a score card.
#'
#' @param analyzedData The JSON string with the analyzed (scoring) portion of
#'   the ADM Factory JSON.
#' @param isAuditModel When \code{True} the data comes from the
#'   full-auditability blob in the datamart and only contains active predictors.
#'   When \code{False} then it will filter the data for only active predictors.
#'
#' @return A \code{data.table} with the binning.
#' @export
getScoringModelFromJSONFactoryString <- function(analyzedData, isAuditModel=F)
{
  buildLabel <- function(PredictorType, BinType, BinLabel, BinLowerBound, BinUpperBound)
  {
    return(ifelse(BinType == "MISSING","MISSING",
                  ifelse(BinType == "INTERVAL",buildIntervalNotation(BinLowerBound, BinUpperBound),
                         ifelse(BinType == "SYMBOL",BinLabel,
                                ifelse(BinType == "REMAININGSYMBOLS", "Other", paste("Internal error: unknown type", BinType))))))
  }

  scoringModelJSON <- fromJSON(analyzedData)
  id <- getJSONModelContextAsString(toJSON(scoringModelJSON$factoryKey$modelPartition))

  internalBinning <- internalBinningFromJSONFactory(scoringModelJSON, id, activePredsOnly=!isAuditModel)

  # the internal binning is not a scorecard yet so relabel some of the columns and split
  # into scorecard and a score mapping

  scaledBinning <- setBinWeights(internalBinning)$binning
  scaledBinning[, Label := buildLabel(PredictorType, BinType, BinLabel, BinLowerBound, BinUpperBound)]

  # The field to score mapping (also returning pos/neg so numbers can be verified)
  scorecard <- scaledBinning[PredictorType != "CLASSIFIER",
                             c("PredictorName", "Label", "binWeight", "BinPos", "BinNeg")]
  setnames(scorecard, c("Field", "Value", "Points", "pos", "neg"))

  # The classifier mapping
  scorecardMapping <- scaledBinning[PredictorType == "CLASSIFIER",
                                    c("Label", "binWeight", "BinPos", "BinNeg")]
  setnames(scorecardMapping, c("Score Range", "Propensity", "pos", "neg"))

  # Both plus the original boundaries/labels
  oriBinning <- scaledBinning[, c("PredictorName", "PredictorType", "BinType", "BinLabel", "BinLowerBound", "BinUpperBound", "binWeight", "BinPos", "BinNeg")]
  setnames(oriBinning, c("Field", "FieldType", "BinType", "Symbol", "LowerBound", "UpperBound", "Weight", "pos", "neg"))

  return(list(scorecard = scorecard, mapping = scorecardMapping, binning = oriBinning))
}

createListFromSingleJSONFactoryString <- function(aFactory, id, overallModelName, tmpFolder=NULL, forceLowerCasePredictorNames=F, activePredsOnly=T)
{
  model <- fromJSON(aFactory)

  # Dump JSON for debugging
  if (!is.null(tmpFolder)) {
    jsonDumpFileName <- paste(tmpFolder, paste(overallModelName, "json", id, "json", sep="."), sep="/")
    sink(file = jsonDumpFileName, append = F)
    cat(aFactory)
    sink()
  }

  predBinningTable <- internalBinningFromJSONFactory(model$analyzedData, id, activePredsOnly)

  if (forceLowerCasePredictorNames) {
    predBinningTable[, PredictorName := tolower(PredictorName)]
  }

  # Dump binning for debugging
  if (!is.null(tmpFolder)) {
    binningDumpFileName <- paste(tmpFolder, paste(overallModelName, "json", id, "csv", sep="."), sep="/")
    utils::write.csv(predBinningTable, file=binningDumpFileName, row.names = F)

    contextDumpFileName <- paste(tmpFolder, paste(overallModelName, "json", id, "txt", sep="."), sep="/")
    sink(file = contextDumpFileName, append = F)
    print(model$factoryKey$modelPartition$partition)
    sink()
  }

  # Check lowercase (for tests)
  currentContext <- model$factoryKey$modelPartition$partition
  if (forceLowerCasePredictorNames) {
    names(currentContext) <- tolower(names(currentContext))
  }

  return(list("context"=currentContext,
              "binning"=predBinningTable))
}

createListFromADMFactory <- function(partitions, overallModelName, tmpFolder=NULL, forceLowerCasePredictorNames=F)
{
  modelList <- lapply(partitions$pymodelpartitionid,
                      function(x) {
                        return(createListFromSingleJSONFactoryString(partitions$pyfactory[which(partitions$pymodelpartitionid==x)],
                                                                     x,
                                                                     overallModelName,
                                                                     tmpFolder,
                                                                     forceLowerCasePredictorNames)) })
  names(modelList) <- partitions$pymodelpartitionid
  return(modelList)
}

#' Turns an ADM Factory JSON into an easy-to-process binning table.
#'
#' It turns the ADM Factory JSON in the exact same format as the datamart binning table which includes aggregating
#' from the internal sub-bin model (to support individual field values) back to aggregate values.
#'
#' @param factoryJSON JSON string for a single model, usually the pyfactory field from the table returned by getModelsFromJSONTable.
#' @param modelname Optional name for the model, filled in into the modelid field of the binning table.
#'
#' @return A data.table containing the flattened-out predictor binning.
#' @export
#'
#' @examples
#' \dontrun{admFactory <- getModelsFromJSONTable(conn)
#' binning <- admJSONFactoryToBinning(admFactory$pyfactory[1])}
admJSONFactoryToBinning <- function(factoryJSON, modelname="Dummy")
{
  factoryDetail <- createListFromSingleJSONFactoryString(factoryJSON,
                                                         id=modelname,
                                                         overallModelName=modelname,
                                                         tmpFolder=NULL,
                                                         forceLowerCasePredictorNames=F,
                                                         activePredsOnly=F)

  SnapshotTime <- toPRPCDateTime(lubridate::now())
  predictorBinning <- factoryDetail$binning [, .(PredictorType = ifelse(PredictorType[1]=="SYMBOLIC", "symbolic", "numeric"),
                                                 Type = ifelse(PredictorType[1]=="SYMBOLIC", "symbolic", "numeric"), # same as PredictorType
                                                 BinType = ifelse(BinType[1]=="MISSING", "MISSING", "EQUIBEHAVIOR"), # RESIDUAL DOES NOT SEEM TO MATTER

                                                 BinSymbol = ifelse(PredictorType[1]=="SYMBOLIC",
                                                                    paste(BinLabel, collapse = ","),
                                                                    buildIntervalNotation(BinLowerBound, BinUpperBound)),
                                                 BinLowerBound = first(BinLowerBound),
                                                 BinUpperBound = first(BinUpperBound),

                                                 EntryType = ifelse(PredictorType[1]=="CLASSIFIER", "Classifier", ifelse(IsActive[1], "Active", "Inactive")),
                                                 SnapshotTime = SnapshotTime,
                                                 BinPositives = first(BinPos),
                                                 BinNegatives = first(BinNeg)
                                                 # pypositives/pynegatives are aggregates at predictor level - can build but may not be needed
                                                 # pygroupindex would be nice but optional
                                                 # BinIndex normally starts at 1.. here always 0 for missings whether they exist or not... issue?
                                                 # pytotalbins is predictor level aggregate
  ), by=c("ModelID", "PredictorName", "BinIndex")]

  # label bin index 0 as MISSING by definition
  # but then drop the missing bins without evidence
  # and finally make sure the index always starts at 1
  predictorBinning[BinIndex==0, BinType := "MISSING"]
  predictorBinning <- predictorBinning[!(BinIndex == 0 & BinPositives == 0 & BinNegatives == 0)]
  predictorBinning[, BinIndex := ifelse(rep(any(BinIndex == 0),.N), BinIndex+1, BinIndex), by=c("ModelID", "PredictorName")]
  predictorBinning[, BinIndex := as.integer(BinIndex)]

  # recalculate auc, z-ratio and lift
  predictorBinning[, Performance := auc_from_bincounts(BinPositives, BinNegatives), by=c("ModelID", "PredictorName")]
  predictorBinning[, Lift := lift(BinPositives, BinNegatives), by=c("ModelID", "PredictorName")]
  predictorBinning[, ZRatio := zratio(BinPositives, BinNegatives), by=c("ModelID", "PredictorName")]
  predictorBinning[, ZRatio := ifelse(is.nan(ZRatio), 0, ZRatio)]

  # drop the empty last bins
  predictorBinning[, maxBinIndex := max(BinIndex), by=c("ModelID", "PredictorName")]
  predictorBinning <- predictorBinning[!(BinIndex == maxBinIndex & BinPositives == 0 & BinNegatives == 0)][, -"maxBinIndex"]

  # order can have been changed after the binindex manipulation above
  setorderv(predictorBinning, c("ModelID", "PredictorName", "BinIndex"))

  return(predictorBinning)
}
