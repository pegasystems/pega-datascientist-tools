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
#' library(jsonlite); getJSONModelContextAsString( "{\"partition\":{\"Color\":\"Green\", \"Size\":\"120\"}}")
getJSONModelContextAsString <- function(p)
{
  partition <- (fromJSON(p))$partition
  flat <- paste(sapply(names(partition)[order(tolower(names(partition)))],
                       function(x){return(paste(tolower(x), partition[[x]], sep="_"))}), collapse = "_")
  return(flat)
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
#' \dontrun{models <- getModelsFromJSONTable(conn); myModel <- fromJSON(models$pyfactory[1])}
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

getNumBinningFromJSON <- function(numField, id, modelname)
{
  labels <- unlist(numField$mapping$labels)
  intervals <- unlist(numField$mapping$intervals)
  if (labels[1] != "Missing") {
    print(numField)
    stop(paste("Assumption invalidated that first numeric bin is MISSING, field=", numField$name, "model=", modelname, "id=", id))
  }
  data.table( modelid = id,
              predictorname = numField$name,
              predictortype = numField$type,
              binlabel = NA, # binlabel not relevant for numerics
              binlowerbound = c(NA, NA, intervals[rlang::seq2(2,length(intervals))]), # bin[1] is for missing values
              binupperbound = c(NA, intervals[rlang::seq2(2,length(intervals))], NA),
              bintype = c("MISSING", rep("INTERVAL", length(intervals))),
              binpos = unlist(numField$positives),
              binneg = unlist(numField$negatives),
              totalpos = numField$totalPositives,
              totalneg = numField$totalNegatives,
              smoothing = numField$laplaceSmoothingValue)
}

getSymBinningFromJSON <- function(symField, id, modelname)
{
  labels <- unlist(symField$mapping$labels)
  mapping <- unlist(symField$mapping$values)
  symkeys <- unlist(symField$mapping$symbolicKeys)
  if (labels[1] != "Missing") {
    print(symField)
    stop(paste("Assumption invalidated that first symbolic bin is MISSING, field=", symField$name, "model=", modelname, "id=", id))
  }
  if (symkeys[1] != "$RemainingSymbols$" |
      symkeys[2] != "$ResidualGroup$") {
    print(symField)
    stop(paste("Assumptions invalidated about symbolic bin structure, field=", symField$name, "model=", modelname, "id=", id))
  }
  remainingSymbolsIdx <- mapping[1]
  # get the symbol list indices but exclude the first 2, and exclude any symbol that maps to the missing bin, or to
  # the remaining symbols bin
  normalValues <- setdiff(which(mapping != 0 & mapping != remainingSymbolsIdx), c(1,2))

  data.table( modelid = id,
              predictorname = symField$name,
              predictortype = symField$type,
              binlabel = c("Missing", symkeys[normalValues], "Other"),
              binlowerbound = NA,
              binupperbound = NA,
              bintype = c("MISSING", rep("SYMBOL", length(normalValues)), "REMAININGSYMBOLS"),
              binpos = unlist(symField$positives) [c(1, 1 + mapping[normalValues], 1 + remainingSymbolsIdx)],
              binneg = unlist(symField$negatives) [c(1, 1 + mapping[normalValues], 1 + remainingSymbolsIdx)],
              totalpos = symField$totalPositives,
              totalneg = symField$totalNegatives,
              smoothing = symField$laplaceSmoothingValue)
}

createListFromSingleJSONFactoryString <- function(aFactory, id, overallModelName, tmpFolder=NULL, forceLowerCasePredictorNames=F)
{
  model <- fromJSON(aFactory)

  # Dump JSON for debugging
  if (!is.null(tmpFolder)) {
    jsonDumpFileName <- paste(tmpFolder, paste(overallModelName, "json", id, "json", sep="."), sep="/")
    sink(file = jsonDumpFileName, append = F)
    cat(aFactory)
    sink()
  }

  binning <- model$analyzedData

  # assuming first of each active group is the active predictor
  activePreds <- list()
  if (length(binning$predictorGroups$active) > 0) {
    activePreds <- sapply( binning$predictorGroups$groups[binning$predictorGroups$active,], function(x) {return(x[1])} )
  }
  predBinningList <- lapply( activePreds, function(x) {return(binning$groupedPredictors[ which(binning$groupedPredictors$name == x),])})

  predBinningTableNum <- data.table()
  numPredictors <- lapply(predBinningList, function(x) {return(x$type == "NUMERIC")})
  if (do.call(sum, numPredictors) > 0) {
    predBinningTableNum <- rbindlist(lapply(predBinningList[as.logical(numPredictors)],
                                            getNumBinningFromJSON, id, modelPartitionFullName))
  }

  predBinningTableSym <- data.table()
  symPredictors <- lapply(predBinningList, function(x) {return(x$type == "SYMBOLIC")})
  if (do.call(sum, symPredictors) > 0) {
    predBinningTableSym <- rbindlist(lapply(predBinningList[as.logical(symPredictors)],
                                            getSymBinningFromJSON, id, modelPartitionFullName))
  }

  lengthClassifierArrays <- length(binning$outcomeProfile$positives) # Element [1] will not be used, would be for missing but contains no data

  classifierTable <- data.table( modelid = id,
                                 predictorname = "Classifier",
                                 predictortype = "CLASSIFIER",
                                 binlabel = NA, # label not relevant for classifier
                                 binlowerbound = c(NA, unlist(binning$outcomeProfile$mapping$intervals[seq_len(lengthClassifierArrays-2)+1])),
                                 binupperbound = c(unlist(binning$outcomeProfile$mapping$intervals[seq_len(lengthClassifierArrays-2)+1]), NA),
                                 bintype = rep("INTERVAL", lengthClassifierArrays-1),
                                 binpos = unlist(binning$outcomeProfile$positives[2:lengthClassifierArrays]),
                                 binneg = unlist(binning$outcomeProfile$negatives[2:lengthClassifierArrays]),
                                 totalpos = binning$outcomeProfile$totalPositives,
                                 totalneg = binning$outcomeProfile$totalNegatives,
                                 smoothing = NA) # not using binning$outcomeProfile$laplaceSmoothingValue

  predBinningTable <- rbindlist(list(predBinningTableNum, predBinningTableSym, classifierTable))

  # Performance calculated from outcome profile to be compatible with DM tables where there is no performance profile
  predBinningTable$performance <- auc_from_bincounts(binning$outcomeProfile$positives,
                                                     binning$outcomeProfile$negatives)

  if (forceLowerCasePredictorNames) {
    predBinningTable$predictorname <- tolower(predBinningTable$predictorname)

    names(model$factoryKey$modelPartition$partition) <- tolower(names(model$factoryKey$modelPartition$partition))
  }

  predBinningTable$bintype <- factor(predBinningTable$bintype, levels=BINTYPES)
  setorder(predBinningTable, predictorname, bintype, binupperbound, binlabel, na.last = T)

  # Dump binning for debugging
  if (!is.null(tmpFolder)) {
    binningDumpFileName <- paste(tmpFolder, paste(overallModelName, "json", id, "csv", sep="."), sep="/")
    write.csv(predBinningTable, file=binningDumpFileName, row.names = F)

    contextDumpFileName <- paste(tmpFolder, paste(overallModelName, "json", id, "txt", sep="."), sep="/")
    sink(file = contextDumpFileName, append = F)
    print(model$factoryKey$modelPartition$partition)
    sink()
  }

  return(list("context"=model$factoryKey$modelPartition$partition,
              "binning"=predBinningTable))
}

createListFromADMFactory <- function(partitions, overallModelName, tmpFolder=NULL, forceLowerCasePredictorNames=F)
{
  pmml <- lapply(partitions$pymodelpartitionid,
                 function(x) { return(createListFromSingleJSONFactoryString(partitions$pyfactory[which(partitions$pymodelpartitionid==x)], x, overallModelName, tmpFolder, forceLowerCasePredictorNames)) })
  names(pmml) <- partitions$pymodelpartitionid
  return(pmml)
}
