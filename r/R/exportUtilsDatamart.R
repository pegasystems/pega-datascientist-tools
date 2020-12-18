# Utility functions to read ADM datamart data.
#
# Part of it is to convert to sanitized binning data so it can easily be
# converted to PMML. The "sanitized" format (shared with sourcing from eg the
# Factory tables) follows the ADM datamart binning format but makes it even
# longer: for symbolic predictors each symbol gets its own row.
#
# Input can be from the ADM Datamart exports as well as through a DB query.


# product versions prior to 7.3.1: use pr_data_dm_admmart_pred_fact instead of pegadata.pr_data_dm_admmart_pred

DATAMART_MODELTABLE <- "pr_data_dm_admmart_mdl_fact"
DATAMART_PREDICTORTABLE <- "pr_data_dm_admmart_pred"

# Returns a descriptive string representation of the model context for easier debugging
getDMModelContextAsString <- function(partition)
{
  flat <- paste(sapply(names(partition)[order(tolower(names(partition)))],
                       function(x){return(paste(tolower(x), partition[[x]], sep="_"))}), collapse = "_")
  return(flat)
}

#' Retrieves ADM models from the ADM Datamart.
#'
#' Reads either all or a selection of the ADM models stored in the ADM datamart into a \code{data.table}. Filters
#' can be passed in to select a particular \code{AppliesTo} classes, particular ADM rules or particular applications.
#' By default the method gets only the most recent snapshots but you can obtain the full history too, if available. To
#' prevent possible OOM errors reading the full table, the retrieval will be batched.
#'
#' @param conn Connection to the database
#' @param appliesToFilter Optional filter on the \code{AppliesTo} class. Can be a list, matching is exact.
#' @param ruleNameFilter Optional filter on the ADM rule name. Can be a list, matching is exact.
#' @param applicationFilter Optional filter on the application field. Can be a list, matching is exact.
#' @param latestOnly Only return results for latest snapshot. Currently this works as a global filter, which is ok
#' because the snapshot agent will also snapshot all models wholesale. If and when that becomes more granular, we may
#' need to make this more subtle, so take the latest after grouping by pyconfigurationname, pyappliestoclass, pxapplication.
#' @param verbose Set to \code{TRUE} to show database queries.
#'
#' @return A \code{data.table} with the ADM model details. The names are lowercased so processing of the results is not dependent on DB type.
#' @export
#'
#' @examples
#' \dontrun{models <- readADMDatamartModelTable(conn)}
#' \dontrun{allModels <- readADMDatamartModelTable(conn, latestOnly = F)}
readADMDatamartModelTable <- function(conn, appliesToFilter=NULL, ruleNameFilter=NULL, applicationFilter=NULL, latestOnly = F, verbose=F)
{
  # Drop Pega internal fields and model data (if present - not all releases have that)
  query <- paste("select * from", DATAMART_MODELTABLE, "where false")
  if(verbose) {
    print(query)
  }
  fields <- names(dbGetQuery(conn, query))
  fields <- setdiff(fields[!grepl("^p[x|z]", fields)], "pymodeldata")

  wheres <- list()
  if(!is.null(appliesToFilter)) {
    wheres[["appliesToFilter"]] <- paste("pyappliestoclass IN (", paste(paste("'",appliesToFilter,"'",sep=""), collapse=","), ")", sep="")
  }
  if(!is.null(ruleNameFilter)) {
    wheres[["ruleNameFilter"]] <- paste("pyconfigurationname IN (", paste(paste("'",ruleNameFilter,"'",sep=""), collapse=","), ")", sep="")
  }
  if(!is.null(applicationFilter)) {
    wheres[["applicationFilter"]] <- paste("pxapplication IN (", paste(paste("'",applicationFilter,"'",sep=""), collapse=","), ")", sep="")
  }

  if (latestOnly) {
    batchConditions <- paste("pysnapshottime IN (select max(pysnapshottime) from", DATAMART_MODELTABLE, ")")
  } else {
    query <- paste("select pymodelid, count(*) as nsnapshots from", DATAMART_MODELTABLE,
                   ifelse(length(wheres) == 0, "", paste("where", paste(wheres, collapse = " and "))),
                   "group by pymodelid")
    if(verbose) {
      print(query)
    }
    modelidz <- as.data.table(dbGetQuery(conn, query))
    setnames(modelidz, toupper(names(modelidz)))
    modelidz[, fetchgroup := as.integer(factor((cumsum(NSNAPSHOTS)) %/% 20000))] # batch up this many unique model snapshots at a time

    batchConditions <- sapply(1:max(modelidz$fetchgroup),
           function(i) {return(paste("pymodelid IN (", paste(paste("'",modelidz[fetchgroup == i]$PYMODELID, "'", sep=""), collapse=","), ")"))})
  }

  allModels <- list()
  for (aBatchCondition in batchConditions) {
    wheres[["batch"]] <- aBatchCondition
    query <- paste("select", paste(fields, collapse=","), "from", DATAMART_MODELTABLE, "where", paste(wheres, collapse = " and "))
    if(verbose) {
      print(query)
    }
    allModels[[aBatchCondition]] <- as.data.table(dbGetQuery(conn, query))
  }

  modelz <- rbindlist(allModels)

  # Time conversion here because DB may read date/time in all kinds of formats
  if ("pysnapshottime" %in% names(result)) { result[, pysnapshottime := fasttime::fastPOSIXct(pysnapshottime)] }
  if ("pyfactoryupdatetime" %in% names(result)) { result[, pyfactoryupdatetime := fasttime::fastPOSIXct(pyfactoryupdatetime)] }

  modelz <- standardizeDatamartModelData(modelz, latestOnly=latestOnly)
  modelz <- expandJSONContextInNameField(modelz)

  return(modelz)
}

#' Retrieves predictor data from the ADM Datamart.
#'
#' Typically, predictor data for a certain set of models obtained from \code{getModelsFromDatamart}. It is possible to
#' retrieve all historical data but by default it only retrieves the most recent snapshot data.
#'
#' @param conn Connection to the database
#' @param modelids Optional list of model IDs to select from
#' @param latestOnly Only return results for latest snapshot. Currently this works as a global filter, which is ok
#' because the snapshot agent will also snapshot all models wholesale. If and when that becomes more granular, we may
#' need to make this more subtle, so take the latest after grouping by model ID.
#' @param verbose Set to \code{TRUE} to show database queries.
#'
#' @return A \code{data.table} with the ADM predictor details. The names are lowercased so processing of the results is not dependent on DB type.
#' @export
#'
#' @examples
#' \dontrun{models <- readADMDatamartModelTable(conn)
#' preds <- readADMDatamartPredictorTable(conn, models$ModelID)}
readADMDatamartPredictorTable <- function(conn, modelids = NULL, latestOnly=T, verbose=T)
{
  # Drop Pega internal fields
  query <- paste("select * from", DATAMART_PREDICTORTABLE, "where false")
  if(verbose) {
    print(query)
  }
  fields <- names(dbGetQuery(conn, query))
  fields <- fields[!grepl("^p[x|z]", fields)]

  # TODO implement some sort of batching

  wheres <- list()
  if(!is.null(modelids)) {
    wheres[["models"]] <- paste("pymodelid IN (",
                                paste(paste("'",unique(modelids),"'",sep=""), collapse = ","),
                                ")")
  }
  if (latestOnly) {
    wheres[["latestOnly"]] <- paste("pysnapshottime IN (",
                                        "select max(pysnapshottime) from",
                                        DATAMART_PREDICTORTABLE)
    if (!is.null(modelids)) {
      wheres[["latestOnly"]] <- paste(wheres[["latestOnly"]],
                                          "where",
                                          wheres[["models"]])
    }
    wheres[["latestOnly"]] <- paste(wheres[["latestOnly"]], ")", sep="")
  }

  query <- paste("select", paste(fields, collapse = ","),
                 "from", DATAMART_PREDICTORTABLE,
                 "where", paste(wheres, collapse = " and "))
  if(verbose) {
    print(query)
  }
  predictors <- as.data.table(dbGetQuery(conn, query))

  applyUniformPegaFieldCasing(predictors)

  #lastsnapshots <- predictors[, .SD[which(pysnapshottime == max(pysnapshottime))], by=c("pymodelid")]
  return(predictors)
}

# Turns a single model definition into a list of key value pairs with the context key values
# if there are keys outside of the standard set, ADM will have encoded these as JSON strings
# in pyName - so pyName is checked for being a JSON string and if so, peeled apart and used
# in combination with any other keys
getContextKeyValuesFromDatamart <- function(aModel, asLowerCase)
{
  stdKeys <- c("pyIssue","pyGroup","pyName","pyChannel","pyDirection","pyTreatment")
  lowercaseKeys <- setdiff(intersect(tolower(stdKeys), tolower(names(aModel))), tolower(names(which(sapply(aModel, is.na))))) # always lowercase

  flexKeys <- list()
  pyNameIdx <- which(tolower(names(aModel)) == "pyname")
  if (length(pyNameIdx)==1) {
    if (startsWith(aModel[[pyNameIdx]], "{")) {
      try (
        flexKeys <- fromJSON(aModel[[pyNameIdx]]),
        silent = T
      )
    }
  }

  # force the keys to become lowercase, this is used for unittesting
  if (asLowerCase) { names(flexKeys) <- tolower(names(flexKeys)) }

  if (length(flexKeys) > 0) {
    lowercaseKeys <- setdiff(lowercaseKeys, "pyname")
  }
  l <- c(flexKeys, lapply(lowercaseKeys, function(x){ return(aModel[[which(tolower(names(aModel))==x)]]) }))
  names(l) <- c(names(flexKeys), lowercaseKeys)

  return(l)
}

getPredictorTypeFromDatamart <- function(dmbin)
{
  if (dmbin["pyentrytype"]=="Classifier") { return ("CLASSIFIER") }
  if (dmbin["pytype"]=="numeric") { return ("NUMERIC") }
  if (dmbin["pytype"]=="symbolic") { return ("SYMBOLIC") }

  print(dmbin)
  stop("Unknown PredictorType")
}

getBinTypeFromDatamart <- function(dmbin)
{
  if (dmbin["pyentrytype"]=="Classifier") { return ("INTERVAL") }
  if (dmbin["pybintype"]=="MISSING") { return ("MISSING") }
  if (dmbin["pybintype"]=="RESIDUAL") { return ("SYMBOL") } # Treat "Residual" bin like any other symbol
  if (dmbin["pybintype"]=="EQUIBEHAVIOR" & dmbin["pytype"] == "numeric") { return ("INTERVAL") }
  if (dmbin["pybintype"]=="EQUIBEHAVIOR" & dmbin["pytype"] == "symbolic" & dmbin["pybinsymbol"] == "Remaining symbols") { return ("REMAININGSYMBOLS") } # dm export doesnt tag remaining symbol group appropriately
  if (dmbin["pybintype"]=="EQUIBEHAVIOR" & dmbin["pytype"] == "symbolic") { return ("SYMBOL") }

  print(dmbin)
  stop("Unknown BinType")
}

# Santitize the predictor table so it contains a MISSING bin, for symbolics a REMAININGSYMBOLS bin,
# and add missing information like smoothing factor etc.
getPredictorDataFromDatamart <- function(dmbinning, id, overallModelName, tmpFolder=NULL)
{
  dmbinning <- dmbinning[which(pysnapshottime == max(pysnapshottime) & (pyentrytype == "Active" | pyentrytype == "Classifier"))]

  # explicit mapping so we can set types as well as do some value mapping
  binning <- data.table( modelid = dmbinning$pymodelid,
                         predictorname = dmbinning$pypredictorname,
                         predictortype = apply(dmbinning, 1, getPredictorTypeFromDatamart),
                         binlabel = dmbinning$pybinsymbol,
                         binlowerbound  = as.numeric(dmbinning$pybinlowerbound),
                         binupperbound = as.numeric(dmbinning$pybinupperbound),
                         bintype = apply(dmbinning, 1, getBinTypeFromDatamart),
                         binpos = as.numeric(dmbinning$pybinpositives),
                         binneg = as.numeric(dmbinning$pybinnegatives),
                         binidx = as.integer(dmbinning$pybinindex),
                         totalpos = as.numeric(dmbinning$pypositives),
                         totalneg = as.numeric(dmbinning$pynegatives),
                         smoothing = as.numeric(NA),
                         isactive = T, # NB currently this only lists the active predictors - subset above
                         performance = as.numeric(dmbinning$pyperformance[dmbinning$pyentrytype=="Classifier"][1])) # performance is of the model, not of the individual predictors

  # Sanitize the representation

  binning[bintype == "MISSING", binlabel := "Missing"] # for consistency
  binning[bintype == "REMAININGSYMBOLS", binlabel := "Other"] # for consistency

  # If there are missings, their binidx should start at 0
  binning[, binidx := binidx - ifelse(any(bintype == "MISSING"), 1L, 0L), by=c("modelid", "predictorname")]

  # - change num intervals to use NA to indicate inclusiveness of bounds
  binning[predictortype != "SYMBOLIC" & bintype != "MISSING",
          c("binlowerbound", "binupperbound") :=
            list(ifelse(binidx==min(binidx),as.numeric(NA),binlowerbound),
                 ifelse(binidx==max(binidx),as.numeric(NA),binupperbound)),
          by=c("modelid", "predictorname")]

  # - add "smoothing" field in the exact same manner that ADM does it (see ISSUE-22141)

  # legacy behavior (of the previous DM-only version of the PMML converter) - not 100% correct results
  # binning[predictortype != "CLASSIFIER", smoothing := 1/.N, by=c("modelid", "predictorname")]

  # this seems to be the right way (only count the nr of bins at training time) - which actually might be exactly the same as the above
  # binning[predictortype != "CLASSIFIER", smoothing := 1/(sum((binpos+binneg) > 0)), by=c("modelid", "predictorname")]

  # but this seems to work: ADM apparently always includes the MISSING and symbolic WOS bin. So we do the same here.
  binning[predictortype != "CLASSIFIER", smoothing := 1/(.N+
                                                           # add 1 for missing bin unless that is already present
                                                           1-as.integer(any(bintype=="MISSING"))+
                                                           # add 1 for symbolic remaining unless already present
                                                           as.integer(predictortype=="SYMBOLIC")-as.integer(predictortype=="SYMBOLIC" & any(bintype=="REMAININGSYMBOLS"))),
          by=c("modelid", "predictorname")]

  # - add MISSING bins when absent
  # - NOTE we set pos/neg to 0 for the newly created bins similar to what ADM does but this is debatable (ISSUE-22143)
  binningSummary <- binning[predictortype != "CLASSIFIER",
                            .(nMissing = sum(bintype=="MISSING"),
                              nRemaining = sum(bintype=="REMAININGSYMBOLS"),
                              maxBinIdx = max(c(0,binidx), na.rm = T)),
                            by=c("modelid", "predictorname", "predictortype", "totalpos", "totalneg", "smoothing", "isactive", "performance")]

  if(any(binningSummary$nMissing < 1)) {
    defaultsForMissingBin <- data.table( modelid = binning$modelid[1],
                                         binlabel = "Missing",
                                         binlowerbound=NA,
                                         binupperbound=NA,
                                         bintype="MISSING",
                                         binpos=0,
                                         binneg=0,
                                         binidx=0)
    missingMissings <- merge(binningSummary[nMissing==0],
                             defaultsForMissingBin,
                             all.x=T, by="modelid")[, BINNINGTABLEFIELDS, with=F]
  } else {
    missingMissings <- data.table()
  }

  # - add WOS/Remaining bin for symbolics when absent
  if(any(binningSummary$predictortype=="SYMBOLIC" & binningSummary$nRemaining < 1)) {
    defaultsForRemainingBin <- data.table( modelid = binning$modelid[1],
                                           binlabel = "Other",
                                           binlowerbound=NA,
                                           binupperbound=NA,
                                           bintype="REMAININGSYMBOLS",
                                           binpos=0,
                                           binneg=0,
                                           binidx=NA)
    missingRemainings <- merge(binningSummary[predictortype=="SYMBOLIC" & nRemaining==0],
                               defaultsForRemainingBin,
                               all.x=T, by="modelid")[, binidx := 1+maxBinIdx][, BINNINGTABLEFIELDS, with=F]
  } else {
    missingRemainings <- data.table()
  }

  # - split symbolic bins that seem to contain multiple symbols
  binning[, tmpNrSymbols:=1]
  binning[predictortype=="SYMBOLIC", tmpNrSymbols:=pmax(1,sapply(strsplit(binlabel,",",fixed=T), length))]
  binning[, tmpOriRowNo:=seq_len(.N)]
  binning <- binning[rep(tmpOriRowNo,tmpNrSymbols)]
  binning[tmpNrSymbols>1, tmpSymbolIdx:=seq_len(.N), by=tmpOriRowNo] # so num is the binlabel really and repidx the idx in the string parse
  binning[tmpNrSymbols>1, binlabel:=sapply(unlist(strsplit(binlabel, ",")),trimws) [tmpSymbolIdx], by=tmpOriRowNo]

  # - Combine all the newly added tables and make sure the order is set appropriately
  predBinningTable <- rbindlist(list(binning,missingMissings,missingRemainings), use.names = T, fill = T)[,BINNINGTABLEFIELDS,with=F]
  predBinningTable[predictortype %in% c("CLASSIFIER", "NUMERIC"), binlabel := NA] # binlabel not relevant for numerics or classifier
  predBinningTable$bintype <- factor(predBinningTable$bintype, levels=BINTYPES)
  setorder(predBinningTable, predictorname, bintype, binupperbound, binlabel, na.last = T)

  # Dump binning for debugging
  if (!is.null(tmpFolder)) {
    binningDumpFileName <- paste(tmpFolder, paste(overallModelName, "dm", id, "csv", sep="."), sep="/")
    utils::write.csv(predBinningTable, file=binningDumpFileName, row.names = F)

    # contextDumpFileName <- paste(tmpFolder, paste(overallModelName, "json", id, "txt", sep="."), sep="/")
    # sink(file = contextDumpFileName, append = F)
    # print(model$factoryKey$modelPartition$partition)
    # sink()
  }

  return(predBinningTable)
}

createListFromDatamart <- function(predictorsForPartition, fullName, tmpFolder=NULL, modelsForPartition=NULL, lowerCasePredictors = F)
{
  predictorsForPartition <- data.table(predictorsForPartition) # just to be sure
  setnames(predictorsForPartition, tolower(names(predictorsForPartition))) # export has proper case but db access often will not - settle on lowercase
  if (lowerCasePredictors) {
    predictorsForPartition[, pypredictorname := tolower(pypredictorname)]
  }

  if (!is.null(modelsForPartition)) {
    modelsForPartition <- as.data.table(modelsForPartition)
    setnames(modelsForPartition, tolower(names(modelsForPartition)))
    modelsForPartition <- modelsForPartition[pysnapshottime == max(pysnapshottime),] # max works because the way Pega date strings are represented

    modelsForPartition[["contextAsString"]] <- sapply(modelsForPartition$pymodelid, function(x)
      { return(getDMModelContextAsString(getContextKeyValuesFromDatamart(modelsForPartition[pymodelid==x,], lowerCasePredictors))) })

    modelList <- lapply(modelsForPartition$pymodelid, function(x){list("binning" = getPredictorDataFromDatamart(predictorsForPartition[pymodelid==x,],
                                                                                                                modelsForPartition[pymodelid==x,]$contextAsString,
                                                                                                                fullName, tmpFolder),
                                                                       "context" = getContextKeyValuesFromDatamart(modelsForPartition[pymodelid==x,], lowerCasePredictors))})
    names(modelList) <- modelsForPartition$pymodelid
  } else {
    modelList <- list(list("binning"=getPredictorDataFromDatamart(predictorsForPartition, "allbinning", fullName, tmpFolder)))
    names(modelList) <- fullName
  }

  return(modelList)
}

