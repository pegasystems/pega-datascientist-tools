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
  flat <- paste(sapply(names(partition)[order(names(partition))],
                       function(x){return(paste(x, partition[[x]], sep="_"))}), collapse = "_")
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
#' @return A \code{data.table} with the ADM model details. The names are
#' CamelCased in a standard way so processing of the results is not dependent
#' on DB type or export tooling.
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
    ModelIDz <- as.data.table(dbGetQuery(conn, query))
    setnames(ModelIDz, toupper(names(ModelIDz)))
    ModelIDz[, fetchgroup := as.integer(factor((cumsum(NSNAPSHOTS)) %/% 20000))] # batch up this many unique model snapshots at a time

    batchConditions <- sapply(1:max(ModelIDz$fetchgroup),
                              function(i) {return(paste("pymodelid IN (", paste(paste("'",ModelIDz[fetchgroup == i]$PYMODELID, "'", sep=""), collapse=","), ")"))})
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
#' @param ModelIDs Optional list of model IDs to select from
#' @param latestOnly Only return results for latest snapshot. Currently this works as a global filter, which is ok
#' because the snapshot agent will also snapshot all models wholesale. If and when that becomes more granular, we may
#' need to make this more subtle, so take the latest after grouping by model ID.
#' @param verbose Set to \code{TRUE} to show database queries.
#'
#' @return A \code{data.table} with the ADM predictor details. The names are
#' CamelCased in a standard way so processing of the results is not dependent
#' on DB type or export tooling.
#' @export
#'
#' @examples
#' \dontrun{models <- readADMDatamartModelTable(conn)
#' preds <- readADMDatamartPredictorTable(conn, models$ModelID)}
readADMDatamartPredictorTable <- function(conn, ModelIDs = NULL, latestOnly=T, verbose=T)
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
  if(!is.null(ModelIDs)) {
    wheres[["models"]] <- paste("pymodelid IN (",
                                paste(paste("'",unique(ModelIDs),"'",sep=""), collapse = ","),
                                ")")
  }
  if (latestOnly) {
    wheres[["latestOnly"]] <- paste("pysnapshottime IN (",
                                    "select max(pysnapshottime) from",
                                    DATAMART_PREDICTORTABLE)
    if (!is.null(ModelIDs)) {
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

  #lastsnapshots <- predictors[, .SD[which(SnapshotTime == max(SnapshotTime))], by=c("ModelID")]
  return(predictors)
}

# Turns a single model definition into a list of key value pairs with the context key values
# if there are keys outside of the standard set, ADM will have encoded these as JSON strings
# in pyName - so pyName is checked for being a JSON string and if so, peeled apart and used
# in combination with any other keys
getContextKeyValuesFromDatamart <- function(aModel, useLowercaseContextKeys=FALSE)
{
  stdKeys <- c("pyIssue","pyGroup","pyName","pyChannel","pyDirection","pyTreatment")

  # If they occur in the model data these will be standardized
  stdMdlKeys <- c("Issue","Group","Name","Channel","Direction","Treatment")

  # The context keys in use in the model
  usedKeys <- setdiff(intersect(stdMdlKeys, names(aModel)), names(which(sapply(aModel, is.na))))

  # Peel apart any JSON embedded fields in the Name data
  flexKeys <- list()
  nameIdx <- which(names(aModel) == "Name")
  if (length(nameIdx)==1) {
    if (startsWith(as.character(aModel[[nameIdx]]), "{")) {
      try (
        flexKeys <- fromJSON(aModel[[nameIdx]]),
        silent = T
      )
    }
  }
  if (length(flexKeys) > 0) {
    usedKeys <- setdiff(usedKeys, "Name")
  }

  l <- c(flexKeys, lapply(usedKeys, function(x){ return(aModel[[which(names(aModel)==x)]]) }))
  names(l) <- c(names(flexKeys), usedKeys)

  # get the py's back in front, as we're using them to match strategy result data
  names(l) <- as.character(sapply( names(l), function(x) { return(ifelse(x %in% stdMdlKeys, stdKeys[which(stdMdlKeys==x)], x))}))

  # lower case names used in test framework
  if (useLowercaseContextKeys) {
    names(l) <- tolower(names(l))
  }

  return(l)
}

getPredictorTypeFromDatamart <- function(dmbin)
{
  if (dmbin["EntryType"]=="Classifier") { return ("CLASSIFIER") }
  if (dmbin["Type"]=="numeric") { return ("NUMERIC") }
  if (dmbin["Type"]=="symbolic") { return ("SYMBOLIC") }

  print(dmbin)
  stop("Unknown PredictorType")
}

getBinTypeFromDatamart <- function(dmbin)
{
  if (dmbin["EntryType"]=="Classifier") { return ("INTERVAL") }
  if (dmbin["BinType"]=="MISSING") { return ("MISSING") }
  if (dmbin["BinType"]=="RESIDUAL") { return ("SYMBOL") } # Treat "Residual" bin like any other symbol
  if (dmbin["BinType"]=="EQUIBEHAVIOR" & dmbin["Type"] == "numeric") { return ("INTERVAL") }
  if (dmbin["BinType"]=="EQUIBEHAVIOR" & dmbin["Type"] == "symbolic" & dmbin["BinSymbol"] == "Remaining symbols") { return ("REMAININGSYMBOLS") } # dm export doesnt tag remaining symbol group appropriately
  if (dmbin["BinType"]=="EQUIBEHAVIOR" & dmbin["Type"] == "symbolic") { return ("SYMBOL") }

  print(dmbin)
  stop("Unknown BinType")
}

# Santitize the predictor table so it contains a MISSING bin, for symbolics a REMAININGSYMBOLS bin,
# and add missing information like Smoothing factor etc.
getPredictorDataFromDatamart <- function(dmbinning, id, overallModelName, tmpFolder=NULL)
{
  dmbinning <- dmbinning[which(SnapshotTime == max(SnapshotTime) & (EntryType == "Active" | EntryType == "Classifier"))]

  if (nrow(dmbinning) == 0) return(NULL) # defensive coding, but we have seen binnings without any active and/or classifier rows

  # explicit mapping so we can set types as well as do some value mapping
  binning <- data.table( ModelID = dmbinning$ModelID,
                         PredictorName = dmbinning$PredictorName,
                         PredictorType = apply(dmbinning, 1, getPredictorTypeFromDatamart),
                         BinLabel = dmbinning$BinSymbol,
                         BinLowerBound  = as.numeric(dmbinning$BinLowerBound),
                         BinUpperBound = as.numeric(dmbinning$BinUpperBound),
                         BinType = apply(dmbinning, 1, getBinTypeFromDatamart),
                         BinPos = as.numeric(dmbinning$BinPositives),
                         BinNeg = as.numeric(dmbinning$BinNegatives),
                         BinIndex = as.integer(dmbinning$BinIndex),
                         TotalPos = as.numeric(dmbinning$Positives),
                         TotalNeg = as.numeric(dmbinning$Negatives),
                         Smoothing = as.numeric(NA),
                         IsActive = T, # NB currently this only lists the active predictors - subset above
                         Performance = as.numeric(dmbinning$Performance[dmbinning$EntryType=="Classifier"][1])) # Performance is of the model, not of the individual predictors

  # Sanitize the representation

  binning[BinType == "MISSING", BinLabel := "Missing"] # for consistency
  binning[BinType == "REMAININGSYMBOLS", BinLabel := "Other"] # for consistency

  # If there are missings, their BinIndex should start at 0
  binning[, BinIndex := BinIndex - ifelse(any(BinType == "MISSING"), 1L, 0L), by=c("ModelID", "PredictorName")]

  # - change num intervals to use NA to indicate inclusiveness of bounds
  binning[PredictorType != "SYMBOLIC" & BinType != "MISSING",
          c("BinLowerBound", "BinUpperBound") :=
            list(ifelse(BinIndex==min(BinIndex),as.numeric(NA),BinLowerBound),
                 ifelse(BinIndex==max(BinIndex),as.numeric(NA),BinUpperBound)),
          by=c("ModelID", "PredictorName")]

  # - add "Smoothing" field in the exact same manner that ADM does it (see ISSUE-22141)

  # legacy behavior (of the previous DM-only version of the PMML converter) - not 100% correct results
  # binning[PredictorType != "CLASSIFIER", Smoothing := 1/.N, by=c("ModelID", "PredictorName")]

  # this seems to be the right way (only count the nr of bins at training time) - which actually might be exactly the same as the above
  # binning[PredictorType != "CLASSIFIER", Smoothing := 1/(sum((BinPos+BinNeg) > 0)), by=c("ModelID", "PredictorName")]

  # but this seems to work: ADM apparently always includes the MISSING and symbolic WOS bin. So we do the same here.
  binning[PredictorType != "CLASSIFIER", Smoothing := 1/(.N+
                                                           # add 1 for missing bin unless that is already present
                                                           1-as.integer(any(BinType=="MISSING"))+
                                                           # add 1 for symbolic remaining unless already present
                                                           as.integer(PredictorType=="SYMBOLIC")-as.integer(PredictorType=="SYMBOLIC" & any(BinType=="REMAININGSYMBOLS"))),
          by=c("ModelID", "PredictorName")]

  # - add MISSING bins when absent
  # - NOTE we set pos/neg to 0 for the newly created bins similar to what ADM does but this is debatable (ISSUE-22143)
  binningSummary <- binning[PredictorType != "CLASSIFIER",
                            .(nMissing = sum(BinType=="MISSING"),
                              nRemaining = sum(BinType=="REMAININGSYMBOLS"),
                              maxBinIndex = max(c(0,BinIndex), na.rm = T)),
                            by=c("ModelID", "PredictorName", "PredictorType", "TotalPos", "TotalNeg", "Smoothing", "IsActive", "Performance")]

  if(any(binningSummary$nMissing < 1)) {
    defaultsForMissingBin <- data.table( ModelID = binning$ModelID[1],
                                         BinLabel = "Missing",
                                         BinLowerBound=NA,
                                         BinUpperBound=NA,
                                         BinType="MISSING",
                                         BinPos=0,
                                         BinNeg=0,
                                         BinIndex=0)
    missingMissings <- merge(binningSummary[nMissing==0],
                             defaultsForMissingBin,
                             all.x=T, by="ModelID")[, BINNINGTABLEFIELDS, with=F]
  } else {
    missingMissings <- data.table()
  }

  # - add WOS/Remaining bin for symbolics when absent
  if(any(binningSummary$PredictorType=="SYMBOLIC" & binningSummary$nRemaining < 1)) {
    defaultsForRemainingBin <- data.table( ModelID = binning$ModelID[1],
                                           BinLabel = "Other",
                                           BinLowerBound=NA,
                                           BinUpperBound=NA,
                                           BinType="REMAININGSYMBOLS",
                                           BinPos=0,
                                           BinNeg=0,
                                           BinIndex=NA)
    missingRemainings <- merge(binningSummary[PredictorType=="SYMBOLIC" & nRemaining==0],
                               defaultsForRemainingBin,
                               all.x=T, by="ModelID")[, BinIndex := 1+maxBinIndex][, BINNINGTABLEFIELDS, with=F]
  } else {
    missingRemainings <- data.table()
  }

  # - split symbolic bins that seem to contain multiple symbols
  binning[, tmpNrSymbols:=1]
  binning[PredictorType=="SYMBOLIC", tmpNrSymbols:=pmax(1,sapply(strsplit(BinLabel,",",fixed=T), length))]
  binning[, tmpOriRowNo:=seq_len(.N)]
  binning <- binning[rep(tmpOriRowNo,tmpNrSymbols)]
  binning[tmpNrSymbols>1, tmpSymbolIdx:=seq_len(.N), by=tmpOriRowNo] # so num is the BinLabel really and repidx the idx in the string parse
  binning[tmpNrSymbols>1, BinLabel:=sapply(unlist(strsplit(BinLabel, ",")),trimws) [tmpSymbolIdx], by=tmpOriRowNo]

  # - Combine all the newly added tables and make sure the order is set appropriately
  predBinningTable <- rbindlist(list(binning,missingMissings,missingRemainings), use.names = T, fill = T)[,BINNINGTABLEFIELDS,with=F]
  predBinningTable[PredictorType %in% c("CLASSIFIER", "NUMERIC"), BinLabel := NA] # BinLabel not relevant for numerics or classifier
  predBinningTable$BinType <- factor(predBinningTable$BinType, levels=BINTYPES)
  setorder(predBinningTable, PredictorName, BinType, BinUpperBound, BinLabel, na.last = T)

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

normalizedBinningFromDatamart <- function(predictorsForPartition,
                                          fullName="",
                                          tmpFolder=NULL,
                                          modelsForPartition=NULL,
                                          useLowercaseContextKeys=FALSE)
{
  predictorsForPartition <- data.table(predictorsForPartition) # just to be sure

  if (!is.null(modelsForPartition)) {
    modelsForPartition <- as.data.table(modelsForPartition)

    modelsForPartition <- modelsForPartition[SnapshotTime == max(SnapshotTime),] # max works because the way Pega date strings are represented

    modelsForPartition[["contextAsString"]] <-
      sapply(modelsForPartition$ModelID,
             function(x) { return(getDMModelContextAsString(getContextKeyValuesFromDatamart(modelsForPartition[ModelID==x,]))) })

    modelList <-
      lapply(modelsForPartition$ModelID,
             function(x){list("binning" = getPredictorDataFromDatamart(predictorsForPartition[ModelID==x,],
                                                                       modelsForPartition[ModelID==x,]$contextAsString,
                                                                       fullName, tmpFolder),
                              "context" = getContextKeyValuesFromDatamart(modelsForPartition[ModelID==x,],
                                                                          useLowercaseContextKeys=useLowercaseContextKeys))})

    names(modelList) <- modelsForPartition$ModelID

    # remove the elements w/o any binning at all
    modelList <- modelList[sapply(modelList, function(x) {return(!is.null(x$binning) & !is.null(x$context))})]

  } else {
    modelList <- list(list("binning"=getPredictorDataFromDatamart(predictorsForPartition, "allbinning", fullName, tmpFolder)))
    names(modelList) <- fullName
  }

  return(modelList)
}

