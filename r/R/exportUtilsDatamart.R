# Utils to convert ADM datamart to sanitized binning data
# so it can be converted to PMML
# Checked mostly with ADM DB dataset exports, and straight Postgress DB access from 7.3.1

DATAMART_MODELTABLE <- "pegadata.pr_data_dm_admmart_mdl_fact"

# product versions prior to 7.3.1: use pr_data_dm_admmart_pred_fact instead of pegadata.pr_data_dm_admmart_pred
DATAMART_PREDICTORTABLE <- "pegadata.pr_data_dm_admmart_pred"

# Flat representation of the context string for easier debugging
getDMModelContextAsString <- function(partition)
{
  flat <- paste(sapply(names(partition)[order(tolower(names(partition)))], 
                       function(x){return(paste(tolower(x), partition[[x]], sep="_"))}), collapse = "_")
  return(flat)
}


getModelsForClassFromDatamart <- function(conn, appliesto=NULL, configurationname=NULL)
{
  wheres <- list()
  if(!is.null(appliesto)) {
    wheres[["appliesto"]] <- paste("pyappliestoclass=", "'", appliesto, "'", sep="")
  }
  if(!is.null(configurationname)) {
    wheres[["configurationname"]] <- paste("pyconfigurationname=", "'", configurationname, "'", sep="")
  }

  if (length(wheres) >= 1) {
    query <- paste("select * from", DATAMART_MODELTABLE, "where", paste(wheres, collapse = " and "))
  } else {
    query <- paste("select * from", DATAMART_MODELTABLE)
  }
  print(query)
  models <- as.data.table(dbGetQuery(conn, query))
  lastsnapshots <- models[, .SD[which(pysnapshottime == max(pysnapshottime))], by=c("pyconfigurationname", "pyappliestoclass", "pxapplication")]
  return(lastsnapshots)  
}

getPredictorsForModelsFromDatamart <- function(conn, models)
{
  query <- paste("select * from",
                 DATAMART_PREDICTORTABLE,
                 "where pymodelid in (",
                 paste(paste("'",models$pymodelid,"'",sep=""), collapse = ","),
                 ")")
  print(query)
  predictors <- as.data.table(dbGetQuery(conn, query))
  lastsnapshots <- predictors[, .SD[which(pysnapshottime == max(pysnapshottime))], by=c("pymodelid")]
  return(lastsnapshots)  
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
                         totalpos = as.numeric(dmbinning$pypositives),
                         totalneg = as.numeric(dmbinning$pynegatives),
                         smoothing = as.numeric(NA),
                         performance = as.numeric(dmbinning$pyperformance[dmbinning$pyentrytype=="Classifier"][1]), # performance is of the model, not of the individual predictors
                         tmpBinIndex = as.integer(dmbinning$pybinindex))
  
  # Sanitize the representation
  
  binning[bintype == "MISSING", binlabel := "Missing"] # for consistency
  binning[bintype == "REMAININGSYMBOLS", binlabel := "Other"] # for consistency
  
  # - change num intervals to use NA to indicate inclusiveness of bounds
  binning[predictortype != "SYMBOLIC" & bintype != "MISSING", 
          c("binlowerbound", "binupperbound") := 
            list(ifelse(tmpBinIndex==min(tmpBinIndex),as.numeric(NA),binlowerbound),
                 ifelse(tmpBinIndex==max(tmpBinIndex),as.numeric(NA),binupperbound)),
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
                            .(nMissing=sum(bintype=="MISSING"), 
                              nRemaining=sum(bintype=="REMAININGSYMBOLS")), 
                            by=c("modelid", "predictorname", "predictortype", "totalpos", "totalneg", "smoothing", "performance")]
  
  if(any(binningSummary$nMissing < 1)) {
    defaultsForMissingBin <- data.table( modelid = binning$modelid[1],
                                         binlabel = "Missing",
                                         binlowerbound=NA,
                                         binupperbound=NA,
                                         bintype="MISSING",
                                         binpos=0,
                                         binneg=0)
    missingMissings <- merge(binningSummary[nMissing==0], defaultsForMissingBin, all.x=T, by="modelid")[, BINNINGTABLEFIELDS, with=F]
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
                                           binneg=0)
    missingRemainings <- merge(binningSummary[predictortype=="SYMBOLIC" & nRemaining==0], defaultsForRemainingBin, all.x=T, by="modelid")[, BINNINGTABLEFIELDS, with=F]
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
    write.csv(predBinningTable, file=binningDumpFileName, row.names = F)
    
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

