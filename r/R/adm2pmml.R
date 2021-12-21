# Generates PMML from ADM models

# FEATURES
# - ensemble of multiple scorecards (including a default model when no match for context) when there are multiple ADM models for one rule
# - for simple models creates just a single scorecard
# - missing, remaining, residual symbols and WOS are handled exactly like ADM does
# - filters out invalid models ("ContextNotDefinedError")
# - combines symbolic bins with the same score distribution to save XML size
# - works with flex keys when using keys outside of standard context (by parsing the JSON in pyname) (also no keys, with numeric keys)
# - works with empty models (no active predictors, no predictors at all, with or without a classifier)
# - works with different evidence numbers for different predictors
# - uses reason codes to explain individual decisions
# - scales scorecard weights for easier interpretation so total sums up to a number between 0 and 1000
# - supports both JSON exports (from the internal ADM factory tables) and Datamart exports
#
# OUTSTANDING ISSUES/QUESTIONS
# - check with parametrized predictors
# - check with wilder predictor names, e.g. from DMSample

# The input format is a list of tuples representing all models ("partitions") of one ADM rule. The tuples
# contain:
# - an element "context" with a list of the context key-value pairs
# - an element "binning" with a data.table with the binning of all active predictors & the classifier
#
# The binning table has all the columns listed below (BINNINGTABLEFIELDS) and is close to the datamart format,
# but made complete (all required bins of every predictor are present and multiple symbols are split) so very
# easy to translate to scorecards.

ADM2PMMLVERSION <- c(name="adm2pmml", version="1.1")

# library(tidyverse)
# library(data.table)
# library(XML)
# library(jsonlite)

modelScoreFieldName <- "Normalized Score"
modelRawScoreFieldName <- "Raw Score"
modelPropensityFieldName <- "Propensity"
modelEvidenceFieldName <- "Evidence"
modelPerformanceFieldName <- "Performance"
modelIDFieldName <- "Model ID"

nrExplanations <- 3 # return top-3 reason codes

BINTYPES <- c("MISSING", "SYMBOL", "INTERVAL", "REMAININGSYMBOLS")
BINNINGTABLEFIELDS <- c("ModelID", "PredictorName", "PredictorType",
                        "BinLabel", "BinLowerBound", "BinUpperBound", "BinType", "BinPos", "BinNeg", "BinIndex",
                        "TotalPos", "TotalNeg", "Smoothing", "IsActive", "Performance")

createNumArray <- function(a)
{
  xmlNode("Array", attrs=c(type="int"), xmlTextNode(paste(a, collapse=" ")))
}

createSymArray <- function(a)
{
  if(any(grepl(" ", a, fixed = T))) {
    xmlNode("Array", attrs=c(type="string"), xmlTextNode(paste(paste("\"",a,"\"",sep=""), collapse=" ")))
  } else {
    # no need to quote - saves a lot of space in the generated XML
    xmlNode("Array", attrs=c(type="string"), xmlTextNode(paste(a, collapse=" ")))
  }
}

# Single numeric bin
createSingleNumericBin <- function(bin, reasonCode)
{
  if (is.na(bin$BinUpperBound)) { # last bin is explicitly >= lower
    xmlNode("Attribute",
            attrs=c(partialScore=bin$BinWeight, reasonCode=reasonCode),
            xmlNode("SimplePredicate", attrs=c(field=bin$PredictorName, operator="greaterOrEqual", value=bin$BinLowerBound)))
  } else {                                  # all other bins are [lower, upper> but are in order, so <upper will do
    xmlNode("Attribute",
            attrs=c(partialScore=bin$BinWeight, reasonCode=reasonCode),
            xmlNode("SimplePredicate", attrs=c(field=bin$PredictorName, operator="lessThan", value=bin$BinUpperBound)))
  }
}

# Simple summarization of a list of symbols
summarizeMultiple <- function(s)
{
  if (length(s) < 3) { return(paste(s, collapse=" "))}
  return(paste(paste(s[1:3], collapse=" "), "...(", length(s)-3, " more)", sep=""))
}

# Scorecard entry for a single predictor bin.
createSinglePredictorBin <- function(predictorBin)
{
  reasonLabel <-   switch(EXPR = as.character(predictorBin$BinType[1]),
                          "MISSING" = "Missing",
                          "INTERVAL" = ifelse(is.na(predictorBin$BinUpperBound),
                                              paste(">=",round(predictorBin$BinLowerBound,1),sep=""),
                                              paste("<",round(predictorBin$BinUpperBound,1),sep="")),
                          "REMAININGSYMBOLS" = "Remaining Symbols",
                          # Otherwise
                          summarizeMultiple(predictorBin$BinLabel))
  reasonCode <- paste(predictorBin$PredictorName,reasonLabel,
                      round(predictorBin$BinWeight),
                      round(predictorBin$minWeight),
                      round(predictorBin$avgWeight),
                      round(predictorBin$maxWeight), sep="|")

  # multiple symbols will be grouped together
  if (nrow(predictorBin) > 1) {
    if (any(predictorBin$BinType == "REMAININGSYMBOLS")) {
      nodeXML <- xmlNode("Attribute",
                         attrs=c(partialScore=predictorBin$BinWeight[1], reasonCode=reasonCode[1]),
                         xmlNode("True"))
    } else {
      nodeXML <- xmlNode("Attribute",
                         attrs=c(partialScore=predictorBin$BinWeight[1], reasonCode=reasonCode[1]),
                         xmlNode("SimpleSetPredicate",
                                 attrs=c("field"=predictorBin$PredictorName[1], "booleanOperator"="isIn"),
                                 createSymArray(predictorBin$BinLabel)))
    }
    return(nodeXML)
  }

  switch(EXPR = as.character(predictorBin$BinType),

         "MISSING" = xmlNode("Attribute",
                             attrs=c(partialScore=predictorBin$BinWeight, reasonCode=reasonCode),
                             xmlNode("SimplePredicate", attrs=c(field=predictorBin$PredictorName, operator="isMissing"))),

         "SYMBOL" = xmlNode("Attribute",
                            attrs=c(partialScore=predictorBin$BinWeight, reasonCode=reasonCode),
                            xmlNode("SimplePredicate", attrs=c(field=predictorBin$PredictorName, operator="equal", value=predictorBin$BinLabel))),

         "INTERVAL" = createSingleNumericBin(predictorBin, reasonCode),

         "REMAININGSYMBOLS" = xmlNode("Attribute",
                                      attrs=c(partialScore=predictorBin$BinWeight, reasonCode=reasonCode),
                                      xmlNode("True")))
}

# Single bin for the model "classifier" (score to propensity mapping)
createSingleClassifierBin <- function(bin)
{
  if (is.na(bin$BinLowerBound)) {
    if (is.na(bin$BinUpperBound)) {   # both NA
      # omit - rely on discretize default value
    } else { # < upper
      xmlNode("DiscretizeBin", attrs=c(binValue=bin$BinWeight),
              xmlNode("Interval", attrs=c(closure="openOpen", rightMargin=bin$BinUpperBound)))
    }
  } else if (is.na(bin$BinUpperBound)) { # >= lower
    xmlNode("DiscretizeBin", attrs=c(binValue=bin$BinWeight),
            xmlNode("Interval", attrs=c(closure="closedOpen", leftMargin=bin$BinLowerBound)))
  } else {                                         # [lower, uppper>
    xmlNode("DiscretizeBin", attrs=c(binValue=bin$BinWeight),
            xmlNode("Interval", attrs=c(closure="closedOpen", leftMargin=bin$BinLowerBound, rightMargin=bin$BinUpperBound)))
  }
}

# Discretization node for the classifier
createClassifierBins <- function(classifierBins)
{
  attrs <- c(field=modelScoreFieldName)
  if (nrow(classifierBins)==1) {
    attrs <- c(attrs, defaultValue=classifierBins$BinWeight[1])
  }
  xmlNode("Discretize", attrs=attrs,
          .children = lapply(seq(nrow(classifierBins)),
                             function(i) {return(createSingleClassifierBin(classifierBins[i]))}))
}


# Create a scorecard entry for a single predictor. The bins need to be complete and in the right order.
# Multiple symbol bins with the same score will be combined.
createPredictorNode <- function(predictorBins)
{
  binGroup <- BinType <- NULL # Trick to silence R CMD Check warnings

  predictorBins[, binGroup := .I]
  if(predictorBins$PredictorType[1] == "SYMBOLIC") {
    if(!any(predictorBins$BinType=="MISSING") | !any(predictorBins$BinType=="REMAININGSYMBOLS")) {
      print(predictorBins)
      stop("Expected a MISSING and a Remaining bin")
    }

    predictorBins[BinType=="MISSING", binGroup := 1L] # exclude MISSING and make sure that remains the first one
    predictorBins[BinType!="MISSING", binGroup := 1L+.GRP, by=c("BinPos","BinNeg")] # now group symbols together that have the same score weight
    predictorBins[binGroup == predictorBins$binGroup[which(predictorBins$BinType=="REMAININGSYMBOLS")],
                  binGroup := max(predictorBins$binGroup)+1 ] # make the group that Remaining is in the last one
  }
  xmlNode("Characteristic", attrs=c(name=paste(predictorBins$PredictorName[1], "score", sep = "___"),
                                    reasonCode=predictorBins$PredictorName[1],
                                    baselineScore=predictorBins$minWeight[1]),
          .children = lapply(sort(unique(predictorBins$binGroup)),
                             function(i) {return(createSinglePredictorBin(predictorBins[binGroup==i]))}))
}

# Returns a data table in the style of the predictor table with names and types of the context keys
getContextKeyDefinitions <- function(modeldata)
{
  contextTable <- rbindlist(lapply(modeldata, function(m) {return(as.data.table(m$context))}), use.names = T, fill = T)
  # try convert the non-missing values to numeric to check the types
  for (fld in names(contextTable)) {
    col <- contextTable[[fld]]
    colasnum <- suppressWarnings(as.numeric(col))
    if(!any(is.na(colasnum[which(col!="")]))) {
      contextTable[[fld]] <- colasnum
    }
  }

  keyDefs <- data.table(PredictorName = names(contextTable),
                        PredictorType = ifelse(sapply(contextTable, is.numeric), "NUMERIC", "SYMBOLIC"))

  return(unique(keyDefs))
}

# Mining schema for the overall model. Very similar to the data dictionary, but contains usage instead of type info.
createMiningSchema <- function(modeldata)
{
  PredictorName <- PredictorType <- NULL # Trick to silence R CMD Check warnings

  predictors <- unique( rbindlist(lapply(modeldata, function(x) { unique( x$binning [PredictorType != "CLASSIFIER", c("PredictorName")] ) } )) ) [order(PredictorName)]
  outputfields <- c(modelPropensityFieldName,
                    modelScoreFieldName,
                    modelRawScoreFieldName,
                    # NB the below are static, but in the Pega PMML rule they need to be listed as "predicted" vs "supplementary",
                    # otherwise they do not show up as outputs.
                    modelEvidenceFieldName,
                    modelPerformanceFieldName,
                    modelIDFieldName,
                    sapply(seq(nrExplanations), function(x){return(paste("Explain",x,sep="-"))}))

  xmlNode("MiningSchema",
          .children = c(lapply(outputfields,
                               function(fld) {return(xmlNode("MiningField",
                                                             # As of PMML 4.2, this is deprecated and it has been replaced by the usage type target.
                                                             attrs=c(name=fld, usageType="predicted")))}),
                        lapply(getContextKeyDefinitions(modeldata)$PredictorName,
                               function(ctx) {return(xmlNode("MiningField",
                                                             attrs=c(name=ctx, usageType="active", invalidValueTreatment="asIs")))}),
                        lapply(predictors$PredictorName,
                               function(pred) {return(xmlNode("MiningField",
                                                              attrs=c(name=pred, usageType="active", invalidValueTreatment="asIs")))})))
}

# Binning of the model "classifier" (score to propensity mapping)
createOutputs <- function(classifierBins, scaleOffset = 0, scaleFactor = 1, isScorecard)
{
  scoreOutputs <- list(xmlNode("OutputField",
                               attrs=c(name=modelScoreFieldName, feature="predictedValue", dataType="double", optype="continuous")),
                       xmlNode("OutputField",
                               attrs=c(name=modelPropensityFieldName, feature="transformedValue", dataType="double", optype="continuous"),
                               createClassifierBins(classifierBins)),
                       xmlNode("OutputField",
                               attrs=c(name=modelRawScoreFieldName, feature="transformedValue", dataType="double", optype="continuous"),
                               xmlNode("NormContinuous", attrs=c(field=modelScoreFieldName),
                                       xmlNode("LinearNorm", attrs=c("orig"=0, "norm"=-scaleOffset)),
                                       xmlNode("LinearNorm", attrs=c("orig"=1000, "norm"=1000/scaleFactor-scaleOffset)))))
  staticOutputs <- list(xmlNode("OutputField",
                                attrs=c(name=modelEvidenceFieldName, feature="transformedValue", dataType="integer", optype="continuous"),
                                xmlNode("Constant",
                                        xmlTextNode(sum(classifierBins$BinNeg)+sum(classifierBins$BinPos)))),
                        xmlNode("OutputField",
                                attrs=c(name=modelPerformanceFieldName, feature="transformedValue", dataType="double", optype="continuous"),
                                xmlNode("Constant",
                                        xmlTextNode(classifierBins$Performance[1]))),
                        xmlNode("OutputField",
                                attrs=c(name=modelIDFieldName, feature="transformedValue", dataType="string", optype="categorical"),
                                xmlNode("Constant",
                                        xmlTextNode(classifierBins$ModelID[1]))))
  if (nrExplanations > 0) {
    if (isScorecard) {
      # Reason code fields will be actual scorecard reason codes
      reasonCodes <- lapply(seq(nrExplanations),
                            function(x){return(xmlNode("OutputField",
                                                       attrs=c("name"=paste("Explain",x,sep="-"), "feature"="reasonCode", "rank"=x)))})
    } else {
      # Reason codes are just normal fields when it's not a scorecard
      reasonCodes <-  lapply(seq(nrExplanations),
                             function(x){return(xmlNode("OutputField",
                                                        attrs=c("name"=paste("Explain",x,sep="-"), feature="transformedValue", dataType="string", optype="categorical"),
                                                        xmlNode("Constant",
                                                                xmlTextNode("Context Mismatch"))))})

    }
  } else {
    reasonCodes <- list()
  }

  xmlNode("Output", .children = c(scoreOutputs, staticOutputs, reasonCodes))
}

# Default model (TreeModel) for models that do not have any active predictors (in which case there still is a classifier).
createEmptyScorecard <- function(classifierBins, modelName)
{
  if (nrow(classifierBins) <= 1) {
    if (is.na(classifierBins$BinUpperBound) & is.na(classifierBins$BinLowerBound)) {
      # fix it up as bounds NA/NA will not work
      classifierBins$BinLowerBound <- 0
    }
  }
  xmlNode("TreeModel", attrs=c("modelName"=modelName, functionName="regression"),
          xmlNode("MiningSchema",
                  xmlNode("MiningField",
                          # As of PMML 4.2, this is deprecated and it has been replaced by the usage type target.
                          attrs=c(name=modelScoreFieldName, invalidValueTreatment="asIs", usageType="predicted"))),
          createOutputs(classifierBins, isScorecard = F),

          xmlNode("Node", attrs=c(score="0"), xmlNode("True")))
}

# Default model (TreeModel) for non-matching context keys.
createDefaultModel <- function(modelName)
{
  defaultClassifierBins <- data.table(ModelID = "Default",
                                      BinLowerBound = 0,
                                      BinUpperBound = NA,
                                      BinType = "INTERVAL",
                                      BinWeight = 0.5,
                                      Performance = 0.5)

  xmlNode("TreeModel", attrs=c("modelName"=modelName, functionName="regression"),
          xmlNode("MiningSchema",
                  xmlNode("MiningField",
                          # As of PMML 4.2, this is deprecated and it has been replaced by the usage type target.
                          attrs=c(name=modelScoreFieldName, invalidValueTreatment="asIs", usageType="predicted"))),
          createOutputs(defaultClassifierBins, isScorecard = F),

          xmlNode("Node", attrs=c(score="0"), xmlNode("True")))
}

# Node with meta-info on the scorecard. Currently not being used anywhere but included to demo the potential benefits.
# With support in the accompanying rule, it could replace the inclusion of Performance/evidence etc as regular but static content.
createModelDescription <- function(classifierBins)
{
  # FPR/TPR code from cdh_utils::auc_from_bincounts
  pos <- classifierBins$BinPos
  neg <- classifierBins$BinNeg
  o <- order(pos/(pos+neg), decreasing = T)
  FPR <- rev(cumsum(neg[o]) / sum(neg))
  TPR <- rev(cumsum(pos[o]) / sum(pos))

  xmlNode("ModelExplanation",
          xmlNode("PredictiveModelQuality",
                  attrs=c(targetField="Classifier", dataUsage="training", numOfRecords=classifierBins$TotalNeg[1]+classifierBins$TotalPos[1]),
                  xmlNode("LiftData",
                          attrs=c(rankingQuality=auc2GINI(classifierBins$Performance[1]),
                                  targetFieldValue="Positive"),
                          xmlNode("ModelLiftGraph",
                                  xmlNode("LiftGraph",
                                          xmlNode("XCoordinates", createNumArray(cumsum(classifierBins$BinPos+classifierBins$BinNeg))),
                                          xmlNode("YCoordinates", createNumArray(classifierBins$BinPos))))),
                  xmlNode("ROC",
                          attrs=c(positiveTargetFieldValue="Positive"),
                          xmlNode("ROCGraph",
                                  xmlNode("XCoordinates", createNumArray(FPR)),
                                  xmlNode("YCoordinates", createNumArray(TPR))))))
}

# Sets the bin weights for all bins of a single "partition" of the ADM rule. The weight is basically the
# log odds, but is normalized to the number of predictors. Returns a summary table with the min/max per predictor.
getNBFormulaWeights <- function(bins)
{
  BinLabel <- binLogOdds <- BinLowerBound <- BinNeg <- BinPos <- BinType <- BinUpperBound <- BinWeight <- minWeight <- ModelID <- nPredictors <- perPredictorContribution <- PredictorName <- PredictorType <- Smoothing <- TotalNeg <- TotalPos <- NULL # Trick to silence R CMD Check warnings

  # Some of the code depends on the order of the bins
  setorder(bins, ModelID, PredictorName, BinType, BinUpperBound, BinLabel, na.last = T)

  # Set the classifier weights
  bins[PredictorType == "CLASSIFIER", BinWeight := (0.5+BinPos)/(1+BinPos+BinNeg)]

  # Weights for ordinary fields
  bins[PredictorType != "CLASSIFIER", nPredictors := uniqueN(PredictorName)]
  bins[PredictorType != "CLASSIFIER", binLogOdds := log(BinPos + Smoothing) - log(BinNeg + Smoothing)]
  bins[PredictorType != "CLASSIFIER", perPredictorContribution := (1/nPredictors - 1)*log(1+TotalPos) + (1 - 1/nPredictors)*log(1+TotalNeg)]
  bins[PredictorType != "CLASSIFIER", BinWeight := (binLogOdds + perPredictorContribution)/(1+nPredictors)]

  # Summarize per predictor
  if (any(bins$PredictorType != "CLASSIFIER")) {
    predMinMaxWeights <- bins[PredictorType != "CLASSIFIER", list(minWeight = min(BinWeight),
                                                                  maxWeight = max(BinWeight)), by=PredictorName]
  } else {
    predMinMaxWeights <- data.table()
  }

  # Scale the weights to 0..1000
  scWeight <- 0 # deliberately chosen to be 0 - contributions of predictors are thru 'perPredictorContribution'
  hasNoPredictors <- nrow(bins[PredictorType != "CLASSIFIER"]) == 0
  if (hasNoPredictors) {
    scaleOffset <- 0.0
    scaleFactor <- 1.0
  } else {
    scaleFactor <- 1000/(sum(predMinMaxWeights$maxWeight) - sum(predMinMaxWeights$minWeight))
    scaleOffset <- -sum(predMinMaxWeights$minWeight)

    # the scaling is sometimes slightly different than the old version because that unconditionally took the wos bin into account, which we no longer need here

    bins <- merge(bins, predMinMaxWeights, all=T, by=c("PredictorName"))

    bins[PredictorType != "CLASSIFIER", BinWeight := (BinWeight-minWeight)*scaleFactor]
    bins[PredictorType == "CLASSIFIER", BinLowerBound := (BinLowerBound+scaleOffset)*scaleFactor]
    bins[PredictorType == "CLASSIFIER", BinUpperBound := (BinUpperBound+scaleOffset)*scaleFactor]
  }

  result <- list(binning = bins, scaleFactor = scaleFactor, scaleOffset = scaleOffset, scWeight = scWeight)

  return(result)
}

# Creates a single scorecard, for a single "partition" of the ADM rule
createScorecard <- function(modelbins, modelName)
{
  avgWeight <- BinNeg <- BinPos <- BinWeight <- maxWeight <- minWeight <- PredictorName <- PredictorType <- NULL # Trick to silence R CMD Check warnings

  # Add the bin weights based on the log odds
  scaling <- getNBFormulaWeights(modelbins)
  modelbins <- scaling$binning
  scaleOffset <- scaling$scaleOffset
  scaleFactor <- scaling$scaleFactor
  scWeight <- scaling$scWeight

  hasNoPredictors <- nrow(modelbins[PredictorType != "CLASSIFIER"]) == 0

  # In support of decision explanation through reason codes, add some meta info to the bins (not used for scoring)
  if (hasNoPredictors) {
    modelbins[PredictorType != "CLASSIFIER", avgWeight := NA]
    modelbins[PredictorType != "CLASSIFIER", minWeight := NA]
    modelbins[PredictorType != "CLASSIFIER", maxWeight := NA]
  } else {
    modelbins[PredictorType != "CLASSIFIER", avgWeight := weighted.mean(BinWeight,BinNeg+BinPos),by=PredictorName]
    modelbins[PredictorType != "CLASSIFIER", minWeight := min(BinWeight),by=PredictorName]
    modelbins[PredictorType != "CLASSIFIER", maxWeight := max(BinWeight),by=PredictorName]
  }

  if (hasNoPredictors) {
    return(createEmptyScorecard(modelbins[PredictorType=="CLASSIFIER"], modelName))
  }
  xmlNode("Scorecard", attrs=c("modelName"=modelName, functionName="regression",
                               algorithmName="PEGA Adaptive Decisioning",
                               useReasonCodes="true", baselineMethod="min", reasonCodeAlgorithm="pointsAbove",
                               initialScore=scWeight),
          xmlNode("MiningSchema",
                  .children = lapply(sort(unique(modelbins[PredictorType != "CLASSIFIER"]$PredictorName)),
                                     function(pred) {return(xmlNode("MiningField",
                                                                    attrs=c(name=pred, usageType="active", invalidValueTreatment="asIs")))})),
          createOutputs(modelbins[PredictorType=="CLASSIFIER"], scaleOffset, scaleFactor, isScorecard = T),
          createModelDescription(modelbins[PredictorType=="CLASSIFIER"]),
          xmlNode("Characteristics", .children=lapply(unique(modelbins[PredictorType != "CLASSIFIER"]$PredictorName),
                                                      function(predictor) {return(createPredictorNode(modelbins[PredictorName==predictor]))})))
}

# Creates the PMML Header with timestamp etc
createHeaderNode <- function(modelName)
{
  xmlNode("Header", attrs=c(description=paste("ADM Model export for", modelName)),
          xmlNode("Application", attrs=ADM2PMMLVERSION),
          xmlNode("Timestamp", xmlTextNode(as.character(date()))))
}

# Creates the overall data dictionary. This includes the superset of all active predictors and context keys, but also the
# standard output fields (score, propensity etc). These seem to be necessary in the data dictionary as well - not 100% sure why.
createDataDictionaryNode <- function(modeldata)
{
  PredictorName <- PredictorType <- NULL # Trick to silence R CMD Check warnings

  predictors <- unique( rbindlist(lapply(modeldata,
                                         function(x) { unique( x$binning [PredictorType != "CLASSIFIER", c("PredictorName","PredictorType")] ) } )) ) [order(PredictorName)]
  outputfields <- data.table(PredictorName=c(modelPropensityFieldName,
                                             modelScoreFieldName,
                                             modelRawScoreFieldName,
                                             modelEvidenceFieldName,
                                             modelPerformanceFieldName,
                                             modelIDFieldName,
                                             sapply(seq(nrExplanations), function(x){return(paste("Explain",x,sep="-"))})),
                             PredictorType=c("NUMERIC", "NUMERIC", "NUMERIC", "INTEGER", "NUMERIC", "SYMBOLIC",
                                             rep("SYMBOLIC", nrExplanations)))
  dataDict <- rbindlist(list(outputfields,
                             getContextKeyDefinitions(modeldata),
                             predictors))

  xmlNode("DataDictionary",
          .children = lapply(seq(nrow(dataDict)),
                             function(i) {return(xmlNode("DataField",
                                                         attrs=c(name=dataDict[i]$PredictorName,
                                                                 dataType=plyr::revalue(dataDict[i]$PredictorType,
                                                                                        c("SYMBOLIC"="string","NUMERIC"="double","INTEGER"="integer"), warn_missing=F),
                                                                 optype=plyr::revalue(dataDict[i]$PredictorType,
                                                                                      c("SYMBOLIC"="categorical","NUMERIC"="continuous","INTEGER"="continuous"), warn_missing=F))))}))
}

# Generates the conditions to select a particular model based on context key values
createSegmentPredicate <- function(contextKVPs)
{
  if (length(contextKVPs) == 1) {
    xmlNode("SimplePredicate",
            attrs=c(field=names(contextKVPs)[1], operator="equal", value=contextKVPs[[1]]))
  } else {
    xmlNode("CompoundPredicate",
            attrs=c(booleanOperator="and"),
            .children = sapply(names(contextKVPs),
                               function(key) {
                                 return(xmlNode("SimplePredicate",
                                                attrs=c(field=key, operator="equal", value=contextKVPs[[key]])))}))
  }
}

# creates complex PMML embedded in an overall MiningModel segmented by context key values
createModelPartitions <- function(modeldata, overallModelName)
{
  segments <- lapply(names(modeldata),
                     function(ModelID) {
                       return(xmlNode("Segment", attrs=c(id=ModelID),
                                      createSegmentPredicate(modeldata[[ModelID]]$context),
                                      createScorecard(modeldata[[ModelID]]$binning,
                                                      paste(overallModelName, ModelID, sep="_"))))})

  segments[[1+length(segments)]] <- xmlNode("Segment", attrs=c(id="default"),
                                            xmlNode("True"),
                                            createDefaultModel(modelName = paste(overallModelName, "default", sep="_")))

  xmlNode("MiningModel", attrs=c(functionName="regression"),
          createMiningSchema(modeldata),
          xmlNode("Segmentation", attrs=c("multipleModelMethod"="selectFirst"), .children = segments))
}

# Checks is a model is valid. ADM data can contain invalid models (for which the context is not fully defined).
isValidModel <- function(singlemodeldata)
{
  !any(!is.null(singlemodeldata$context) & singlemodeldata$context == "ContextNotDefinedError" )
}

# Creates single PMML Segment model with multiple embedded Scorecards. Each segment is defined
# by the context key values. Input is a list of context-binning tuples.
createPMML <- function(modeldata, overallModelName)
{
  BinType <- PredictorName <- PredictorType <- NULL # Trick to silence R CMD Check warnings

  # Sanity checks to validate assumptions on the input:

  # Drop models with invalid context keys
  modeldata <- modeldata[sapply(modeldata, isValidModel)]

  # Make sure the input has the required fields
  sapply(modeldata, function(m) { if (!identical(sort(names(m$binning)),
                                                 sort(BINNINGTABLEFIELDS)))
    stop(paste("Not all required columns present.\nFound:",
               paste(sort(names(m$binning)), collapse = ","),
               "\nExpected:", paste(sort(BINNINGTABLEFIELDS), collapse = ",")))})

  # Make sure the "enums" only have the expected values
  for (i in length(modeldata)) {
    oriBinTypes <- unique(modeldata[[i]]$binning$BinType)
    modeldata[[i]]$binning$BinType <- factor(modeldata[[i]]$binning$BinType, levels=BINTYPES)
    if (any(is.na(modeldata[[i]]$binning$BinType))) {
      print(oriBinTypes)
      stop(paste("Unexpected bin types encountered.",
                 "\nExpected:", paste(BINTYPES,collapse = ","),
                 "\nGot:", paste(oriBinTypes, collapse = ",")))
    }
  }

  # Make sure the predictor binning is according to expectations
  # numerics: INTERVALBELOW & INTERVALABOVE also set bound for last interval. Both should be present.
  binningSummary <- rbindlist(lapply(modeldata, function(m) {return(m$binning[, list(nMissing=sum(BinType=="MISSING")),
                                                                              by=c("ModelID", "PredictorName", "PredictorType")])}))
  if (any(binningSummary$nMissing != 1 & binningSummary$PredictorType != "CLASSIFIER")) {
    print(binningSummary[which(binningSummary$nMissing != 1 & binningSummary$PredictorType != "CLASSIFIER")])
    stop("Incorrect number of missing bins.")
  }

  # Make sure the predictor binning is according to expectations
  modelSummary <- rbindlist(lapply(modeldata, function(m) {return(m$binning[, list(hasClassifier=any(PredictorType=="CLASSIFIER"),
                                                                                   hasPredictors=any(PredictorType=="NUMERIC" | PredictorType=="SYMBOLIC"),
                                                                                   PredictorNames=paste(unique(PredictorName), collapse = ",")),
                                                                            by=c("ModelID")])}))
  if (!all(modelSummary$hasClassifier)) {
    print(modelSummary)
    stop("No classifier present.")
  }

  hasNoContext <- sapply(modeldata, function(x) {return(is.null(x[["context"]]))})

  if (length(modeldata)==1 & all(hasNoContext)) {
    # Single scorecard when no context and there is only 1 model
    xmlNode("PMML", attrs=c("xmlns"="http://www.dmg.org/PMML-4_1", "version"="4.1"),
            createHeaderNode(overallModelName),
            createDataDictionaryNode(modeldata),
            createScorecard(modeldata[[1]]$binning, overallModelName))
  } else {
    xmlNode("PMML", attrs=c("xmlns"="http://www.dmg.org/PMML-4_1", "version"="4.1"),
            createHeaderNode(overallModelName),
            createDataDictionaryNode(modeldata),
            createModelPartitions(modeldata, overallModelName))
  }
}

#' Generates PMML files from ADM models.
#'
#' Every model configuration becomes one big PMML file. The big ensemble contains
#' score cards for every model partition. The ensemble switches between the
#' models based on context key values.
#'
#' @param datamart Model and predictor data from the ADM datamart.
#' @param destDir Directory to create the PMML files in. Defaults to the current
#'   directory.
#' @param tmpDir Directory for temporary files. Defaults to \code{NULL}
#'   indicating no temp files will be left behind.
#' @param verbose Whether to print the steps it's performing. Defaults to
#'   \code{TRUE} if the \code{tmpDir} is set.
#'
#' @return Creates the PMML file(s) in the given destination and returns a list
#'   of model ID's per file.
#' @export
#'
#' @examples
#' \dontrun{
#'   dm <- ADMDatamart("~/Downloads")
#'
#'   adm2pmml(dm, verbose=TRUE)
#' }
adm2pmml <- function(datamart,
                     destDir = ".",
                     tmpDir = NULL, verbose = (!is.null(tmpDir)))
{
  ConfigurationName <- ModelID <- ConfigpartitionID <- NULL # Trick to silence R CMD Check warnings

  dmModels <- copy(datamart$modeldata)
  dmPredictors <- copy(datamart$predictordata)

  # PMML conversion code assumes character types
  for (f in names(dmPredictors)[sapply(dmPredictors, is.factor)]) dmPredictors[[f]] <- as.character(dmPredictors[[f]])
  for (f in names(dmModels)[sapply(dmModels, is.factor)]) dmModels[[f]] <- as.character(dmModels[[f]])

  result <- list()

  for (admConfigurationName in unique(dmModels$ConfigurationName)) {
    if (verbose) cat("Processing", admConfigurationName, fill=T)

    pmmlFileName <- file.path(destDir, paste(admConfigurationName, "pmml", sep="."))

    # Create list with normalized binning data for every partition
    models <- unique(dmModels[ConfigurationName == admConfigurationName])
    modelList <- normalizedBinningFromDatamart(dmPredictors,
                                               admConfigurationName,
                                               tmpDir,
                                               modelsForPartition = models)

    # Create an individual PMML file
    pmml <- createPMML(modelList, admConfigurationName)

    sink(pmmlFileName)
    print(pmml)
    sink()

    result[[pmmlFileName]] <- unique(models$ModelID)

    if (verbose) cat("Generated", pmmlFileName, paste("(", length(modelList), " models)", sep=""), fill=T)
  }
  # } else if (!is.null(modelFactory)) {
  #   standardizeFieldCasing(modelFactory)
  #
  #   # returns a list of binning, context pairs - we just want to pass this list to the PMML exporter
  #   for (p in unique(modelFactory$ConfigpartitionID)) {
  #     # get all model "partitions" for this model rule
  #     modelPartitions <- modelFactory[ConfigpartitionID==p]
  #     modelPartitionInfo <- fromJSON(modelPartitions[1]$Configpartition)
  #     modelPartitionName <- modelPartitionInfo$partition$Purpose
  #     modelPartitionFullName <- paste(modelPartitionInfo$partition$ClassName, modelPartitionInfo$partition$Purpose, sep="_")
  #
  #     if(!is.null(appliesToFilter)) {
  #       if (!grepl(appliesToFilter, modelPartitionInfo$partition$ClassName, ignore.case=T, perl=T)) next
  #     }
  #     if(!is.null(ruleNameFilter)) {
  #       if (!grepl(ruleNameFilter, modelPartitionInfo$partition$Purpose, ignore.case=T, perl=T)) next
  #     }
  #
  #     if(verbose) {
  #       print(paste("Processing", modelPartitionFullName))
  #     }
  #     pmmlFileName <- paste(destDir, paste(modelPartitionName, "pmml", sep="."), sep="/")
  #
  #     modelList <- normalizedBinningFromADMFactory(modelPartitions, modelPartitionFullName, tmpDir)
  #
  #     pmml <- createPMML(modelList, modelPartitionFullName)
  #
  #     sink(pmmlFileName)
  #     print(pmml)
  #     sink()
  #
  #     generatedPMMLFiles[[pmmlFileName]] <- unique(modelPartitions$ModelpartitionID)
  #
  #     if(verbose) {
  #       print(paste("Generated",pmmlFileName,paste("(",nrow(modelPartitions)," models)", sep="")))
  #     }
  #   }
  # }

  if (length(result)>0) return(result[order(names(result))])
  return(result)
}
