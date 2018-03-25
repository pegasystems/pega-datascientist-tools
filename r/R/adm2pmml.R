# Generates PMML from ADM models

# FEATURES
# - single scorecard
# - ensemble of multiple scorecards (including a default model when no match for context) when there are multiple ADM models for one rule
# - missing, remaining, residual symbols and WOS are handled like ADM does
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
# - the way ADM defines the score for missings and residuals not seen at train time is questionable, run test to see if this matters
# - PMML specific: is there a better place for evidence, performance etc? Now just dumping as (transformed) output fields
# - check with parameterized predictors
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
BINNINGTABLEFIELDS <- c("modelid", "predictorname", "predictortype",
                        "binlabel", "binlowerbound", "binupperbound", "bintype", "binpos", "binneg",
                        "totalpos", "totalneg", "smoothing", "performance")

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
  if (is.na(bin$binupperbound)) { # last bin is explicitly >= lower
    xmlNode("Attribute",
            attrs=c(partialScore=bin$binWeight, reasonCode=reasonCode),
            xmlNode("SimplePredicate", attrs=c(field=bin$predictorname, operator="greaterOrEqual", value=bin$binlowerbound)))
  } else {                                  # all other bins are [lower, upper> but are in order, so <upper will do
    xmlNode("Attribute",
            attrs=c(partialScore=bin$binWeight, reasonCode=reasonCode),
            xmlNode("SimplePredicate", attrs=c(field=bin$predictorname, operator="lessThan", value=bin$binupperbound)))
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
  reasonLabel <-   switch(EXPR = as.character(predictorBin$bintype[1]),
                          "MISSING" = "Missing",
                          "INTERVAL" = ifelse(is.na(predictorBin$binupperbound),
                                              paste(">=",round(predictorBin$binlowerbound,1),sep=""),
                                              paste("<",round(predictorBin$binupperbound,1),sep="")),
                          "REMAININGSYMBOLS" = "Remaining Symbols",
                          # Otherwise
                          summarizeMultiple(predictorBin$binlabel))
  reasonCode <- paste(predictorBin$predictorname,reasonLabel,
                      round(predictorBin$binWeight),round(predictorBin$avgWeight),round(predictorBin$maxWeight),sep="|")

  # multiple symbols will be grouped together
  if (nrow(predictorBin) > 1) {
    if (any(predictorBin$bintype == "REMAININGSYMBOLS")) {
      nodeXML <- xmlNode("Attribute",
                         attrs=c(partialScore=predictorBin$binWeight[1], reasonCode=reasonCode[1]),
                         xmlNode("True"))
    } else {
      nodeXML <- xmlNode("Attribute",
                         attrs=c(partialScore=predictorBin$binWeight[1], reasonCode=reasonCode[1]),
                         xmlNode("SimpleSetPredicate",
                                 attrs=c("field"=predictorBin$predictorname[1], "booleanOperator"="isIn"),
                                 createSymArray(predictorBin$binlabel)))
    }
    return(nodeXML)
  }

  switch(EXPR = as.character(predictorBin$bintype),

         "MISSING" = xmlNode("Attribute",
                             attrs=c(partialScore=predictorBin$binWeight, reasonCode=reasonCode),
                             xmlNode("SimplePredicate", attrs=c(field=predictorBin$predictorname, operator="isMissing"))),

         "SYMBOL" = xmlNode("Attribute",
                            attrs=c(partialScore=predictorBin$binWeight, reasonCode=reasonCode),
                            xmlNode("SimplePredicate", attrs=c(field=predictorBin$predictorname, operator="equal", value=predictorBin$binlabel))),

         "INTERVAL" = createSingleNumericBin(predictorBin, reasonCode),

         "REMAININGSYMBOLS" = xmlNode("Attribute",
                                      attrs=c(partialScore=predictorBin$binWeight, reasonCode=reasonCode),
                                      xmlNode("True")))
}

# Single bin for the model "classifier" (score to propensity mapping)
createSingleClassifierBin <- function(bin)
{
  if (is.na(bin$binlowerbound)) {        # < upper
    xmlNode("DiscretizeBin", attrs=c(binValue=bin$binWeight),
            xmlNode("Interval", attrs=c(closure="openOpen", rightMargin=bin$binupperbound)))
  } else if (is.na(bin$binupperbound)) { # >= lower
    xmlNode("DiscretizeBin", attrs=c(binValue=bin$binWeight),
            xmlNode("Interval", attrs=c(closure="closedOpen", leftMargin=bin$binlowerbound)))
  } else {                                         # [lower, uppper>
    xmlNode("DiscretizeBin", attrs=c(binValue=bin$binWeight),
            xmlNode("Interval", attrs=c(closure="closedOpen", leftMargin=bin$binlowerbound, rightMargin=bin$binupperbound)))
  }
}

# Create a scorecard entry for a single predictor. The bins need to be complete and in the right order.
# Multiple symbol bins with the same score will be combined.
createPredictorNode <- function(predictorBins)
{
  predictorBins[, binGroup := .I]
  if(predictorBins$predictortype[1] == "SYMBOLIC") {
    if(!any(predictorBins$bintype=="MISSING") | !any(predictorBins$bintype=="REMAININGSYMBOLS")) {
      print(predictorBins)
      stop("Expected a MISSING and a Remaining bin")
    }

    predictorBins[bintype=="MISSING", binGroup := 1L] # exclude MISSING and make sure that remains the first one
    predictorBins[bintype!="MISSING", binGroup := 1L+.GRP, by=c("binpos","binneg")] # now group symbols together that have the same score weight
    predictorBins[binGroup == predictorBins$binGroup[which(predictorBins$bintype=="REMAININGSYMBOLS")],
                  binGroup := max(predictorBins$binGroup)+1 ] # make the group that Remaining is in the last one
  }
  xmlNode("Characteristic", attrs=c(name=paste(predictorBins$predictorname[1], "score", sep = "___"),
                                    reasonCode=predictorBins$predictorname[1],
                                    baselineScore=predictorBins$maxWeight[1]),
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

  keyDefs <- data.table(predictorname = names(contextTable),
                        predictortype = ifelse(sapply(contextTable, is.numeric), "NUMERIC", "SYMBOLIC"))

  return(unique(keyDefs))
}

# Mining schema for the overall model. Very similar to the data dictionary, but contains usage instead of type info.
createMiningSchema <- function(modeldata)
{
  predictors <- unique( rbindlist(lapply(modeldata, function(x) { unique( x$binning [predictortype != "CLASSIFIER", c("predictorname")] ) } )) ) [order(predictorname)]
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
                                                             attrs=c(name=fld, usageType="predicted")))}),
                        lapply(getContextKeyDefinitions(modeldata)$predictorname,
                               function(ctx) {return(xmlNode("MiningField",
                                                             attrs=c(name=ctx, usageType="active", invalidValueTreatment="asIs")))}),
                        lapply(predictors$predictorname,
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
                               xmlNode("Discretize", attrs=c(field=modelScoreFieldName),
                                       .children = lapply(seq(nrow(classifierBins)),
                                                          function(i) {return(createSingleClassifierBin(classifierBins[i]))}))),
                       xmlNode("OutputField",
                               attrs=c(name=modelRawScoreFieldName, feature="transformedValue", dataType="double", optype="continuous"),
                               xmlNode("NormContinuous", attrs=c(field=modelScoreFieldName),
                                       xmlNode("LinearNorm", attrs=c("orig"=0, "norm"=-scaleOffset)),
                                       xmlNode("LinearNorm", attrs=c("orig"=1000, "norm"=1000/scaleFactor-scaleOffset)))))
  staticOutputs <- list(xmlNode("OutputField",
                                attrs=c(name=modelEvidenceFieldName, feature="transformedValue", dataType="integer", optype="continuous"),
                                xmlNode("Constant",
                                        xmlTextNode(sum(classifierBins$binneg)+sum(classifierBins$binpos)))),
                        xmlNode("OutputField",
                                attrs=c(name=modelPerformanceFieldName, feature="transformedValue", dataType="double", optype="continuous"),
                                xmlNode("Constant",
                                        xmlTextNode(classifierBins$performance[1]))),
                        xmlNode("OutputField",
                                attrs=c(name=modelIDFieldName, feature="transformedValue", dataType="string", optype="categorical"),
                                xmlNode("Constant",
                                        xmlTextNode(classifierBins$modelid[1]))))
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
    if (is.na(classifierBins$binupperbound) & is.na(classifierBins$binlowerbound)) {
      # fix it up as bounds NA/NA will not work
      classifierBins$binlowerbound <- 0
    }
  }
  xmlNode("TreeModel", attrs=c("modelName"=modelName, functionName="regression"),
          xmlNode("MiningSchema",
                  xmlNode("MiningField",
                          attrs=c(name=modelScoreFieldName, invalidValueTreatment="asIs", usageType="predicted"))),
          createOutputs(classifierBins, isScorecard = F),

          xmlNode("Node", attrs=c(score="0"), xmlNode("True")))
}

# Default model (TreeModel) for non-matching context keys.
createDefaultModel <- function(modelName)
{
  defaultClassifierBins <- data.table(modelid = "Default",
                              binlowerbound = 0,
                              binupperbound = NA,
                              bintype = "INTERVAL",
                              binWeight = 0.5,
                              performance = 0.5)

  xmlNode("TreeModel", attrs=c("modelName"=modelName, functionName="regression"),
          xmlNode("MiningSchema",
                  xmlNode("MiningField",
                          attrs=c(name=modelScoreFieldName, invalidValueTreatment="asIs", usageType="predicted"))),
          createOutputs(defaultClassifierBins, isScorecard = F),

          xmlNode("Node", attrs=c(score="0"), xmlNode("True")))
}

# Node with meta-info on the scorecard. Currently not being used anywhere but included to demo the potential benefits.
# With support in the accompanying rule, it could replace the inclusion of performance/evidence etc as regular but static content.
createModelDescription <- function(classifierBins)
{
  xmlNode("ModelExplanation",
          xmlNode("PredictiveModelQuality",
                  attrs=c(targetField="Classifier", dataUsage="training", numOfRecords=classifierBins$totalneg[1]+classifierBins$totalpos[1]),
                  xmlNode("LiftData",
                          attrs=c(rankingQuality=auc2GINI(classifierBins$performance[1]),
                                  targetFieldValue="Positive"),
                          xmlNode("ModelLiftGraph",
                                  xmlNode("LiftGraph",
                                          xmlNode("XCoordinates", createNumArray(cumsum(classifierBins$binpos+classifierBins$binneg))),
                                          xmlNode("YCoordinates", createNumArray(classifierBins$binpos)))))))
}

# Creates a single scorecard, for a single "partition" of the ADM rule
createScorecard <- function(modelbins, modelName)
{
  # Weight calculations
  modelbins[predictortype != "CLASSIFIER", nPredictors := uniqueN(predictorname)]
  modelbins[predictortype != "CLASSIFIER", binLogOdds := log(binpos + smoothing) - log(binneg + smoothing)]
  modelbins[predictortype != "CLASSIFIER", perPredictorContribution := (1/nPredictors - 1)*log(1+totalpos) + (1 - 1/nPredictors)*log(1+totalneg)]
  modelbins[predictortype != "CLASSIFIER", binWeight := (binLogOdds + perPredictorContribution)/(1+nPredictors)]

  scWeight <- 0 # deliberately chosen to be 0 - contributions of predictors are thru 'perPredictorContribution'
  hasNoPredictors <- nrow(modelbins[predictortype != "CLASSIFIER"]) == 0

  # Scaling to 0..1000
  if (hasNoPredictors) {
    scaleOffset <- 0.0
    scaleFactor <- 1.0
  } else {
    scaling <- modelbins[predictortype != "CLASSIFIER", .(minWeight = min(binWeight),
                                                          maxWeight = max(binWeight)), by=predictorname]
    scaleFactor <- 1000/(sum(scaling$maxWeight) - sum(scaling$minWeight))
    scaleOffset <- -sum(scaling$minWeight)

    # the scaling is sometimes slightly different than the old version because that unconditionally took the wos bin into account, which we no longer need here

    modelbins <- merge(modelbins, scaling, all=T, by=c("predictorname"))

    modelbins[predictortype != "CLASSIFIER", binWeight := (binWeight-minWeight)*scaleFactor]
    modelbins[predictortype == "CLASSIFIER", binlowerbound := (binlowerbound+scaleOffset)*scaleFactor]
    modelbins[predictortype == "CLASSIFIER", binupperbound := (binupperbound+scaleOffset)*scaleFactor]
  }

  # In support of decision explanation through reason codes, add some meta info to the bins (not used for scoring)
  if (hasNoPredictors) {
    modelbins[predictortype != "CLASSIFIER", avgWeight := NA]
    modelbins[predictortype != "CLASSIFIER", maxWeight := NA]
  } else {
    modelbins[predictortype != "CLASSIFIER", avgWeight := weighted.mean(binWeight,binneg+binpos),by=predictorname]
    modelbins[predictortype != "CLASSIFIER", maxWeight := max(binWeight),by=predictorname]
  }

  if (hasNoPredictors) {
    return(createEmptyScorecard(modelbins[predictortype=="CLASSIFIER"], modelName))
  }
  xmlNode("Scorecard", attrs=c("modelName"=modelName, functionName="regression",
                               algorithmName="PEGA Adaptive Decisioning",
                               useReasonCodes="true", baselineMethod="min", reasonCodeAlgorithm="pointsBelow",
                               initialScore=scWeight),
          xmlNode("MiningSchema",
                  .children = lapply(sort(unique(modelbins[predictortype != "CLASSIFIER"]$predictorname)),
                                     function(pred) {return(xmlNode("MiningField",
                                                                    attrs=c(name=pred, usageType="active", invalidValueTreatment="asIs")))})),
          createOutputs(modelbins[predictortype=="CLASSIFIER"], scaleOffset, scaleFactor, isScorecard = T),
          createModelDescription(modelbins[predictortype=="CLASSIFIER"]),
          xmlNode("Characteristics", .children=lapply(unique(modelbins[predictortype != "CLASSIFIER"]$predictorname),
                                                      function(predictor) {return(createPredictorNode(modelbins[predictorname==predictor]))})))
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
  predictors <- unique( rbindlist(lapply(modeldata,
                                         function(x) { unique( x$binning [predictortype != "CLASSIFIER", c("predictorname","predictortype")] ) } )) ) [order(predictorname)]
  outputfields <- data.table(predictorname=c(modelPropensityFieldName,
                                             modelScoreFieldName,
                                             modelRawScoreFieldName,
                                             modelEvidenceFieldName,
                                             modelPerformanceFieldName,
                                             modelIDFieldName,
                                             sapply(seq(nrExplanations), function(x){return(paste("Explain",x,sep="-"))})),
                             predictortype=c("NUMERIC", "NUMERIC", "NUMERIC", "INTEGER", "NUMERIC", "SYMBOLIC",
                                             rep("SYMBOLIC", nrExplanations)))
  dataDict <- rbindlist(list(outputfields, getContextKeyDefinitions(modeldata), predictors))

  xmlNode("DataDictionary",
          .children = lapply(seq(nrow(dataDict)),
                             function(i) {return(xmlNode("DataField",
                                                         attrs=c(name=dataDict[i]$predictorname,
                                                                 dataType=plyr::revalue(dataDict[i]$predictortype,
                                                                                        c("SYMBOLIC"="string","NUMERIC"="double","INTEGER"="integer"), warn_missing=F),
                                                                 optype=plyr::revalue(dataDict[i]$predictortype,
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
            .children = sapply(names(contextKVPs), function(key) {
              return(xmlNode("SimplePredicate",
                             attrs=c(field=key, operator="equal", value=contextKVPs[[key]])))}))
  }
}

# creates complex PMML embedded in an overall MiningModel segmented by context key values
createModelPartitions <- function(modeldata, overallModelName)
{
  segments <- lapply(names(modeldata), function(modelid) {return(xmlNode("Segment", attrs=c(id=modelid),
                                                                         createSegmentPredicate(modeldata[[modelid]]$context),
                                                                         createScorecard(modeldata[[modelid]]$binning,
                                                                                         paste(overallModelName, modelid, sep="_"))))})

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

# Creates single PMML Segment model with multiple embedded Scorecards. Each segment is defined by the context key values.
# Input is a list of context-binning tuples.
createPMML <- function(modeldata, overallModelName)
{
  # Sanity checks to validate assumptions on the input:

  # Drop models with invalid context keys
  modeldata <- modeldata[sapply(modeldata, isValidModel)]

  # Make sure the input has the required fields
  sapply(modeldata, function(m) { if (!identical(sort(names(m$binning)),
                                                 sort(BINNINGTABLEFIELDS)))
    stop(paste("Not all required columns present. Found:", paste(sort(names(m$binning)), collapse = ","), ", expected:", paste(sort(BINNINGTABLEFIELDS), collapse = ",")))})

  # Make sure the "enums" only have the expected values
  for (i in length(modeldata)) {
    oribintypes <- unique(modeldata[[i]]$binning$bintype)
    modeldata[[i]]$binning$bintype <- factor(modeldata[[i]]$binning$bintype, levels=BINTYPES)
    if (any(is.na(modeldata[[i]]$binning$bintype))) {
      print(oribintypes)
      stop(paste("Unexpected bin types encountered.", "Expected:", paste(BINTYPES,collapse = ","), ", got:", paste(oribintypes, collapse = ",")))
    }
  }

  # Make sure the predictor binning is according to expectations
  # numerics: INTERVALBELOW & INTERVALABOVE also set bound for last interval. Both should be present.
  # classifier! forgotten
  binningSummary <- rbindlist(lapply(modeldata, function(m) {return(m$binning[,.(nMissing=sum(bintype=="MISSING")),
                                                                              by=c("modelid", "predictorname", "predictortype")])}))
  if (any(binningSummary$nMissing != 1 & binningSummary$predictortype != "CLASSIFIER")) {
    print(binningSummary)
    stop("Incorrect number of missing bins.")
  }

  # Make sure the predictor binning is according to expectations
  modelSummary <- rbindlist(lapply(modeldata, function(m) {return(m$binning[,.(hasClassifier=any(predictortype=="CLASSIFIER"),
                                                                               hasPredictors=any(predictortype=="NUMERIC" | predictortype=="SYMBOLIC"),
                                                                               predictornames=paste(unique(predictorname), collapse = ",")),
                                                                            by=c("modelid")])}))
  if (!all(modelSummary$hasClassifier)) {
    print(modelSummary)
    stop("No classifier present.")
  }

  # Massage the data for easier processing. Already set the classifier weights here.
  for (i in seq(length(modeldata))) {
    setorder(modeldata[[i]]$binning, modelid, predictorname, bintype, binupperbound, binlabel, na.last = T)

    modeldata[[i]]$binning [predictortype == "CLASSIFIER", binWeight := (0.5+binpos)/(1+binpos+binneg)]
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
#' Generates PMML files from the ADM models in either the ADM data mart,
#' directly from the database or from dataset exports, or from the (internal)
#' "Factory" tables. The generated PMML contains all the individual models for
#' each model partition inside a big ensemble that selects between them based on
#' context keys.
#'
#' @section Exporting a specific version: When exporting from the datamart, only
#'   the models from the most recent model snapshot will be exported, and for
#'   each of these, only the most recent of the predictor snapshots of that
#'   model will be used. To export specific snapshots subset the data before
#'   calling this method.
#'
#' @param dmModels Data frame (or anything else that converts to
#'   \code{data.table}) with the ADM datamart model data
#' @param dmPredictors Data frame (or anything else that converts to
#'   \code{data.table}) with the ADM datamart predictor data. Required if
#'   \code{dmModels} is given. If \code{dmPredictors} is given but
#'   \code{dmModels} is not, the assumption will be made that all the predictor
#'   binning belongs to a single model.
#' @param dbConn Database connection. If given, the data will be extracted from
#'   the internal ADM Factory table unless the \code{forceUseDM} flag is set.
#' @param forceUseDM Only in combination with the \code{dbConn} flag. If TRUE,
#'   will use the ADM datamart tables. If FALSE (the default) will use the ADM
#'   Factory tables.
#' @param appliesToFilter Optional filter for the applies to class of the models
#'   as a regexp passed on to \code{\link[base]{grepl}} with \code{ignore.case =
#'   TRUE}, \code{perl = TRUE}, \code{fixed = FALSE}, \code{useBytes = FALSE}.
#' @param ruleNameFilter Optional filter for the rule name of the models as a
#'   regexp passed on to \code{\link[base]{grepl}} with \code{ignore.case =
#'   TRUE}, \code{perl = TRUE}, \code{fixed = FALSE}, \code{useBytes = FALSE}.
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
#' adm2pmml(verbose=TRUE)
#' \dontrun{
#'   models <- readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots","~/Downloads")
#'   preds <- readDSExport("Data-Decision-ADM-PredictorBinningSnapshot_AllPredictorSnapshots","~/Downloads")
#'
#'   adm2pmml(models, preds, verbose=TRUE)
#' }
adm2pmml <- function(dmModels = NULL, dmPredictors = NULL, dbConn = NULL, forceUseDM = FALSE,
                     appliesToFilter = NULL, ruleNameFilter = NULL,
                     destDir = ".",
                     tmpDir = NULL, verbose = (!is.null(tmpDir)))
{
  generatedPMMLFiles <- list()

  # DM dataset exports
  if (!is.null(dmModels) & !is.null(dmPredictors)) {
    setnames(dmModels, tolower(names(dmModels)))
    setnames(dmPredictors, tolower(names(dmPredictors)))

    if(!is.null(appliesToFilter)) {
      dmModels <- dmModels[ grepl(appliesToFilter, pyappliestoclass, ignore.case=T, perl=T) ]
    }
    if(!is.null(ruleNameFilter)) {
      dmModels <- dmModels[ grepl(ruleNameFilter, pyconfigurationname, ignore.case=T, perl=T) ]
    }
  } else {
    # Connection --> Datamart
    if (!is.null(dbConn)) {
      dmModels <- getModelsForClassFromDatamart(conn, verbose=verbose)
      if(!is.null(appliesToFilter)) {
        dmModels <- dmModels[ grepl(appliesToFilter, pyappliestoclass, ignore.case=T, perl=T) ]
      }
      if(!is.null(ruleNameFilter)) {
        dmModels <- dmModels[ grepl(ruleNameFilter, pyconfigurationname, ignore.case=T, perl=T) ]
      }
      dmPredictors <- getPredictorsForModelsFromDatamart(conn, dmModels, verbose=verbose)
    }
  }

  # ADM datamart, either from DB or from DS exports
  if (!is.null(dmModels) & !is.null(dmPredictors)) {
    # Iterate over unique model rules (model configurations). Every single one of these results in a seperate PMML file.
    modelRules <- unique(dmModels[, c("pyconfigurationname", "pyappliestoclass", "pxapplication"), with=F])

    if(nrow(modelRules)>0) {
      for (m in seq(nrow(modelRules))) {
        modelPartitionName <- modelRules$pyConfigurationName[m]

        modelPartitionFullName <- paste(modelRules$pxapplication[m], modelRules$pyappliestoclass[m], modelRules$pyconfigurationname[m], sep="_")  # get all model "partitions" for this model rule
        # pmmlFileName <- paste(destDir, paste(modelRules$pyConfigurationName[m], "ds", "pmml", "xml", sep="."), sep="/")
        pmmlFileName <- paste(destDir, paste(modelRules$pyconfigurationname[m], "pmml", sep="."), sep="/")

        if (verbose) {
          print(paste("Processing", modelPartitionFullName))
        }

        # Each of the model instances is an instantiation of a model rule for a specific set of context key values
        modelInstances <- unique(dmModels[pyconfigurationname==modelRules$pyconfigurationname[m] &
                                            pyappliestoclass==modelRules$pyappliestoclass[m] &
                                            pxapplication==modelRules$pxapplication[m]])

        modelList <- createListFromDatamart(dmPredictors[pymodelid %in% modelInstances$pymodelid],
                                            modelPartitionFullName, tmpDir, modelsForPartition=modelInstances)

        pmml <- createPMML(modelList, modelPartitionFullName)

        sink(pmmlFileName)
        print(pmml)
        sink()

        generatedPMMLFiles[[pmmlFileName]] <- unique(modelInstances$pymodelid)

        if (verbose) {
          print(paste("Generated",pmmlFileName,paste("(",length(modelList)," models)", sep="")))
        }
      }
    }
  }

  if (length(generatedPMMLFiles) == 0) {
    if (verbose) {
      print("No models available matching the filter criteria")
    }
    return(generatedPMMLFiles)
  } else {
    return(generatedPMMLFiles[sort(names(generatedPMMLFiles))])
  }
}
