
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
#' and \code{Rank} plus columns for each of the \code{facets} if supplied. In
#' addition there is a column \code{ConfigurationNames} with a concatenation
#' of all the values for \code{ConfigurationName} in the original model data.
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

  # Iterate over all models,
  allbins <- lapply(unique(dmModels$ModelID), function(m) {
    if (nrow(dmPredictors[ModelID == m & EntryType == "Active"]) < 1) return (NULL)

    modelPartitionFullName <- paste(unlist(unique(dmModels[ModelID == m, c("AppliesToClass", "ConfigurationName", "ModelID")])), collapse = "_")

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

  allbins <- rbindlist(allbins, fill=T)

  # Merge in any fields that are requested for facetting the result
  if (!is.null(facets)) {
    allbins <- merge(allbins, unique(dmModels[, c("ModelID", facets), with=F]), by="ModelID")
  }

  # Calculate and scale the predictor weights
  varImp <- allbins[PredictorType != "CLASSIFIER", .(Importance = stats::weighted.mean(BinWeight/BinIndexOccurences, BinPos+BinNeg)), by=c("PredictorName",facets)][order(-Importance)]
  varImp[, Rank := frank(-Importance), by=facets]
  varImp[, Importance := 100.0*Importance/max(Importance), by=facets]

  # Return name of all model configurations as a constant field value
  varImp[, ConfigurationNames := paste(sort(unique(dmModels$ConfigurationName)), collapse=",")]

  return(varImp)
}

#' A plot from the ADM variable importance data.
#'
#' Creates a \code{ggplot} object from the provided ADM variable importance
#' \code{data.table} typically created by \code{admVarImp}.
#'
#' @param varimp \code{data.table} with the variable importance.
#'
#' @return A \code{ggplot} object that can directly be printed.
#' @export
#'
#' @examples
#' \dontrun{
#'   models <- readADMDatamartModelExport("~/Downloads")
#'   preds <- readADMDatamartPredictorExport("~/Downloads", noBinning = F)
#'
#'   varimp <- admVarImp(models, preds)
#'   print (admVarImpPlot(varimp))
#' }
admVarImpPlot <- function(varimp)
{
  facets <- setdiff(names(varimp), c("PredictorName", "Importance", "Rank", "ConfigurationNames"))

  plt <- ggplot(varimp, aes(Importance, factor(Rank, levels=rev(sort(unique(Rank)))), fill=Importance)) +
    geom_col() +
    geom_text(aes(x=0,label=PredictorName),size=3,hjust=0,colour="black")+
    guides(fill=F)
  if (length(facets) == 0) {
    plt <- plt +
      labs(title="Global Predictor Importance", subtitle = varimp$ConfigurationNames[1], y="")
  } else {
    plt <- plt +
      # create facet_grid expression first facet ~ other facets
      facet_grid( paste( ifelse(length(setdiff(facets, facets[1]))==0,".",paste(setdiff(facets, facets[1]), collapse="+")), "~", facets[1] ), scales="free") +
      labs(title="Predictor Importance", subtitle = paste(varimp$ConfigurationNames[1], "by", paste(facets, collapse=" x ")), y="")
  }

  return(plt)
}
