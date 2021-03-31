# Abbreviate lengthy names
plotsAbbreviateName <- function(str, len = 32)
{
  parts <- strsplit(str,".",fixed=T)
  if (length(parts[[1]]) == 1) return(str)
  rhs <- paste(parts[[1]][2:length(parts[[1]])],collapse=".")
  if (nchar(rhs) < len) return(str)
  return(paste(parts[[1]][1], stringi::stri_sub(rhs,-len,-1), sep="..."))
}

# Rewrite interval label with only limited precision
plotsAbbreviateInterval <- function(str)
{
  # split inputs by space, abbreviate numbers in the elements (if matching,
  # otherwise they remain untouched), and paste them back together again
  regexpNumber <- "(.*)([[:digit:]]+)([.])([[:digit:]]{1,3})[[:digit:]]*(.*)"
  sapply(strsplit(str, " ", fixed=T), function(v) {paste(gsub(regexpNumber, "\\1\\2\\3\\4\\5", v), collapse = " ")} )
}

# Build faceting expression
plotsGetFacets <- function(facets, scales = "free")
{
  if (length(facets) == 0) return(NULL)
  if (length(facets) == 1 && facets == "") return(NULL)

  facettingExpr <- paste( ifelse(length(setdiff(facets, facets[1]))==0,".",paste(setdiff(facets, facets[1]), collapse="+")), "~", facets[1] )

  return(facet_wrap(facettingExpr, scales=scales))
}

# Get plot title and subtitle adding in description of the facets
plotsGetTitles <- function(baseTitle, aggregation = c(), facets = c(), limit = NA)
{
  if (length(aggregation) == 0 || (length(aggregation) == 1 && aggregation == "")) {
    aggregationSpecs <- NULL
  } else {
    aggregationSpecs <- paste(aggregation, collapse="/")
  }

  if (length(facets) == 0 || (length(facets) == 1 && facets == "")) {
    facetSpecs <- NULL
  } else {
    facetSpecs <- paste("by", paste(facets, collapse=" x "))
  }

  if (is.null(limit) || is.na(limit) || limit == .Machine$integer.max) {
    limitSpecs <- NULL
  } else {
    limitSpecs <- paste("(Top", limit, ")")
  }

  return(ggplot2::labs(title=baseTitle,
                       subtitle = paste(c(aggregationSpecs, facetSpecs, limitSpecs))))
}

# Verify the provided list of faceting names, reduce if not available
plotsCheckFacetsExist <- function(data, facets)
{
  if (length(facets) == 0) return(c())
  if (length(facets) == 1 && facets == "") return(c())

  if (length(setdiff(facets, names(data))) > 0) {
    warning(paste("Not all facets present in provided data,", setdiff(facets, names(data)), "will be ignored."))

    facets <- intersect(facets, names(data))
  }

  return (facets)
}

# default splits by dot
defaultPredictorClassification <- function(names)
{
  return (sapply(strsplit(names, ".", fixed=T),
                 function(x) {
                   return(ifelse(length(x)<2, "Primary", x[[1]]))
                 }))
}

# older versions don't use PredictorType but Type
getPredictorType <- function(data)
{
  if (!is.null(data$PredictorType)) {
    return (data$PredictorType)
  }
  if (!is.null(data$Type)) {
    return (data$Type)
  }
  return(NULL)
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
#'   print (plotADMVarImp(varimp))
#' }
plotADMVarImp <- function(varimp)
{
  facets <- setdiff(names(varimp), c("PredictorName", "Importance", "Rank"))

  ggplot(varimp, aes(Importance, factor(Rank, levels=rev(sort(unique(Rank)))),
                     fill=Importance)) +
    geom_col() +
    geom_text(aes(x=0,label=PredictorName),size=3,hjust=0,colour="black")+
    guides(fill=F) +
    ylab("") +
    plotsGetFacets(facets) +
    plotsGetTitles(paste(ifelse(length(facets)==0, "Global", ""), "Predictor Importance"), "", facets)
}


# try-out for plotADMVarImp: different palettes for different predictor categories
# could work - todo take base colors from dicrete colors of category
# then shade each of these from max to min, use rank to select between
# xxx <- varimp[ConfigurationName=="Handset"]
# xxx[,ConfigurationNames:=NULL]
# xxx[,Category:=defaultPredictorClassification(PredictorName)]
# xxx[,ColorRank := frank(-Importance, ties.method = "first"), by=Category] # + facets
# xxx[,Color := hcl.colors(.N, palette = c("Blues", "Greens", "Purples", "Reds")[.GRP]), by=Category] # + facets
#
# ggplot(xxx, aes(Importance, factor(Rank, levels=rev(sort(unique(Rank)))),
#                 fill=factor(Rank, levels=rev(sort(unique(Rank)))))) +
#   geom_col() +
#   geom_text(aes(x=0,label=PredictorName),size=3,hjust=0,colour="black")+
#   scale_fill_manual(values = xxx$Color)+
#   guides(fill=F)


#' Create bubble chart from ADM datamart model data.
#'
#' Creates \code{ggplot} scatterplot from success rate x performance of the
#' models.
#'
#' @param modeldata ADM Datamart model data.
#' @param aggregation Vector of field names to aggregate by. Defaults to
#' "Issue","Group","Name","Treatment" (if present).
#' @param facets Optional vector of fields for faceting. Defaults to empty.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMPerformanceSuccessRateBubbleChart(mdls, facets = "ConfigurationName")
#' }
plotADMPerformanceSuccessRateBubbleChart <- function(modeldata, aggregation=intersect(c("Issue","Group","Name","Treatment"), names(modeldata)), facets = c())
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  ggplot(modeldata, aes(100*Performance, Positives/ResponseCount, colour=Name, size=ResponseCount)) +
    geom_point(alpha=0.7) +
    guides(colour=FALSE, size=FALSE)+
    scale_x_continuous(limits = c(50, 100), name = "AUC") +
    scale_y_continuous(limits = c(0, NA), name = "Success Rate", labels = scales::percent) +
    plotsGetFacets(facets) +
    plotsGetTitles("Performance vs Success Rate", aggregation, facets)
}


#' Create box plot of performance x success rate from ADM datamart model data.
#'
#' @param modeldata ADM Datamart model data.
#' @param primaryCriterion Field to separate by; there will be separate box plots
#' for each of the values of this field. Can only be a single field. Defaults
#' to "ConfigurationName".
#' @param facets Optional vector of additional fields for faceting. Defaults to empty.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMPerformanceSuccessRateBoxPlot(admdatamart_models, "ConfigurationName") +
#'   scale_x_continuous(limits = c(50, 70), name = "Performance") + coord_flip()
#' }
plotADMPerformanceSuccessRateBoxPlot <- function(modeldata, primaryCriterion = "ConfigurationName", facets = c())
{
  facets <- plotsCheckFacetsExist(modeldata, facets)
  primaryCriterionSym <- rlang::sym(primaryCriterion)

  ggplot(modeldata, aes(100*Performance, Positives/ResponseCount, colour=!!primaryCriterionSym)) +
    geom_boxplot(alpha=0.7, lwd=1) +
    scale_x_continuous(limits = c(50, 100), name = "AUC") +
    scale_y_continuous(limits = c(0, NA), name = "Success Rate", labels = scales::percent) +
    plotsGetFacets(facets) +
    plotsGetTitles("Performance vs Success Rate", primaryCriterion, facets)
}

#' Plot ADM Model Performance over time
#'
#' @param modeldata ADM Datamart model data.
#' @param aggregation Vector of field names to aggregate by. There will be a
#' trend line for each combination of these fields. Defaults to
#' "Issue","Group","Name","Treatment" (if present).
#' @param facets Optional vector of additional fields for faceting. Defaults to
#' "ConfigurationName"
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMModelSuccessRateOverTime(mdls)
#' }
plotADMModelPerformanceOverTime <- function(modeldata,
                                            aggregation=intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                            facets=c("ConfigurationName"))
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  plotdata <- modeldata[!is.na(SnapshotTime), .(Performance = 100*stats::weighted.mean(Performance, ResponseCount),
                                                ResponseCount = max(ResponseCount)), by=c(facets, aggregation, "SnapshotTime")]
  plotdata$Proposition <- apply(plotdata[, aggregation, with=F], 1, paste, collapse="/")

  # order by final performance
  propositionOrder <- plotdata[, .(Performance = stats::weighted.mean(Performance[which.max(SnapshotTime)],
                                                                      ResponseCount[which.max(SnapshotTime)], na.rm = T)), by=Proposition][order(-Performance)]$Proposition

  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  ggplot(plotdata, aes(SnapshotTime, Performance, color=Proposition)) +
    geom_line() +
    xlab("") + ylab("AUC") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    plotsGetFacets(facets, scales="free_x") +
    plotsGetTitles("Model Performance over Time", aggregation, facets)
}

#' Plot ADM Model Success Rate over time
#'
#' @param modeldata ADM Datamart model data.
#' @param aggregation Vector of field names to aggregate by. There will be a
#' trend line for each combination of these fields. Defaults to
#' "Issue","Group","Name","Treatment" (if present).
#' @param facets Optional vector of additional fields for faceting. Defaults to
#' "ConfigurationName"
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMModelSuccessRateOverTime(mdls)
#' }
plotADMModelSuccessRateOverTime <- function(modeldata,
                                            aggregation=intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                            facets=c("ConfigurationName"))
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  plotdata <- modeldata[!is.na(SnapshotTime), .(SuccessRate = stats::weighted.mean(Positives/ResponseCount, ResponseCount),
                                                ResponseCount = max(ResponseCount)), by=c(facets, aggregation, "SnapshotTime")]
  plotdata$Proposition <- apply(plotdata[, aggregation, with=F], 1, paste, collapse="/")

  # order by final success rate
  propositionOrder <- plotdata[, .(SuccessRate = stats::weighted.mean(SuccessRate[which.max(SnapshotTime)],
                                                                      ResponseCount[which.max(SnapshotTime)], na.rm = T)), by=Proposition][order(-SuccessRate)]$Proposition

  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  ggplot(plotdata, aes(SnapshotTime, SuccessRate, color=Proposition)) +
    geom_line() +
    xlab("") +
    scale_y_continuous(limits = c(0, NA), name = "Success Rate", labels = scales::percent) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    plotsGetFacets(facets, scales="free_x") +
    plotsGetTitles("Model Success Rate over Time", aggregation, facets)
}

#' Create boxplots per predictor with performance range in models.
#'
#' An additional diamond is added to the boxplots to indicate weighted
#' performance (by response counts in the models).
#'
#' @param predictordata ADM Datamart predictor data. Binning data not used and
#' can be left out.
#' @param modeldata Optional ADM Datamart model data, necessary when using
#' model facets like Channel, ConfigurationName etc.
#' @param facets Optional vector of additional fields for faceting. Defaults to
#' "ConfigurationName" if model data present, otherwise empty.
#' @param limit Optional limit of number of predictors. Useful if there are
#' many predictors to keep the plot readable.
#' @param predictorClassifier Optional function that returns the predictor
#' categories from the names. Defaults to a function that returns the text
#' before the first dot.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMPredictorPerformance(modelPredictorBins, limit = 40) + xlab("AUC")
#' }
plotADMPredictorPerformance <- function(predictordata,
                                        modeldata = NULL,
                                        facets = ifelse(is.null(modeldata), "", "ConfigurationName"),
                                        limit = .Machine$integer.max,
                                        predictorClassifier = defaultPredictorClassification)
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  # predictor performance for latest snapshots - this should work regardless whether
  # the input data is per bin or is for only one snapshot
  plotdata <- predictordata[EntryType != "Classifier" & Positives > 0,
                            .(Performance = 100*Performance[which.max(SnapshotTime)],
                              Positives = Positives[which.max(SnapshotTime)],
                              Negatives = Negatives[which.max(SnapshotTime)],
                              ResponseCount = ResponseCount[which.max(SnapshotTime)]),
                            by=c("ModelID", "PredictorName",
                                 # older versions did not have PredictorType
                                 ifelse(("PredictorType" %in% names(predictordata)), "PredictorType", "Type"),
                                 "EntryType")]
  if(!"PredictorType" %in% names(plotdata)) {
    plotdata[, PredictorType := Type]
    plotdata[, Type := NULL]
  }

  # abbreviate lengthy predictor names
  plotdata[, PredictorName := factor(PredictorName)]
  plotdata[, predictorname_ori := PredictorName]
  levels(plotdata$PredictorName) <- sapply(levels(plotdata$PredictorName), plotsAbbreviateName)
  if (length(unique(levels(plotdata$PredictorName))) != length(unique(levels(plotdata$predictorname_ori)))) {
    # plotsAbbreviateNameiation would loose predictors, revert
    plotdata[, PredictorName := predictorname_ori]
  }
  plotdata[, PredictorName := sapply(as.character(PredictorName), plotsAbbreviateName)]

  # join with model data (for facetting)
  if (!is.null(modeldata)) {
    plotdata <- merge(plotdata, unique(modeldata[, c("ModelID", facets), with=F]), by="ModelID", all.x=F, all.y=F)
  }

  # set predictor order (globally) and category
  # predictorOrder <- plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
  #                           by=PredictorName][order(-meanPerf)] $ PredictorName
  predictorOrder <- plotdata[, .(meanPerf = mean(Performance, na.rm = T)),
                             by=PredictorName][order(-meanPerf)] $ PredictorName
  plotdata[, Category := predictorClassifier(PredictorName)]
  plotdata[, Type := factor(PredictorType)] #### todo also via a default function
  plotdata[, PredictorName := factor(PredictorName, levels=predictorOrder)]

  # add weighted mean
  plotdata[, weightedPerformance := stats::weighted.mean(Performance, ResponseCount, na.rm=T), by=c("PredictorName", facets)]

  plt <- ggplot(plotdata[as.integer(PredictorName) <= limit],
                aes(Performance, PredictorName)) +
    xlim(c(50,NA)) +
    scale_y_discrete(limits=rev, name="")

  suppressWarnings(
    if (length(unique(plotdata$Category))>1) {
      # use fill as well as line color
      plt <- plt + geom_boxplot(mapping = aes(fill=Category, linetype=Type, alpha=Type), lwd=0.5, outlier.size = 0.1) +
        scale_alpha_discrete(limits = rev, guide=F)
    } else {
      # only type available
      plt <- plt + geom_boxplot(mapping = aes(fill=Type), lwd=0.5, outlier.size = 0.1)
    }
  )

  # Add little diamonds for weighted performance
  plt <- plt +
    geom_point(mapping=aes(x=weightedPerformance,y=PredictorName),
               shape = 23, size = 2, fill ="white", inherit.aes=FALSE)

  plt <- plt +
    plotsGetFacets(facets) +
    plotsGetTitles("Predictor Performance", "", facets, limit)

  return (plt)
}

#' Performance of predictors per proposition
#'
#' Creates a visual grid of predictor names vs propositions (configurable) with
#' colors indicating the strength of the predictor for that proposition.
#'
#' @param predictordata ADM Datamart predictor data. Binning data not used and
#' can be left out.
#' @param modeldata ADM Datamart model data.
#' @param aggregation Vector of field names to aggregate by. Defaults to
#' "Issue","Group","Name","Treatment" (if present).
#' @param facets Vector of additional fields for faceting. Defaults to
#' "ConfigurationName".
#' @param limit Optional limit of number of predictors. Useful if there are
#' many predictors to keep the plot readable.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#'     plt <- plotADMPredictorPerformanceMatrix(modelPredictorBins, mdls, limit=50) +
#'     theme(axis.text.y = element_text(size=8),
#'     axis.text.x = element_text(size=8, angle = 45, hjust = 1),
#'     strip.text = element_text(size=8))
#'     print(plt)
#' }
plotADMPredictorPerformanceMatrix <- function(predictordata,
                                              modeldata,
                                              aggregation = intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                              facets = "ConfigurationName",
                                              limit = .Machine$integer.max)
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  myGoodness <- function(x)
  {
    minOK <- 0.52
    maxOK <- 0.85
    midPt <- 0.30
    return (ifelse(x < minOK, midPt*(x-0.50)/(minOK-0.50),
                   ifelse(x < maxOK, midPt+(1.0-midPt)*(x-minOK)/(maxOK-minOK),
                          1.0 - (x-maxOK)/(1.0-maxOK))))
  }

  plotdata <- merge(predictordata[EntryType != "Classifier",
                                  # prevent situation where some config data exists in the predictor data as well
                                  setdiff(names(predictordata), c(aggregation, facets)), with=F],
                    unique(modeldata[, c("ModelID", aggregation, facets), with=F]),
                    by="ModelID", all.x=F, all.y=F)[, .(Performance = stats::weighted.mean(Performance, 1+ResponseCount, na.rm = T),
                                                        ResponseCount = sum(ResponseCount)),
                                                    by=c("PredictorName", aggregation, facets)]


  # Propositions, order by performance
  plotdata$Proposition <- apply(plotdata[, aggregation, with=F], 1, paste, collapse="/")
  propositionOrder <-  plotdata[, .(meanPerf = stats::weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
                                by=Proposition][order(-meanPerf)] $ Proposition
  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  # Predictors, order by performance
  plotdata[, PredictorName := sapply(as.character(PredictorName), plotsAbbreviateName)]
  predictorOrder <- plotdata[, .(meanPerf = stats::weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
                             by=PredictorName][order(-meanPerf)] $ PredictorName
  plotdata[, PredictorName := factor(PredictorName, levels = predictorOrder)]

  # primaryCriterionOrder <- plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
  #                                         by=c(primaryCriterion)][order(-meanPerf)] [[primaryCriterion]]
  # plotdata[[primaryCriterion]] <- factor(plotdata[[primaryCriterion]], levels = primaryCriterionOrder)
  # primaryCriterion <- sym(primaryCriterion)

  ggplot(plotdata[as.integer(PredictorName) <= limit & as.integer(Proposition) <= limit], aes(Proposition, PredictorName)) +
    geom_raster(aes(fill=myGoodness(Performance))) +
    scale_fill_gradient2(low="red", mid="green", high="white", midpoint=0.50) +
    geom_text(aes(label=sprintf("%.2f",Performance)), size=2)+
    guides(fill=F) +
    xlab("") +
    scale_y_discrete(limits=rev, name="") +
    theme(axis.text.x = element_text(size=8, angle = 45, hjust = 1)) +
    plotsGetFacets(facets) +
    plotsGetTitles("Predictor Performance", aggregation, facets, limit)
}

#' Plot ADM Proposition Success Rates
#'
#' Creates a bar char of proposition success rates. Most useful if done per
#' channel or other field.
#'
#' @param modeldata ADM Datamart model data.
#' @param aggregation Vector of field names to aggregate by. Defaults to
#' "Issue","Group","Name","Treatment" (if present).
#' @param facets Vector of additional fields for faceting. Defaults to
#' "ConfigurationName".
#' @param limit Optional limit of number of propositions.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMPropositionSuccessRates(mdls[Positives > 10], limit=20, facets="Channel") +
#' scale_fill_continuous_divergingx()
#' }
plotADMPropositionSuccessRates <- function(modeldata,
                                           aggregation=intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                           facets="ConfigurationName",
                                           limit = .Machine$integer.max)
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  latestMdls <- modeldata[, .SD[which(SnapshotTime==max(SnapshotTime))], by=c("ModelID")]

  propSuccess <- latestMdls[, .(`Success Rate` = stats::weighted.mean(Positives/ResponseCount, ResponseCount, na.rm = T)),
                            by=c(aggregation, facets)]
  propSuccess[, `Success Rate` := ifelse(is.nan(`Success Rate`), 0, `Success Rate`)]
  propSuccess$Proposition <- apply(propSuccess[, aggregation, with=F], 1, paste, collapse="/")
  propSuccess[, PropositionRank := frank(-`Success Rate`, ties.method="first"), by=facets]

  # propSuccess[, Name := factor(Name, levels=rev(levels(Name)))]

  ggplot(propSuccess[PropositionRank <= limit],
         aes(`Success Rate`, factor(PropositionRank, levels=rev(sort(unique(PropositionRank)))), fill=`Success Rate`)) +
    geom_col() +
    # geom_text(aes(label=sprintf("%.2f%%", 100*`Success Rate`)), color="blue", hjust=0.5, size=2)+
    geom_text(aes(x=0, label=plotsAbbreviateName(Proposition)), color="black", hjust=0, size=2)+
    scale_x_continuous(labels=percent) +
    guides(fill=FALSE) +
    ylab("") +
    theme(axis.text.y = element_text(hjust = 1, size=6)) +
    plotsGetFacets(facets, scales="free_y") +
    plotsGetTitles("Proposition Success Rate", aggregation, facets, limit)
}

#' Create cum gains plot for the model classifier.
#'
#' @param binning Binning of the classifier (technically it could
#' be any predictor).
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMCumulativeGains(classifierBinning)
#' }
plotADMCumulativeGains <- function(binning)
{
  binning[, BinIndex := as.numeric(BinIndex)] # just in case
  setorder(binning, BinIndex)

  binning[, CumPositivesPct := rev(100.0*cumsum(rev(BinPositives))/sum(BinPositives))]
  binning[, CumVolumePct := rev(100.0*cumsum(rev(BinResponseCount))/sum(BinResponseCount))]

  secAxisFactor <- sum(binning$BinPositives)
  lastRow <- copy(binning[1,])[, c("CumPositivesPct", "CumVolumePct") := 0 ]

  ggplot(rbind(binning, lastRow), aes(CumVolumePct/100, CumPositivesPct/100)) +
    geom_ribbon(aes(ymin=CumVolumePct/100, ymax=CumPositivesPct/100), color = "steelblue3", size=0, fill="steelblue3", alpha=0.6) +
    geom_abline(slope = 1, linetype = "dashed", color = "grey") +
    #geom_area(color = "steelblue3", size=1, fill="steelblue3", alpha=0.6) +
    geom_line(color = "steelblue3", size=2) +
    geom_point(color = "black", size=1) +
    scale_x_continuous(labels = scales::percent, name = "% of Population", breaks = (0:10)/10, limits = c(0,1)) +
    scale_y_continuous(labels = scales::percent, name = "% of Total Positive Responses", limits = c(0,1),
                       sec.axis = sec_axis(~.*secAxisFactor, name = "Total Positive Responses")) +
    ggtitle("Cumulative Gains") +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Create cum lift plot for the model classifier.
#'
#' @param binning Binning of the classifier (technically it could
#' be any predictor).
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMCumulativeLift(classifierBinning)
#' }
plotADMCumulativeLift <- function(binning)
{
  binning[, BinIndex := as.numeric(BinIndex)] # just in case
  setorder(binning, BinIndex)

  binning[, CumPositivesPct := rev(100.0*cumsum(rev(BinPositives))/sum(BinPositives))]
  binning[, CumVolumePct := rev(100.0*cumsum(rev(BinResponseCount))/sum(BinResponseCount))]

  secAxisFactorBaseRate <- sum(binning$BinPositives)/(sum(binning$BinPositives) + sum(binning$BinNegatives))

  ggplot(binning, aes(CumVolumePct/100, CumPositivesPct/CumVolumePct)) +
    geom_ribbon(aes(ymin=1.0, ymax=CumPositivesPct/CumVolumePct), color = "steelblue3", size=0, fill="steelblue3", alpha=0.6) +
    geom_line(color = "steelblue3", size=2) +
    geom_point(color = "black", size=1) +
    scale_x_continuous(labels = scales::percent, name = "% of Population", breaks = (0:10)/10, limits = c(0,1)) +
    scale_y_continuous(name = "Lift", limits = c(1.0,NA),
                       sec.axis = sec_axis(~.*secAxisFactorBaseRate, labels = scales::percent, name = "Success Rate")) +
    ggtitle("Lift") +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Create binning plot for any predictor.
#'
#' Combined bar-line plot with bars indicating volume and lines propensity.
#'
#' @param binning Binning of the predictor.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMBinning(predictorbinning[PredictorName=="NetWealth"])
#' }
plotADMBinning <- function(binning)
{
  binning[, BinIndex := as.numeric(BinIndex)] # just in case
  setorder(binning, BinIndex)

  successRateMax <- max(binning$BinPositives/binning$BinResponseCount, na.rm = T)
  if (0 == successRateMax) { successRateMax <- 1 }
  secAxisFactor <- ceiling(max(binning$BinResponseCount)/successRateMax)

  ggplot(binning, aes(factor(BinIndex), BinPositives/BinResponseCount, group=1))+
    geom_col(aes(y=BinResponseCount/secAxisFactor), fill=ifelse(binning$EntryType[1]=="Inactive","darkgrey","steelblue3"))+
    geom_line(colour="orange", size=2)+geom_point()+
    geom_hline(data=binning[1,], mapping = aes(yintercept = Positives/(Positives+Negatives)),
               colour="orange", linetype="dashed") +
    scale_y_continuous(limits=c(0, successRateMax), name="Success Rate", labels=percent,
                       sec.axis = sec_axis(~.*secAxisFactor, name = "Responses"))+
    scale_x_discrete(name = "",
                     labels=ifelse(getPredictorType(binning) == "numeric",
                                   plotsAbbreviateInterval(binning$BinSymbol),
                                   ifelse(nchar(binning$BinSymbol) <= 25,
                                          binning$BinSymbol,
                                          paste(substr(binning$BinSymbol, 1, 25), "...")))) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
}

