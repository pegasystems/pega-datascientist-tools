
# Colors picked from score distribution of the classifier of ADM
pegaClassifierBlueBar <- "#278DC1"
pegaClassifierYellowLine <- "#EF8B08"


# Abbreviate lengthy names
plotsAbbreviateName <- function(str, abbreviateLength)
{
  if (length(str) > 1) return(sapply(str, plotsAbbreviateName, abbreviateLength))

  parts <- strsplit(str,".",fixed=T)

  if (length(parts[[1]]) == 1) {
    if (nchar(str) > abbreviateLength) {
      substrlen <- abbreviateLength-ceiling((abbreviateLength-3)/2)-3
      if (substrlen < 1) {
        return(paste0(substr(paste0(str, "..."), 1, abbreviateLength)))
      } else {
        return(paste0(substr(str,1,ceiling((abbreviateLength-3)/2)), "...", stringi::stri_sub(str, -substrlen)))
      }
    } else {
      return(str)
    }
  }

  rhs <- paste(parts[[1]][2:length(parts[[1]])],collapse=".")
  if (nchar(rhs) < abbreviateLength) return(str)

  substrlen <- abbreviateLength - nchar(parts[[1]][[1]]) - 3
  if (substrlen < 1) {
    return(substr(paste0(parts[[1]][1], "..."), 1, abbreviateLength))
  }

  return(paste(parts[[1]][1], stringi::stri_sub(rhs, -substrlen, -1), sep="..."))
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
# TODO figure out how to use facet_grid then use first facet as x
plotsGetFacets <- function(facets, scales = "fixed")
{
  if (length(facets) == 0) return(NULL)
  if (length(facets) == 1 && facets == "") return(NULL)

  # facettingExpr <- paste( ifelse(length(setdiff(facets, facets[1]))==0,".",
  #                                paste(setdiff(facets, facets[1]), collapse="+")), "~", facets[1] )

  return(facet_wrap(facets, scales=scales))
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
    limitSpecs <- paste0("(Top ", abs(limit), ifelse(limit<0," Reversed",""), ")")
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

# Returns predictor category as the element before the first dot, or "TopLevel" if there is no dot.
defaultPredictorCategorization <- function(p)
{
  hasDot <- grepl(".", p, fixed = T)
  return (ifelse(hasDot, gsub("^([^.]*)\\..*$", "\\1", p), rep("TopLevel", length(p))))
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
  Name <- Performance <- Positives <- ResponseCount <- NULL # Trick to silence R CMD Check warnings

  facets <- plotsCheckFacetsExist(modeldata, facets)

  # TODO move color/size aes into geom_point so we can override
  # OR allow for something to pass as color, e.g. maturity
  # OhMyGoodness := pmin(cdhtools::auc2GINI(Performance), 0.7) + (1.0/0.7) * pmin(Positives,200) * (1.0/200)
  # TODO consider coloring by my goodness by default
  # figure out a way to remove the legend from plotly

  ggplot(modeldata, aes(100*Performance, Positives/ResponseCount, colour=Name, size=ResponseCount)) +
    geom_point(alpha=0.7) +
    guides(colour="none", size="none")+
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
  Performance <- Positives <- ResponseCount <- NULL # Trick to silence R CMD Check warnings

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
  SnapshotTime <- Performance <- ResponseCount <- Proposition <- NULL  # Trick to silence warnings from R CMD Check

  facets <- plotsCheckFacetsExist(modeldata, facets)

  plotdata <- modeldata[!is.na(SnapshotTime), .(Performance = 100*weighted.mean(Performance, ResponseCount),
                                                ResponseCount = max(ResponseCount)), by=c(facets, aggregation, "SnapshotTime")]
  plotdata$Proposition <- apply(plotdata[, aggregation, with=F], 1, paste, collapse="/")

  # order by final performance
  propositionOrder <- plotdata[, .(Performance = weighted.mean(Performance[safe_which_max(SnapshotTime)],
                                                                      ResponseCount[safe_which_max(SnapshotTime)], na.rm = T)), by=Proposition][order(-Performance)]$Proposition

  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  ggplot(plotdata, aes(SnapshotTime, Performance, color=Proposition)) +
    geom_line() +
    xlab("") + ylab("AUC") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    plotsGetFacets(facets) +
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
  SnapshotTime <- Positives <- ResponseCount <- SuccessRate <- Proposition <- NULL  # Trick to silence warnings from R CMD Check

  facets <- plotsCheckFacetsExist(modeldata, facets)

  plotdata <- modeldata[!is.na(SnapshotTime), .(SuccessRate = weighted.mean(Positives/ResponseCount, ResponseCount),
                                                ResponseCount = max(ResponseCount)), by=c(facets, aggregation, "SnapshotTime")]
  plotdata$Proposition <- apply(plotdata[, aggregation, with=F], 1, paste, collapse="/")

  # order by final success rate
  propositionOrder <- plotdata[, .(SuccessRate = weighted.mean(SuccessRate[safe_which_max(SnapshotTime)],
                                                                      ResponseCount[safe_which_max(SnapshotTime)], na.rm = T)), by=Proposition][order(-SuccessRate)]$Proposition

  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  ggplot(plotdata, aes(SnapshotTime, SuccessRate, color=Proposition)) +
    geom_line() +
    xlab("") +
    scale_y_continuous(limits = c(0, NA), name = "Success Rate", labels = scales::percent) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    plotsGetFacets(facets, scales="free_y") +
    plotsGetTitles("Model Success Rate over Time", aggregation, facets)
}

# TODO the order seems a bit weird, see default modeloverview.Rmd for example
# TODO may add weighted perf only optionally

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
#' many predictors to keep the plot readable. Use a negative value to swap the
#' order, so showing the least performing predictors first.
#' @param predictorClassifier Optional function that returns the predictor
#' categories from the names. Defaults to a function that returns the text
#' before the first dot.
#' @param categoryAggregateView If TRUE shows an overview by predictor categories
#' instead of the individual predictors.
#' @param maxNameLength Max length of predictor names. Above this length
#' they will be abbreviated.
#' @param showAsBoxPlot When TRUE (default), will show as a box plot and sort
#' by median. When FALSE will show as a simple bar chart and use the weighted
#' mean (weighted by response count, which is more accurate than the box plot
#' view).
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMPredictorImportance(modelPredictorBins, limit = 40) + xlab("AUC")
#' }
plotADMPredictorImportance <- function(predictordata,
                                       modeldata = NULL,
                                       facets = ifelse(is.null(modeldata), "", "ConfigurationName"),
                                       limit = .Machine$integer.max,
                                       predictorClassifier = defaultPredictorCategorization,
                                       categoryAggregateView = F,
                                       maxNameLength = .Machine$integer.max,
                                       showAsBoxPlot = T)
{
  Importance <- ImportanceMean <- ImportanceMedian <- ImportanceRank <-
    ResponseCount <- Performance <- PerformanceMean <- PerformanceMedian <-
    PerformanceRank <- Category <- PredictorName <- AbbreviatedPredictorName <-
    OriginalPredictorName <- NULL  # Trick to silence warnings from R CMD Check

  facets <- plotsCheckFacetsExist(modeldata, facets)

  if (showAsBoxPlot) {
    featureImportance <- admVarImp(modeldata, predictordata, facets = NULL)

    # Now join in any facets - if there are any
    if (!is.null(facets)) {
      featureImportance <- merge(featureImportance,
                                 unique(modeldata[, c(facets, "ModelID"), with=F]), by="ModelID")
    }

    featureImportance[, Importance := Importance*100.0/max(Importance), by=facets]
    featureImportance[, ImportanceMean := weighted.mean(Importance, ResponseCount), by=c("PredictorName", facets)]
    featureImportance[, PerformanceMean := weighted.mean(Performance, ResponseCount), by=c("PredictorName", facets)]
    featureImportance[, ImportanceMedian := median(Importance), by=c("PredictorName", facets)]
    featureImportance[, PerformanceMedian := median(Performance), by=c("PredictorName", facets)]
    featureImportance[, ImportanceRank := frank(-ImportanceMedian, ties.method="dense"), by=facets]
    featureImportance[, PerformanceRank := frank(-PerformanceMedian, ties.method="dense"), by=facets]
  } else {
    featureImportance <- admVarImp(modeldata, predictordata, facets = facets)
  }

  # Predictor Category
  # TODO: support a character that just represents the field name
  if (is.logical(predictorClassifier)) {
    if (predictorClassifier) {
      predictorClassifier <- defaultPredictorCategorization
    } else {
      predictorClassifier <- NULL
    }
  }
  nameMapping <- data.table( PredictorName = as.character(unique(featureImportance$PredictorName)) )
  if (is.function(predictorClassifier)) {
    nameMapping[, Category := sapply(PredictorName, predictorClassifier)]
  } else {
    if (is.character(predictorClassifier) | is.factor(predictorClassifier)) {
      if (predictorClassifier %in% names(predictordata)) {
        nameMapping <- merge(nameMapping, unique(predictordata[,c("PredictorName", predictorClassifier), with=F]), by="PredictorName")
        tmp <- nameMapping[[predictorClassifier]]
        nameMapping[[predictorClassifier]] <- NULL
        nameMapping[["Category"]] <- tmp
      } else {
        nameMapping[,Category := "Missing Category"]
        predictorClassifier <- NULL
      }
    } else {
      nameMapping[,Category := "Missing Category"]
      predictorClassifier <- NULL
    }
  }
  nameMapping[, AbbreviatedPredictorName := make.names(sapply(PredictorName, plotsAbbreviateName, maxNameLength), unique = T)]
  featureImportance <- merge(featureImportance, nameMapping, by="PredictorName")
  featureImportance[, OriginalPredictorName := PredictorName]
  featureImportance[, PredictorName := AbbreviatedPredictorName]
  featureImportance[, AbbreviatedPredictorName := NULL]

  if (nrow(featureImportance) == 0) {
    warning("No data to plot")
    return(NULL)
  }

  categoryOrder <- featureImportance[, .(ImportanceMedian = median(Importance)), by=Category][order(ImportanceMedian)]
  featureImportance[, Category := factor(Category, levels = categoryOrder$Category)]

  # Base plot
  if (categoryAggregateView) {
    plt <- ggplot(featureImportance, aes(Importance, Category))
  } else {
    # Note: the y-axis order is not completely correct. A predictor name has a fixed
    # rank in the levels of the factor, but it can occur at different ranks in the
    # different plots.
    setorder(featureImportance, ImportanceRank)
    featureImportance[, PredictorName := factor(PredictorName, levels=rev(unique(featureImportance$PredictorName)))]

    if (limit < 0) {
      featureImportance[, ImportanceRank := max(ImportanceRank) - ImportanceRank + 1]
    }
    plt <- ggplot(featureImportance[ImportanceRank <= abs(limit)], aes(Importance, PredictorName))
  }

  # Aesthetics
  if (showAsBoxPlot) {
    if (is.null(predictorClassifier)) {
      plt <- plt + geom_boxplot(aes(fill=ImportanceMean), lwd=0.5, outlier.size = 0.1, alpha=0.8) +
        scale_fill_continuous(guide="none")
    } else {
      plt <- plt + geom_boxplot(aes(fill=Category, alpha=ImportanceMean), lwd=0.5, outlier.size = 0.1) +
        scale_alpha_continuous(guide="none")
    }
  } else {
    if (is.null(predictorClassifier)) {
      plt <- plt + geom_col(aes(fill=ImportanceMean), alpha=0.8) +
        scale_fill_continuous(guide="none")
    } else {
      plt <- plt + geom_col(aes(fill=Category, alpha=Importance)) +
        scale_alpha_continuous(guide="none")
    }
  }

  # TODO consider Univariate flag using Performance instead of Importance
  # plt <- plt + xlim(c(50,NA)) +
  #   scale_y_discrete(limits=rev, name="")

  # Faceting
  plt <- plt + plotsGetFacets(facets)

  # Titles
  if (categoryAggregateView) {
    plt <- plt + plotsGetTitles("Predictor Category Importance", "", facets)
  } else {
    plt <- plt + plotsGetTitles("Predictor Importance", "", facets, limit)
  }

  return(plt)
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
#' @param maxNameLength Max length of predictor names. Above this length
#' they will be abbreviated.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#'     plt <- plotADMPredictorImportanceHeatmap(modelPredictorBins, mdls, limit=50) +
#'     theme(axis.text.y = element_text(size=8),
#'     axis.text.x = element_text(size=8, angle = 45, hjust = 1),
#'     strip.text = element_text(size=8))
#'     print(plt)
#' }
plotADMPredictorImportanceHeatmap <- function(predictordata,
                                             modeldata,
                                             aggregation = intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                             facets = "ConfigurationName",
                                             limit = .Machine$integer.max,
                                             maxNameLength = .Machine$integer.max)
{
  EntryType <- meanPerf <- Performance <- PredictorName <- Proposition <- ResponseCount <- NULL # Trick to silence R CMD Check warnings

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
                    by="ModelID", all.x=F, all.y=F)[, .(Performance = weighted.mean(Performance, 1+ResponseCount, na.rm = T),
                                                        ResponseCount = sum(ResponseCount)),
                                                    by=c("PredictorName", aggregation, facets)]


  # Propositions, order by performance
  plotdata$Proposition <- apply(plotdata[, aggregation, with=F], 1, paste, collapse="/")
  propositionOrder <-  plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
                                by=Proposition][order(-meanPerf)] $ Proposition
  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  # Predictors, order by performance
  plotdata[, PredictorName := sapply(as.character(PredictorName), plotsAbbreviateName, maxNameLength)]
  predictorOrder <- plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
                             by=PredictorName][order(-meanPerf)] $ PredictorName
  plotdata[, PredictorName := factor(PredictorName, levels = predictorOrder)]

  # primaryCriterionOrder <- plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
  #                                         by=c(primaryCriterion)][order(-meanPerf)] [[primaryCriterion]]
  # plotdata[[primaryCriterion]] <- factor(plotdata[[primaryCriterion]], levels = primaryCriterionOrder)
  # primaryCriterion <- sym(primaryCriterion)

  ggplot(plotdata[as.integer(PredictorName) <= limit & as.integer(Proposition) <= limit], aes(Proposition, PredictorName)) +
    geom_raster(aes(fill=myGoodness(Performance)), alpha=0.4) +
    scale_fill_gradient2(low=muted("red"), mid="green", high="white", midpoint=0.50) +
    geom_text(aes(label=sprintf("%.2f", Performance*100)), size=3)+
    guides(fill="none") +
    xlab("") +
    scale_y_discrete(limits=rev, name="") +
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
#' @param maxNameLength Max length of predictor names. Above this length
#' they will be abbreviated.
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
                                           limit = .Machine$integer.max,
                                           maxNameLength = .Machine$integer.max)
{
  Positives <- Proposition <- PropositionRank <- ResponseCount <- SnapshotTime <- `Success Rate` <- NULL # Trick to silence R CMD Check warnings

  facets <- plotsCheckFacetsExist(modeldata, facets)

  latestMdls <- modeldata[, .SD[which(SnapshotTime==max(SnapshotTime))], by=c("ModelID")]

  propSuccess <- latestMdls[, .(`Success Rate` = weighted.mean(Positives/ResponseCount, ResponseCount, na.rm = T)),
                            by=c(aggregation, facets)]
  propSuccess[, `Success Rate` := ifelse(is.nan(`Success Rate`), 0, `Success Rate`)]
  propSuccess$Proposition <- apply(propSuccess[, aggregation, with=F], 1, paste, collapse="/")
  propSuccess[, PropositionRank := frank(-`Success Rate`, ties.method="first"), by=facets]

  # propSuccess[, Name := factor(Name, levels=rev(levels(Name)))]
  ggplot(propSuccess[PropositionRank <= limit],
         aes(`Success Rate`, factor(PropositionRank, levels=rev(sort(unique(PropositionRank)))), fill=`Success Rate`)) +
    geom_col() +
    # geom_text(aes(label=sprintf("%.2f%%", 100*`Success Rate`)), color="blue", hjust=0.5, size=2)+
    geom_text(aes(x=0, label=plotsAbbreviateName(Proposition, maxNameLength)), color="black", hjust=0, size=2)+
    scale_x_continuous(labels=percent) +
    guides(fill="none") +
    ylab("") +
    theme(axis.text.y = element_text(hjust = 1, size=6)) +
    plotsGetFacets(facets, scales="free") +
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
  BinIndex <- CumPositivesPct <- BinPositives <- CumVolumePct <- BinResponseCount <- NULL  # Trick to silence warnings from R CMD Check

  binning[, BinIndex := as.numeric(BinIndex)] # just in case
  setorder(binning, BinIndex)

  binning[, CumPositivesPct := rev(100.0*cumsum(rev(BinPositives))/sum(BinPositives))]
  binning[, CumVolumePct := rev(100.0*cumsum(rev(BinResponseCount))/sum(BinResponseCount))]

  secAxisFactor <- sum(binning$BinPositives)
  lastRow <- copy(binning[1,])[, c("CumPositivesPct", "CumVolumePct") := 0 ]

  ggplot(rbind(binning, lastRow), aes(CumVolumePct/100, CumPositivesPct/100)) +
    geom_ribbon(aes(ymin=CumVolumePct/100, ymax=CumPositivesPct/100),
                color = pegaClassifierBlueBar, size=0, fill=pegaClassifierBlueBar, alpha=0.6) +
    geom_abline(slope = 1, linetype = "dashed", color = "grey") +
    #geom_area(color = "steelblue3", size=1, fill="steelblue3", alpha=0.6) +
    geom_line(color = pegaClassifierYellowLine, size=2) +
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
  BinIndex <- CumPositivesPct <- BinPositives <- CumVolumePct <- BinResponseCount <- NULL  # Trick to silence warnings from R CMD Check

  binning[, BinIndex := as.numeric(BinIndex)] # just in case
  setorder(binning, BinIndex)

  binning[, CumPositivesPct := rev(100.0*cumsum(rev(BinPositives))/sum(BinPositives))]
  binning[, CumVolumePct := rev(100.0*cumsum(rev(BinResponseCount))/sum(BinResponseCount))]

  secAxisFactorBaseRate <- sum(binning$BinPositives)/(sum(binning$BinPositives) + sum(binning$BinNegatives))

  ggplot(binning, aes(CumVolumePct/100, CumPositivesPct/CumVolumePct)) +
    geom_ribbon(aes(ymin=1.0, ymax=CumPositivesPct/CumVolumePct),
                color = pegaClassifierBlueBar, size=0, fill=pegaClassifierBlueBar, alpha=0.6) +
    geom_line(color = pegaClassifierYellowLine, size=2) +
    geom_point(color = "black", size=1) +
    scale_x_continuous(labels = scales::percent, name = "% of Population", breaks = (0:10)/10, limits = c(0,1)) +
    scale_y_continuous(name = "Lift", limits = c(1.0,NA),
                       sec.axis = sec_axis(~.*secAxisFactorBaseRate, labels = scales::percent, name = "Success Rate")) +
    ggtitle("Lift") +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1))
}

# TODO perhaps call this plotADMScoreDistribution as an alias

#' Create binning plot for any predictor.
#'
#' Combined bar-line plot with bars indicating volume and lines propensity.

#' @param binning Binning of the predictor.
#' @param useSmartLabels If true (default) it will abbreviate lengthy symbols
#' and for numerics it will re-create a shorter interval notation from the
#' lower and upper bounds. This will generally look better than the labels in
#' the binning. Turn it off to use the default bin labels. Smart labels do not work
#' well with additional faceting, so turn it off in that case as well.
#'
#' @return A \code{ggplot} object that can either be printed directly, or to
#' which additional decoration (e.g. coloring or labels) can be added first.
#' @export
#'
#' @examples
#' \dontrun{
#' plotADMBinning(predictorbinning[PredictorName=="NetWealth"])
#' }
plotADMBinning <- function(binning, useSmartLabels = T) # TODO consider adding laplaceSmoothing = F to add 0.5 and 1.0
{
  BinIndex <- BinPositives <- BinResponseCount <- BinSymbol <- Positives <- Negatives <- NULL  # Trick to silence warnings from R CMD Check

  binning[, BinIndex := as.numeric(BinIndex)] # just in case
  setorder(binning, BinIndex)

  responsesMax <- max(binning$BinResponseCount, na.rm = T)
  if (0 == responsesMax) { responsesMax <- 1 }
  secAxisFactor <- max(binning$BinPositives/binning$BinResponseCount) / responsesMax

  if (useSmartLabels) {
    plt <- ggplot(binning, aes(factor(BinIndex), BinPositives/BinResponseCount, group=1))
  } else {
    plt <- ggplot(binning, aes(BinSymbol, BinPositives/BinResponseCount, group=1))
  }

  plt <- plt +
    geom_col(aes(y=BinResponseCount),
             fill=ifelse(binning$EntryType[1]=="Inactive", "darkgrey", pegaClassifierBlueBar),
             alpha=0.8)+
    geom_line(aes(y=(BinPositives/BinResponseCount)/secAxisFactor),
              colour="orange", size=2) +
    geom_point(aes(y=(BinPositives/BinResponseCount)/secAxisFactor),
              size=1) +
    geom_hline(data=data.table(Positives = sum(binning$BinPositives),
                               Negatives = sum(binning$BinNegatives)),
               mapping = aes(yintercept = (Positives/(Positives+Negatives))/secAxisFactor),
               colour="orange", linetype="dashed") +
    scale_y_continuous(limits=c(0, NA), name="Responses",
                       sec.axis = sec_axis(~.*secAxisFactor, name = "Propensity (%)", labels=percent))+
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))

  if (useSmartLabels) {
    plt <- plt + scale_x_discrete(name = "",
                                  labels=ifelse(getPredictorType(binning) == "numeric",
                                                plotsAbbreviateInterval(binning$BinSymbol),
                                                ifelse(nchar(binning$BinSymbol) <= 25,
                                                       binning$BinSymbol,
                                                       paste(substr(binning$BinSymbol, 1, 25), "..."))))
  }

  plt
}

