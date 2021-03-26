# Abbreviate lengthy names
plotsAbbreviateName <- function(str, len = 32)
{
  parts <- strsplit(str,".",fixed=T)
  if (length(parts[[1]]) == 1) return(str)
  rhs <- paste(parts[[1]][2:length(parts[[1]])],collapse=".")
  if (nchar(rhs) < len) return(str)
  return(paste(parts[[1]][1], stringi::stri_sub(rhs,-len,-1), sep="..."))
}

plotsAbbreviateInterval <- function(str)
{
  # split inputs by space, abbreviate numbers in the elements (if matching,
  # otherwise they remain untouched), and paste them back together again
  regexpNumber <- "(.*)([[:digit:]]+)([.])([[:digit:]]{1,3})[[:digit:]]*(.*)"
  sapply(strsplit(str, " ", fixed=T), function(v) {paste(gsub(regexpNumber, "\\1\\2\\3\\4\\5", v), collapse = " ")} )
}

# Get faceting expression
plotsGetFacets <- function(facets, scales = "free")
{
  if (length(facets) == 0 || facets == "") {
    return(NULL)
  } else {
    facettingExpr <- paste( ifelse(length(setdiff(facets, facets[1]))==0,".",paste(setdiff(facets, facets[1]), collapse="+")), "~", facets[1] )
    return(facet_wrap(facettingExpr, scales=scales))
  }
}

# Get plot title and subtitle adding in description of the facets
plotsGetTitles <- function(baseTitle, baseSubTitle, facets)
{
  if (length(facets) == 0 || facets == "") {
    titles <- labs(title=baseTitle, subtitle = baseSubTitle)
  } else {
    titles <- labs(title=baseTitle, subtitle = paste(baseSubTitle, "by", paste(facets, collapse=" x ")))
  }

  return(titles)
}

# Verify the provided list of faceting names, reduce if not available
plotsCheckFacetsExist <- function(data, facets)
{
  if (length(facets) == 0 || facets == "") {
    facets <- c()
  } else {
    if (length(setdiff(facets, names(data))) > 0) {
      warning(paste("Not all facets present in provided data,", setdiff(facets, names(data)), "will be ignored."))
      facets <- intersect(facets, names(data))
    }
  }

  return (facets)
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
  facets <- setdiff(names(varimp), c("PredictorName", "Importance", "Rank", "ConfigurationNames"))

  plt <- ggplot(varimp, aes(Importance, factor(Rank, levels=rev(sort(unique(Rank)))),
                            fill=Importance)) +
    geom_col() +
    geom_text(aes(x=0,label=PredictorName),size=3,hjust=0,colour="black")+
    guides(fill=F) +
    ylab("") +
    plotsGetFacets(facets) +
    plotsGetTitles(paste(ifelse(length(facets)==0, "Global", ""), "Predictor Importance"), varimp$ConfigurationNames[1], facets)

  return(plt)
}

# Create a bubble chart of performance x success rate from ADM model data
plotADMPerformanceSuccessRateBubbleChart <- function(modeldata, facets = c())
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  plt <- ggplot(modeldata, aes(100*Performance, Positives/ResponseCount, colour=Name, size=ResponseCount)) +
    geom_point(alpha=0.7) +
    guides(colour=FALSE, size=FALSE)+
    scale_x_continuous(limits = c(50, 100), name = "AUC") +
    scale_y_continuous(limits = c(0, NA), name = "Success Rate", labels = scales::percent) +
    plotsGetFacets(facets) +
    plotsGetTitles("Performance vs Success Rate", paste(unique(modeldata$ConfigurationName), collapse=","), facets)

  return (plt)
}


# Create a box plot of performance x success rate from ADM model data
plotADMPerformanceSuccessRateBoxPlot <- function(modeldata, primaryCriterion, facets = c())
{
  facets <- plotsCheckFacetsExist(modeldata, facets)
  primaryCriterion <- sym(primaryCriterion)

  plt <- ggplot(modeldata, aes(100*Performance, Positives/ResponseCount, colour=!!primaryCriterion)) +
    geom_boxplot(alpha=0.7, lwd=1) +
    scale_x_continuous(limits = c(50, 100), name = "AUC") +
    scale_y_continuous(limits = c(0, NA), name = "Success Rate", labels = scales::percent) +
    plotsGetFacets(facets) +
    plotsGetTitles("Performance vs Success Rate", paste(unique(modeldata$ConfigurationName), collapse=","), facets)

  return (plt)
}

plotADMModelPerformanceOverTime <- function(modeldata,
                                            by_fields=intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                            facets=c("ConfigurationName"))
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  plotdata <- modeldata[!is.na(SnapshotTime), .(Performance = 100*weighted.mean(Performance, ResponseCount),
                                                ResponseCount = max(ResponseCount)), by=c(facets, by_fields, "SnapshotTime")]
  plotdata$Proposition <- apply(plotdata[, by_fields, with=F], 1, paste, collapse="/")

  # order by final performance
  propositionOrder <- plotdata[, .(Performance = weighted.mean(Performance[which.max(SnapshotTime)],
                                                               ResponseCount[which.max(SnapshotTime)], na.rm = T)), by=Proposition][order(-Performance)]$Proposition

  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  plt <- ggplot(plotdata, aes(SnapshotTime, Performance, color=Proposition)) +
    geom_line() +
    xlab("") + ylab("AUC") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    plotsGetFacets(facets, scales="free_x") +
    plotsGetTitles("Model Performance over Time", paste(unique(modeldata$ConfigurationName), collapse=","), facets)

  return (plt)
}

plotADMModelSuccessRateOverTime <- function(modeldata,
                                            by_fields=intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                            facets=c("ConfigurationName"))
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  plotdata <- modeldata[!is.na(SnapshotTime), .(SuccessRate = weighted.mean(Positives/ResponseCount, ResponseCount),
                                                ResponseCount = max(ResponseCount)), by=c(facets, by_fields, "SnapshotTime")]
  plotdata$Proposition <- apply(plotdata[, by_fields, with=F], 1, paste, collapse="/")

  # order by final success rate
  propositionOrder <- plotdata[, .(SuccessRate = weighted.mean(SuccessRate[which.max(SnapshotTime)],
                                                               ResponseCount[which.max(SnapshotTime)], na.rm = T)), by=Proposition][order(-SuccessRate)]$Proposition

  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  plt <- ggplot(plotdata, aes(SnapshotTime, SuccessRate, color=Proposition)) +
    geom_line() +
    xlab("") +
    scale_y_continuous(limits = c(0, NA), name = "Success Rate", labels = scales::percent) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    plotsGetFacets(facets, scales="free_x") +
    plotsGetTitles("Model Success Rate over Time", paste(unique(modeldata$ConfigurationName), collapse=","), facets)

  return (plt)
}

# default splits by dot
defaultPredictorClassification <- function(names)
{
  return (sapply(strsplit(names, ".", fixed=T),
                 function(x) {
                   return(ifelse(length(x)<2, "Primary", x[[1]]))
                 }))
}

plotADMPredictorPerformance <- function(predictordata,
                                        modeldata = NULL,
                                        facets = ifelse(is.null(modeldata), "", "ConfigurationName"),
                                        limit = .Machine$integer.max,
                                        predictorClassifier = defaultPredictorClassification)
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  # in older versions "PredictorType" was called "Type"
  if("Type" %in% names(predictordata) & !"PredictorType" %in% names(predictordata)) {
    predictordata[, PredictorType := Type]
  }

  predictorDataWithoutBinning <- predictordata[EntryType != "Classifier",
                           .(PredictorType = first(PredictorType),
                             EntryType = first(EntryType),
                             Performance = 100*first(Performance),
                             Negatives = first(Negatives),
                             Positives = first(Positives),
                             ResponseCount = sum(first(Negatives))+sum(first(Positives))),
                           by=c("ModelID", "SnapshotTime", "PredictorName")]

  # only keep latest snapshot if there are multiple
  if (length(unique(predictorDataWithoutBinning$SnapshotTime)) > 1) {
    predictorDataWithoutBinning <- predictorDataWithoutBinning[, .SD[which(SnapshotTime==max(SnapshotTime))], by=c("ModelID", "PredictorName")]
  }

  predictorDataWithoutBinning[, PredictorName := factor(PredictorName)]
  predictorDataWithoutBinning[, predictorname_ori := PredictorName]
  levels(predictorDataWithoutBinning$PredictorName) <- sapply(levels(predictorDataWithoutBinning$PredictorName), plotsAbbreviateName)

  # plotsAbbreviateNameiation would loose predictors, revert
  if (length(unique(levels(predictorDataWithoutBinning$PredictorName))) != length(unique(levels(predictorDataWithoutBinning$predictorname_ori)))) {
    predictorDataWithoutBinning[, PredictorName := predictorname_ori]
  }
  predictorDataWithoutBinning[, PredictorName := sapply(as.character(PredictorName), plotsAbbreviateName)]

  # join with model data
  if (!is.null(modeldata)) {
    predictorDataWithoutBinning <- merge(predictorDataWithoutBinning, unique(modeldata[, c("ModelID", facets), with=F]), by="ModelID", all.x=F, all.y=F)
  }

  predictorOrder <- predictorDataWithoutBinning[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
                            by=PredictorName][order(-meanPerf)] $ PredictorName

  predictorDataWithoutBinning[, Category := predictorClassifier(PredictorName)]
  predictorDataWithoutBinning[, Type := factor(PredictorType)] #### todo also via a default function
  predictorDataWithoutBinning[, PredictorName := factor(PredictorName, levels=predictorOrder)]

  plt <- ggplot(predictorDataWithoutBinning[as.integer(PredictorName) <= limit],
                aes(Performance, PredictorName)) +
    xlim(c(50,NA)) +
    scale_y_discrete(limits=rev, name="")

  suppressWarnings(
    if (length(unique(predictorDataWithoutBinning$Category))>1) {
      # use fill as well as line color
      plt <- plt + geom_boxplot(mapping = aes(fill=Category, linetype=Type, alpha=Type), lwd=0.5, outlier.size = 0.1) +
        scale_alpha_discrete(limits = rev, guide=F)
    } else {
      # only type available
      plt <- plt + geom_boxplot(mapping = aes(fill=Type), lwd=0.5, outlier.size = 0.1)
    }
  )

  plt <- plt +
    plotsGetFacets(facets) +
    plotsGetTitles("Predictor Performance", "", facets)

  return (plt)
}

plotADMPredictorPerformanceMatrix <- function(predictordata,
                                              modeldata,
                                              by_fields = intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
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

  plotdata <- merge(predictordata[EntryType != "Classifier"],
                    unique(modeldata[, c("ModelID", by_fields, facets), with=F]),
                    by="ModelID", all.x=F, all.y=F)[, .(Performance = weighted.mean(Performance, 1+ResponseCount, na.rm = T),
                                                        ResponseCount = sum(ResponseCount)),
                                                    by=c("PredictorName", by_fields, facets)]


  # Propositions, order by performance
  plotdata$Proposition <- apply(plotdata[, by_fields, with=F], 1, paste, collapse="/")
  propositionOrder <-  plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
                                by=Proposition][order(-meanPerf)] $ Proposition
  plotdata[, Proposition := factor(Proposition, levels=propositionOrder)]

  # Predictors, order by performance
  plotdata[, PredictorName := sapply(as.character(PredictorName), plotsAbbreviateName)]
  predictorOrder <- plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
                             by=PredictorName][order(-meanPerf)] $ PredictorName
  plotdata[, PredictorName := factor(PredictorName, levels = predictorOrder)]

  # primaryCriterionOrder <- plotdata[, .(meanPerf = weighted.mean(Performance, 1+ResponseCount, na.rm = T)),
  #                                         by=c(primaryCriterion)][order(-meanPerf)] [[primaryCriterion]]
  # plotdata[[primaryCriterion]] <- factor(plotdata[[primaryCriterion]], levels = primaryCriterionOrder)
  # primaryCriterion <- sym(primaryCriterion)

  plt <- ggplot(plotdata[as.integer(PredictorName) <= limit & as.integer(Proposition) <= limit], aes(Proposition, PredictorName)) +
    geom_raster(aes(fill=myGoodness(Performance))) +
    scale_fill_gradient2(low="red", mid="green", high="white", midpoint=0.50) +
    geom_text(aes(label=sprintf("%.2f",Performance)), size=2)+
    guides(fill=F) +
    xlab("") +
    scale_y_discrete(limits=rev, name="") +
    theme(axis.text.x = element_text(size=8, angle = 45, hjust = 1)) +
    plotsGetFacets(facets) +
    plotsGetTitles("Predictor Performance", "Performance across Propositions", facets)

  return(plt)
}

# by fields will be collapsed to the proposition name on the y axis, by default using Name + Treatment (if available)
plotADMPropositionSuccessRates <- function(modeldata,
                                           by_fields=intersect(c("Issue","Group","Name","Treatment"), names(modeldata)),
                                           facets="ConfigurationName",
                                           limit = .Machine$integer.max)
{
  facets <- plotsCheckFacetsExist(modeldata, facets)

  latestMdls <- modeldata[, .SD[which(SnapshotTime==max(SnapshotTime))], by=c("ModelID")]

  propSuccess <- latestMdls[, .(`Success Rate` = weighted.mean(Positives/ResponseCount, ResponseCount, na.rm = T)),
                            by=c(by_fields, facets)]
  propSuccess[, `Success Rate` := ifelse(is.nan(`Success Rate`), 0, `Success Rate`)]
  propSuccess$Proposition <- apply(propSuccess[, by_fields, with=F], 1, paste, collapse="/")
  propSuccess[, PropositionRank := frank(-`Success Rate`, ties.method="first"), by=facets]

  # propSuccess[, Name := factor(Name, levels=rev(levels(Name)))]

  plt <- ggplot(propSuccess[PropositionRank <= limit],
                aes(`Success Rate`, factor(PropositionRank, levels=rev(sort(unique(PropositionRank)))), fill=`Success Rate`)) +
    geom_col() +
    # geom_text(aes(label=sprintf("%.2f%%", 100*`Success Rate`)), color="blue", hjust=0.5, size=2)+
    geom_text(aes(x=0, label=plotsAbbreviateName(Proposition)), color="black", hjust=0, size=2)+
    scale_x_continuous(labels=percent) +
    guides(fill=FALSE) +
    ylab("") +
    theme(axis.text.y = element_text(hjust = 1, size=6)) +
    plotsGetFacets(facets, scales="free_y") +
    plotsGetTitles("Proposition Success Rate", ifelse(limit == .Machine$integer.max, "", paste("top", limit)), facets)

  return(plt)
}

plotADMCumulativeGains <- function(binsOneModel)
{
  ## TODO: perhaps assert assumptions on the binning provided
  ## - single snapshot, certain columns present, this is the classifier

  binsOneModel[, CumPositivesPct := rev(100.0*cumsum(rev(BinPositives))/sum(BinPositives))]
  binsOneModel[, CumVolumePct := rev(100.0*cumsum(rev(BinResponseCount))/sum(BinResponseCount))]

  secAxisFactor <- sum(binsOneModel$BinPositives)
  lastRow <- copy(binsOneModel[1,])[, c("CumPositivesPct", "CumVolumePct") := 0 ]
  plt <- ggplot(rbind(binsOneModel, lastRow), aes(CumVolumePct/100, CumPositivesPct/100)) +
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

  return(plt)
}

plotADMCumulativeLift <- function(binsOneModel)
{
  ## TODO: perhaps assert assumptions on the binning provided
  ## - single snapshot, certain columns present, this is the classifier

  binsOneModel[, CumPositivesPct := rev(100.0*cumsum(rev(BinPositives))/sum(BinPositives))]
  binsOneModel[, CumVolumePct := rev(100.0*cumsum(rev(BinResponseCount))/sum(BinResponseCount))]

  secAxisFactorBaseRate <- sum(binsOneModel$BinPositives)/(sum(binsOneModel$BinPositives) + sum(binsOneModel$BinNegatives))
  plt <- ggplot(binsOneModel, aes(CumVolumePct/100, CumPositivesPct/CumVolumePct)) +
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

  return(plt)
}

plotADMBinning <- function(binning)
{
  successRateMax <- max(binning$BinPositives/binning$BinResponseCount, na.rm = T)
  if (0 == successRateMax) { successRateMax <- 1 }
  secAxisFactor <- ceiling(max(binning$BinResponseCount)/successRateMax)

  p <- ggplot(binning, aes(factor(BinIndex), BinPositives/BinResponseCount, group=1))+
    geom_col(aes(y=BinResponseCount/secAxisFactor), fill=ifelse(binning$EntryType[1]=="Inactive","darkgrey","steelblue3"))+
    geom_line(colour="orange", size=2)+geom_point()+
    geom_hline(data=binning[1,], mapping = aes(yintercept = Positives/(Positives+Negatives)),
               colour="orange", linetype="dashed") +
    scale_y_continuous(limits=c(0, successRateMax), name="Success Rate", labels=percent,
                       sec.axis = sec_axis(~.*secAxisFactor, name = "Responses"))+
    scale_x_discrete(name = "",
                     labels=ifelse(binning$PredictorType == "numeric",
                                   plotsAbbreviateInterval(binning$BinSymbol),
                                   ifelse(nchar(binning$BinSymbol) <= 25,
                                          binning$BinSymbol,
                                          paste(substr(binning$BinSymbol, 1, 25), "...")))) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))

  return(p)
}

printADMPredictorInfo <- function(binning, extra = list())
{
  # predictorinfo <- as.list(modelPredictorBins[PredictorName==f][1])

  kvps <- list( "Univariate Performance (AUC)" = binning$Performance[1],
                "Status" = binning$EntryType[1],
                "Total Positives" = sum(binning$BinPositives),
                "Total Negatives" = sum(binning$BinNegatives),
                "Overall Propensity" = sprintf("%.2f%%",100*sum(binning$BinPositives)/sum(binning$BinResponseCount)),
                "Predictor Group" = binning$GroupIndex[1])

  kvps <- c(kvps, extra)

  cat(paste0("\n<p></p>## ", f, "\n<p></p>"))
  cat("\n<p></p>|Field|Value|\n")
  cat("|---|---|\n")

  for (key in names(kvps)) {
    cat(paste0("|", key, "|", kvps[[key]],"|\n"))
  }

  cat("<p></p>")
}

userFriendlyADMBinning <- function(bins)
{
  if (bins$EntryType[1] == "Classifier") {
    return (data.table( Index = bins$BinIndex,
                        Bin = bins$BinSymbol,
                        Positives = bins$BinPositives,
                        Negatives = bins$BinNegatives,
                        `Cum. Total (%)` = rev(100.0*cumsum(rev(bins$BinResponseCount))/sum(bins$BinResponseCount)),
                        `Success Rate (%)` =  100*bins$BinPositives/bins$BinResponseCount,
                        `Adjusted Propensity (%)` = 100*(0.5+bins$BinPositives)/(1+bins$BinResponseCount),
                        `Cum. Positives (%)` = rev(100.0*cumsum(rev(bins$BinPositives))/sum(bins$BinPositives)),
                        `Z-Ratio` = bins$ZRatio,
                        `Lift (%)` = 100*bins$Lift) )
  } else {
    return (data.table( Index = bins$BinIndex,
                      Bin = bins$BinSymbol,
                      Positives = bins$BinPositives,
                      Negatives = bins$BinNegatives,
                      `Success Rate (%)` =  100*bins$BinPositives/(bins$BinResponseCount),
                      `Z-Ratio` = bins$ZRatio,
                      `Lift (%)` = 100*bins$Lift) )
  }
}

# if (nrow(classifierBinning) >= 1) {
#   binningTable <- classifierBinning[, c("BinIndex", "BinSymbol", "BinPositives", "BinNegatives", "CumVolumePct", "successratepct", "adjustedpropensity", "CumPositivesPct", "ZRatio", "Lift"), with=F]
#   setnames(binningTable, c("Index", "Bin", "Positives", "Negatives", "Cum. Total (%)", "Success Rate (%)", "Adjusted Propensity (%)", "Cum. Positives (%)", "Z-Ratio", "Lift (%)"))
#
#   kable(binningTable) %>% kable_styling()
# }

## very generic function to read from CSV, from zip and generally try its
## best to read data properly
## TODO move elsewhere
readDatamartFromFile <- function(file)
{
  if (!file.exists(file)) {
    warning(paste("File does not exist:", file))
    return(NULL)
  }

  if (endsWith(file, ".zip")) {
    # checking if this looks like a generic DS export
    if (grepl("_.*_[0-9]{8}T[0-9]{6}_GMT.zip$", file)) {
      if (grepl("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_[0-9]{8}T[0-9]{6}_GMT.zip$", file)) {

        # specialized model read, perhaps more efficient
        data <- readADMDatamartModelExport(file, latestOnly = F)
      } else if (grepl("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_[0-9]{8}T[0-9]{6}_GMT.zip$", file)) {

        # specialized predictor read, perhaps more efficient
        data <- readADMDatamartPredictorExport(file, latestOnly = T, noBinning = F)
      } else {

        # generic dataset read
        data <- readDSExport(file)
      }
    } else {

      # read with fread, unzipping on-the-fly; this might work only on Linux/Mac; consider making configurable or on error handler
      data <- fread(cmd=paste("unzip -p", gsub(" ", "\\ ", file, fixed = T)))
    }
  } else {

    # Plain read
    data <- fread(file)
  }

  # Set names in universal way. May have been done already, will be harmless to do twice.
  applyUniformPegaFieldCasing(data)

  # in older versions "PredictorType" was called "Type"
  if("Type" %in% names(data) & !"PredictorType" %in% names(data)) {
    data[, PredictorType := Type]
  }

  # some fields notoriously returned as char but are numeric
  for (f in c("Performance", "Positives", "Negatives")) {
    if (!is.numeric(data[[f]])) {
      data[[f]] <- as.numeric(data[[f]])
    }
  }

  # Excel exports sometimes screw up formatting of large numeric values - drop the comma used as thousands separators
  # NB not sure how generic this code will turn out to be
  for (f in intersect(c("BinNegatives", "BinPositives"), names(data))) {
    if (class(data[[f]]) == "character") {
      data[[f]] <- as.numeric(gsub(',','',data[[f]],fixed=T))
    }
  }

  # Time (if not converted already)
  if (!is.POSIXt(data$SnapshotTime)) {
    # try be smart about the date/time format - is not always Pega format in some of the database exports
    suppressWarnings(timez <- fromPRPCDateTime(data$SnapshotTime))
    if (sum(is.na(timez))/length(timez) > 0.2) {
      suppressWarnings(timez <- parse_date_time(data$SnapshotTime, orders=c("%Y-%m-%d %H:%M:%S", "%y-%b-%d") )) # TODO: more formats here
      if (sum(is.na(timez))/length(timez) > 0.2) {
        warning("Assumed Pega date-time string but resulting in over 20% NA's in snapshot time after conversion. Check that this is valid or update the code that deals with date/time conversion.")
      }
    }
    data[, SnapshotTime := timez]
  }

  # # with customized model contexts, Name will be a JSON string, sanitize that a bit
  # mdls[, Name := trimws(gsub("\\.+", " ", make.names(paste(".", Name))))]

  return (data)
}


