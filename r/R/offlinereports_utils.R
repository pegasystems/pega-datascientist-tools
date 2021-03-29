#' HTML snippet with predictor info.
#'
#' Prints (cat) HTML snippet with predictor level info from a \code{data.table} with
#' the predictor binning.
#'
#' @param binning Binning of the predictor.
#' @param extra Optional list with extra key-value pairs added to the list.
#'
#' @export
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

#' Creates user friendly table with binning info.
#'
#' @param bins Binning of the predictor.
#'
#' @return A \code{data.table} table with user friendly binning data.
#' @export
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

#' Generic datamart reader.
#'
#' Function primarily used by the offline-model report notebooks. If recognized
#' as such will return model data with all snapshots, or predictor data for only
#' the latest snapshot.
#'
#' @param file Name of the file. Can be a dataset export, a csv, a zip etc.
#'
#' @return A \code{data.table}
#' @export
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

  # Check for duplicate names - happens in some real life exports
  if (length(names(data)) != length(unique(names(data)))) {
    warning(paste("Duplicate names in", file, ":", sort(names(data))))
    return(NULL)
  }

  return (data)
}


