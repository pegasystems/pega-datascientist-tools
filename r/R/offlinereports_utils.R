#' # if (nrow(classifierBinning) >= 1) {
#' #   binningTable <- classifierBinning[, c("BinIndex", "BinSymbol", "BinPositives", "BinNegatives", "CumVolumePct", "successratepct", "adjustedpropensity", "CumPositivesPct", "ZRatio", "Lift"), with=F]
#' #   setnames(binningTable, c("Index", "Bin", "Positives", "Negatives", "Cum. Total (%)", "Success Rate (%)", "Adjusted Propensity (%)", "Cum. Positives (%)", "Z-Ratio", "Lift (%)"))
#' #
#' #   kable(binningTable) %>% kable_styling()
#' # }
#'
#' #' Generic datamart reader.
#' #'
#' #' Function primarily used by the offline-model report notebooks. If recognized
#' #' as such will return model data with all snapshots, or predictor data for only
#' #' the latest snapshot.
#' #'
#' #' @param file Name of the file. Can be a dataset export, a csv, a zip or
#' #' a parquet file.
#' #'
#' #' @return A \code{data.table}
#' #' @export
#' readDatamartFromFile <- function(file)
#' {
#'   PredictorType <- Type <- SnapshotTime <- name <- N <- NULL # Trick to silence warnings from R CMD Check
#'
#'   if (!file.exists(file)) {
#'     warning(paste("File does not exist:", file))
#'     return(NULL)
#'   }
#'
#'   if (endsWith(file, ".zip")) {
#'     # checking if this looks like a generic DS export
#'     if (grepl("_.*_[0-9]{8}T[0-9]{6}_GMT.zip$", file)) {
#'       if (grepl("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_[0-9]{8}T[0-9]{6}_GMT.zip$", file)) {
#'
#'         # specialized model read, perhaps more efficient
#'         data <- readADMDatamartModelExport(file, latestOnly = F)
#'       } else if (grepl("Data-Decision-ADM-PredictorBinningSnapshot_pyADMPredictorSnapshots_[0-9]{8}T[0-9]{6}_GMT.zip$", file)) {
#'
#'         # specialized predictor read, perhaps more efficient
#'         data <- readADMDatamartPredictorExport(file, latestOnly = T, noBinning = F)
#'       } else {
#'
#'         # generic dataset read
#'         data <- readDSExport(file)
#'       }
#'     } else {
#'
#'       # read with fread, unzipping on-the-fly; this might work only on Linux/Mac; consider making configurable or on error handler
#'       data <- fread(cmd=paste("unzip -p", gsub(" ", "\\ ", file, fixed = T)))
#'     }
#'   } else if (endsWith(file, ".csv")) {
#'     # CSV read
#'     data <- fread(file)
#'   } else if (endsWith(file, ".parquet")) {
#'     # Parquet format
#'     data <- as.data.table(read_parquet(file))
#'   } else {
#'     stop(paste("Unknown source file type", file))
#'   }
#'
#'   # Set names in universal way. May have been done already, will be harmless to do twice.
#'   standardizeFieldCasing(data)
#'
#'   # in older versions "PredictorType" was called "Type"
#'   if("Type" %in% names(data) & !"PredictorType" %in% names(data)) {
#'     data[, PredictorType := Type]
#'   }
#'
#'   # some fields notoriously returned as char but are numeric
#'   for (f in intersect(c("Performance", "Positives", "Negatives"), names(data))) {
#'     if (!is.numeric(data[[f]])) {
#'       data[[f]] <- as.numeric(data[[f]])
#'     }
#'   }
#'
#'   # Excel exports sometimes screw up formatting of large numeric values - drop the comma used as thousands separators
#'   # NB not sure how generic this code will turn out to be
#'   for (f in intersect(c("BinNegatives", "BinPositives"), names(data))) {
#'     if (class(data[[f]]) == "character") {
#'       data[[f]] <- as.numeric(gsub(',','',data[[f]],fixed=T))
#'     }
#'   }
#'
#'   # Time (if not converted already)
#'   if (!is.POSIXt(data$SnapshotTime)) {
#'     data[, SnapshotTime := standardizedParseTime(SnapshotTime)]
#'   }
#'
#'   # # with customized model contexts, Name will be a JSON string, sanitize that a bit
#'   # mdls[, Name := trimws(gsub("\\.+", " ", make.names(paste(".", Name))))]
#'
#'   # Check for duplicate names - happens in some real life exports
#'   if (length(names(data)) != length(unique(names(data)))) {
#'     warning(paste("Duplicate names in", file, ":",
#'                   paste(as.character(data.table(name = names(data))[, .N,by=name][N>1]$name), collapse=", ")))
#'     return(NULL)
#'   }
#'
#'   return (data)
#' }
#'
#'
