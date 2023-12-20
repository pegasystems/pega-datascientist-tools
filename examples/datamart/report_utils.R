library(data.table)
library(markdown)
library(quarto)
library(pdstools)
library(arrow)
library(jsonlite)
library(lubridate)
library(R.utils)

# Sets a few global variables for the run_report functions. Call this first. Acts
# like a constructor in a proper language.
# TODO: keep global setting under just one list object. Perhaps even the
# various mapping functions (file <--> hash, report type --> filename).
init_report_utils <- function(pdstool_root_folder, results_folder, intermediates_folder = results_folder)
{
  report_utils_results_folder <<- results_folder
  report_utils_intermediates_folder <<- intermediates_folder

  # cache folder MUST be without spaces otherwise I can't get the quarto call to work
  if (grepl(" ", report_utils_results_folder, fixed = T) | grepl(" ", report_utils_intermediates_folder, fixed = T)) {
    stop(paste("Folders must not contain spaces:", report_utils_intermediates_folder, report_utils_results_folder))
  }

  # Pandoc is needed by RMarkdown and part of RStudio. If you run this
  # script outside of RStudio you'll need to make sure pandoc is installed
  # and known to R / markdown.
  if (!rmarkdown::pandoc_available()) {
    stop("Pandoc is not available. This is needed by RMarkdown and Quarto.")
  }

  if (!dir.exists(report_utils_results_folder)) {
    dir.create(report_utils_results_folder)
  }
  if (!dir.exists(report_utils_intermediates_folder)) {
    dir.create(report_utils_intermediates_folder)
  }

  # Markdown/Quatro files for the reports
  report_utils_healthcheck_notebook_R <<- file.path(pdstool_root_folder, "examples/datamart/healthcheck.Rmd")
  report_utils_healthcheck_notebook_python <<- file.path(pdstool_root_folder, "python/pdstools/reports/HealthCheck.qmd")
  report_utils_offlinemodelreport_notebook_R <<- file.path(pdstool_root_folder, "examples/datamart/modelreport.Rmd")
  report_utils_offlinemodelreport_notebook_python <<- file.path(pdstool_root_folder, "python/pdstools/reports/ModelReport.qmd")
}

# Names of cached files for a given customer
report_utils_cached_dm_filenames <- function(customer)
{
  # Escape space esp in customer names - quarto does not like spaces
  unspace <- function(c) { gsub(" ", "_", c, fixed = T) }

  return(c(
    "modeldata" = paste0(unspace(customer), "_ModelSnapshots", ".parquet"),
    "predictordata" = paste0(unspace(customer), "_PredictorSnapshots", ".parquet")
  ))
}

# Hash file associated with given file
# TODO consider making this a hidden file again, make sure the reverse function
# exists so cleanup works properly
report_utils_hashfilename <- function(f) {
  paste0(f, ".hash")
  # file.path(dirname(targetfiles), report_utils_hashfilename(targetfiles))
  # paste0(".",basename(f), ".hash")
}

# Write hash values in files ending with ".hash" as siblings to given files
report_utils_write_hashfiles <- function(targetfiles, hashvalue) {
  sapply(report_utils_hashfilename(targetfiles), function(f) {
    writeLines(hashvalue, f)
  })
}

# Returns true if and only if:
# - the given targetfiles exist
# - the given targetfiles with ".hash" appended and "." prefixed exist
# - the ".hash" files contain the expected hash
report_utils_is_target_current <- function(targetfiles, expectedhash, quiet) {
  targets <- paste(basename(targetfiles), collapse = ", ")

  if (!quiet) cat(targets, "exist:", sapply(targetfiles, file.exists), fill=TRUE)
  if (any(!sapply(targetfiles, file.exists))) {
    cat(targets, "do(es) not exist", fill = TRUE)

    return(FALSE)
  }

  hashfiles <- report_utils_hashfilename(targetfiles)

  if (!quiet) cat(hashfiles, "exist:", sapply(hashfiles, file.exists), fill=TRUE)
  if (any(!sapply(hashfiles, file.exists))) {
    cat("Hash file(s) for", targets, "do(es) not exist", fill = TRUE)

    return(FALSE)
  }

  hashcodes <- sapply(hashfiles, readLines)
  if (!quiet) cat("Actual hash codes:", hashcodes, fill=TRUE)
  if (!quiet) cat("Expected hash codes:", expectedhash, fill=TRUE)
  if (any(hashcodes != expectedhash)) {
    cat("Hash code mismatch for", targets, fill = TRUE)

    return(FALSE)
  }

  return(TRUE)
}

# write DM to cached parquet files
report_utils_write_cached_files <- function(dm, model_filename, preds_filename)
{
  # write cached data
  arrow::write_parquet(dm$modeldata, model_filename, compression = "uncompressed")
  if (is.null(dm$predictordata)) {
    file.create(preds_filename) # create empty, dummy file
  } else {
    arrow::write_parquet(dm$predictordata, preds_filename, compression = "uncompressed")
  }
}

# Drop old HTML files and orphaned hash files (that have no reference)
report_utils_cleanup_helper <- function(folder, keep_days, file_pattern, is_obsolete)
{
  generated_files <- list.files(folder,
                                pattern=file_pattern,
                                full.names = TRUE, recursive = FALSE)
  obsolete_files <- c()
  if (length(generated_files) > 0) {
    obsolete_files <- generated_files[sapply(generated_files, is_obsolete)]
  }

  # print(obsolete_files)

  cat("Removing", length(obsolete_files), "obsolete", file_pattern, "files from", folder, fill = T)
  if (length(obsolete_files) > 0) {
    file.remove(obsolete_files)
  }
}

report_utils_cleanup_cache <- function(folder = report_utils_results_folder, keep_days = 7)
{
  is_too_old <- function(f, before = now() - days(keep_days)) {
    return(fileModificationTime(f) < before)
  }

  has_no_reference <- function(f) {
    hashFileReference <- gsub("(.*)[.]hash$", "\\1", f) # reverse of report_utils_hashfilename
    return(!file.exists(hashFileReference))
  }

  report_utils_cleanup_helper(folder, keep_days, ".*[.]html$", is_too_old)
  report_utils_cleanup_helper(folder, keep_days, ".*[.]hash$", has_no_reference)

  # Do it for the report subfolders as well
  sapply(list.files(folder, pattern = ".*Generated Model Reports$", full.names = T, include.dirs = T),
         function(d) {
           report_utils_cleanup_helper(d, keep_days, ".*[.]html$", is_too_old)

           report_utils_cleanup_helper(d, keep_days, ".*[.]hash$", has_no_reference)
         }
  )

  return()
}



# Change time of a file
# TODO maybe this is overly sensitive on onedrive folders
fileModificationTime <- function(f) { as.POSIXct(file.info(f)$mtime) }

# Generic markdown/quarto call that will check hashes and dates to prevent
# unnecessary re-creation
report_utils_run_report <- function(customer, dm, target_fullfilename, target_generator_hash, renderer, quiet)
{
  cachedDMFilesFullName <- file.path(report_utils_intermediates_folder, report_utils_cached_dm_filenames(customer))

  # make sure cached source exist, otherwise re-create from dm data
  if (!all(sapply(cachedDMFilesFullName, file.exists))) {
    cat("Writing to parquet cache", dirname(cachedDMFilesFullName[1]), fill=T)
    report_utils_write_cached_files(dm, cachedDMFilesFullName[1], cachedDMFilesFullName[2])
  }

  # check if generator script has changed
  if (report_utils_is_target_current(target_fullfilename, target_generator_hash, quiet = quiet)) {
    if (!quiet) cat("Modification date", basename(target_fullfilename), ":", fileModificationTime(target_fullfilename), fill=T)
    if (!quiet) cat("Modification date DM data", report_utils_cached_dm_filenames(customer), ":", fileModificationTime(cachedDMFilesFullName), fill=T)

    # or if source files are newer
    if (is.na(fileModificationTime(target_fullfilename)) | any(is.na(fileModificationTime(cachedDMFilesFullName))) |
        any(fileModificationTime(cachedDMFilesFullName) > fileModificationTime(target_fullfilename))) {

      cat(target_fullfilename, "out of date wrt source files", fill = TRUE)

      doRegenerate <- TRUE
    } else {
      doRegenerate <- FALSE
    }
  } else {
    doRegenerate <- TRUE
  }

  if (doRegenerate) {
    cat("Creating", basename(target_fullfilename), fill = TRUE)

    title <- paste0(customer, ' - Adaptive Models')
    subtitle <- paste(unique(c(
      strftime(min(dm$modeldata$SnapshotTime), "%b %Y"),
      strftime(max(dm$modeldata$SnapshotTime), "%b %Y")
    )), collapse = " - ")

    # call the renderer
    renderer(basename(cachedDMFilesFullName[1]),
             ifelse(!is.null(dm$predictordata), basename(cachedDMFilesFullName[2]), ""),
             title,
             subtitle,
             target_fullfilename)

    # writer renderer hash
    report_utils_write_hashfiles(target_fullfilename, target_generator_hash)
  } else {
    # Touch target
    Sys.setFileTime(normalizePath(target_fullfilename), lubridate::now())
    Sys.setFileTime(report_utils_hashfilename(normalizePath(target_fullfilename)), lubridate::now())

    cat("Skipped re-generation of", basename(target_fullfilename), fill = T)
  }

  return(basename(target_fullfilename))
}

run_r_healthcheck <- function(customer, dm, quiet = T)
{
  r_health_check_hash <- digest::digest(readLines(report_utils_healthcheck_notebook_R), "sha256")

  report_utils_run_report(customer,
                          dm,
                          target_fullfilename = file.path(report_utils_results_folder,
                                                          paste0(customer, ' - ADM Health Check - classic.html')),
                          target_generator_hash = r_health_check_hash,
                          renderer = function(filenameModelData,
                                              filenamePredictorData,
                                              title,
                                              subtitle,
                                              destinationfile)
                          {
                            # parameters dumped so they can be copy/pasted into the notebook directly
                            if (!quiet) cat("  modelfile:", paste0('"', file.path(report_utils_intermediates_folder, filenameModelData), '"'), fill=T)
                            if (!quiet) cat("  predictordatafile:", paste0('"', file.path(report_utils_intermediates_folder, filenamePredictorData), '"'), fill=T)

                            R.utils::mkdirs(dirname(destinationfile))
                            rmarkdown::render(
                              report_utils_healthcheck_notebook_R,
                              params = list(
                                "modelfile" = file.path(report_utils_intermediates_folder, filenameModelData),
                                "predictordatafile" = file.path(report_utils_intermediates_folder, filenamePredictorData),
                                "title" = title,
                                "subtitle" = subtitle
                              ),
                              output_dir = dirname(destinationfile),
                              output_file = basename(destinationfile),
                              quiet = quiet,
                              intermediates_dir = report_utils_intermediates_folder,
                              knit_root_dir = report_utils_intermediates_folder
                            )
                          },
                          quiet = quiet
  )
}

run_python_healthcheck <- function(customer, dm, quiet = T)
{
  python_health_check_hash <- digest::digest(readLines(report_utils_healthcheck_notebook_python), "sha256")

  report_utils_run_report(customer,
                          dm,
                          target_fullfilename = file.path(report_utils_results_folder,
                                                          paste0(customer, ' - ADM Health Check - new.html')),
                          target_generator_hash = python_health_check_hash,
                          renderer = function(filenameModelData,
                                              filenamePredictorData,
                                              title,
                                              subtitle,
                                              destinationfile)
                          {
                            # parameters dumped so they can be copy/pasted into the notebook directly
                            if (!quiet) cat("datafolder =", paste0('"', path.expand(report_utils_intermediates_folder), '"'), fill=T)
                            if (!quiet) cat("modelfilename =", paste0('"', filenameModelData, '"'), fill=T)
                            if (!quiet) cat("predictorfilename =", paste0('"', filenamePredictorData, '"'), fill=T)

                            # using output name results in loss of JS/CSS files so sticking to default name then copying.

                            if (quiet) {
                              quarto::quarto_render(
                                report_utils_healthcheck_notebook_python,
                                execute_params = list(
                                  "include_tables" = "True",
                                  "datafolder" = path.expand(report_utils_intermediates_folder),
                                  "modelfilename" = filenameModelData,
                                  "predictorfilename" = filenamePredictorData,
                                  "title" = title,
                                  "subtitle" = subtitle
                                ),
                                quiet = quiet,
                                pandoc_args = "--quiet"
                              )
                            } else {
                              quarto::quarto_render(
                                report_utils_healthcheck_notebook_python,
                                execute_params = list(
                                  "include_tables" = "True",
                                  "datafolder" = path.expand(report_utils_intermediates_folder),
                                  "modelfilename" = filenameModelData,
                                  "predictorfilename" = filenamePredictorData,
                                  "title" = title,
                                  "subtitle" = subtitle
                                ),
                                quiet = quiet
                              )
                            }

                            # TODO check status??
                            R.utils::mkdirs(dirname(destinationfile))
                            file.copy(paste0(sub('\\..[^\\.]*$', '', report_utils_healthcheck_notebook_python), ".html"),
                                      destinationfile,
                                      overwrite = TRUE,
                                      copy.date = TRUE
                            )
                          },
                          quiet = quiet
  )
}

# Choose some "interesting" model IDs for individual model reports
default_model_id_selection <- function(dm)
{
  ids <- unique(as.character(as.matrix(
    filterLatestSnapshotOnly(dm$modeldata)[, .(
      maxSuccessRate = ModelID[which.max(SuccessRate)],
      maxPerformance = ModelID[which.max(Performance)],
      maxPositives = ModelID[which.max(Positives)],
      maxResponseCount = ModelID[which.max(ResponseCount)]
    ), by = c("ConfigurationName", "Channel")] [, 3:6]
  )))

  ids[!is.na(ids)]
}

run_r_model_reports <-function(customer, dm,
                               modelids = unique(dm$modeldata$ModelID), # default_model_id_selection(dm)
                               max = length(modelids), quiet = T)
{
  cachedDMFilesFullName <- file.path(report_utils_intermediates_folder, report_utils_cached_dm_filenames(customer))
  r_model_report_hash <- digest::digest(readLines(report_utils_offlinemodelreport_notebook_R), "sha256")

  # TODO: predictor data could be null
  modelids <- head(intersect(unique(modelids), unique(dm$predictordata$ModelID)), max)

  for (n in seq_along(modelids)) {
    id <- modelids[n]
    modelName <- make.names(
      paste(sapply(unique(dm$modeldata[ModelID == id,
                                       intersect(c("ConfigurationName",
                                                   "Direction",
                                                   "Channel",
                                                   "Issue",
                                                   "Group",
                                                   "Name",
                                                   "Treatment"), names(dm$modeldata)), with=F]), as.character),
            collapse = "_"))

    cat("Model:", modelName, n, "of", length(modelids), fill = T)

    report_utils_run_report(customer,
                            dm,
                            target_fullfilename = file.path(report_utils_results_folder,
                                                            paste(customer, "Generated Model Reports", sep = " - "),
                                                            paste0(customer, " ", modelName, " - classic", ".html")),
                            target_generator_hash = r_model_report_hash,
                            renderer = function(filenameModelData,
                                                filenamePredictorData,
                                                title,
                                                subtitle,
                                                destinationfile) {
                              # parameters dumped so they can be copy/pasted into the notebook directly
                              if (!quiet) cat("  predictordatafile:", paste0('"', file.path(report_utils_intermediates_folder, filenamePredictorData), '"'), fill=T)
                              if (!quiet) cat("  modelid:", paste0('"', id, '"'), fill=T)

                              R.utils::mkdirs(dirname(destinationfile))
                              rmarkdown::render(
                                report_utils_offlinemodelreport_notebook_R,
                                params = list(
                                  "predictordatafile" = file.path(report_utils_intermediates_folder, filenamePredictorData),
                                  "modeldescription" = modelName,
                                  "modelid" = id
                                ),
                                output_dir = dirname(destinationfile),
                                output_file = basename(destinationfile),
                                quiet = quiet,
                                intermediates_dir = report_utils_intermediates_folder,
                                knit_root_dir = report_utils_intermediates_folder
                              )
                            },
                            quiet = quiet
    )
  }

  return(paste("Created", length(modelids), "R off-line model reports for", customer))
}

run_python_model_reports <-function(customer, dm,
                                    modelids = unique(dm$modeldata$ModelID), # default_model_id_selection(dm)
                                    max = length(modelids), quiet = T)
{
  cachedDMFilesFullName <- file.path(report_utils_intermediates_folder, report_utils_cached_dm_filenames(customer))
  python_model_report_hash <- digest::digest(readLines(report_utils_offlinemodelreport_notebook_python), "sha256")

  # TODO: predictor data could be null
  modelids <- head(intersect(unique(modelids), unique(dm$predictordata$ModelID)), max)

  for (n in seq_along(modelids)) {
    id <- modelids[n]
    modelName <- make.names(
      paste(sapply(unique(dm$modeldata[ModelID == id,
                                       intersect(c("ConfigurationName",
                                                   "Direction",
                                                   "Channel",
                                                   "Issue",
                                                   "Group",
                                                   "Name",
                                                   "Treatment"), names(dm$modeldata)), with=F]), as.character),
            collapse = "_"))

    cat("Model:", modelName, n, "of", length(modelids), fill = T)

    report_utils_run_report(customer,
                            dm,
                            target_fullfilename = file.path(report_utils_results_folder,
                                                            paste(customer, "Generated Model Reports", sep = " - "),
                                                            paste0(customer, " ", modelName, " - new", ".html")),
                            target_generator_hash = python_model_report_hash,
                            renderer = function(filenameModelData,
                                                filenamePredictorData,
                                                title,
                                                subtitle,
                                                destinationfile)
                            {
                              # parameters dumped so they can be copy/pasted into the notebook directly
                              if (!quiet) cat("datafolder =", paste0('"', path.expand(report_utils_intermediates_folder), '"'), fill=T)
                              if (!quiet) cat("modelfilename =", paste0('"', filenameModelData, '"'), fill=T)
                              if (!quiet) cat("predictorfilename =", paste0('"', filenamePredictorData, '"'), fill=T)
                              if (!quiet) cat("modelid =", paste0('"', id, '"'), fill=T)

                              # using output name results in loss of JS/CSS files so sticking to default name then copying.

                              if (quiet) {
                                quarto::quarto_render(
                                  report_utils_offlinemodelreport_notebook_python,
                                  execute_params = list(
                                    "datafolder" = path.expand(report_utils_intermediates_folder),
                                    "modelfilename" = filenameModelData,
                                    "predictorfilename" = filenamePredictorData,
                                    "modelid" = id,
                                    "title" = title,
                                    "subtitle" = subtitle
                                  ),
                                  quiet = quiet,
                                  pandoc_args = "--quiet"
                                )
                              } else {
                                quarto::quarto_render(
                                  report_utils_offlinemodelreport_notebook_python,
                                  execute_params = list(
                                    "datafolder" = path.expand(report_utils_intermediates_folder),
                                    "modelfilename" = filenameModelData,
                                    "predictorfilename" = filenamePredictorData,
                                    "modelid" = id,
                                    "title" = title,
                                    "subtitle" = subtitle
                                  ),
                                  quiet = quiet
                                )
                              }

                              # TODO check status??
                              R.utils::mkdirs(dirname(destinationfile))
                              file.copy(paste0(sub('\\..[^\\.]*$', '', report_utils_offlinemodelreport_notebook_python), ".html"),
                                        destinationfile,
                                        overwrite = TRUE,
                                        copy.date = TRUE
                              )

                              if (!quiet) cat("Created", destinationfile, fill=T)
                            },
                            quiet = quiet
    )
  }

  return(paste("Created", length(modelids), "python off-line model reports for", customer))
}

# read ADM data from cache or using given code block, write cached versions
# back alongside a .hash file representing the hash of the code block (not the
# data!)
read_adm_datamartdata <- function(customer, block, quiet = T)
{
  # Hash of the code block to actually read the data - R specific trick, not portable
  codeHash <- digest::digest(substitute(block), "sha256")

  # Target files with the datamart cached in parquet files
  cachedDMFilesFullName <- file.path(report_utils_intermediates_folder, report_utils_cached_dm_filenames(customer))

  # Only do full read if necessary
  if (!report_utils_is_target_current(cachedDMFilesFullName, codeHash, quiet = quiet)) {
    cat("Reading datamart data for", customer, "from sources", fill = TRUE)

    dm <- eval(block)

    report_utils_write_cached_files(dm, cachedDMFilesFullName[1], cachedDMFilesFullName[2])
    report_utils_write_hashfiles(cachedDMFilesFullName, codeHash)

  } else {
    has_predictordata <- file.size(cachedDMFilesFullName[2]) > 0
    if (has_predictordata) {
      dm <- ADMDatamart(modeldata = cachedDMFilesFullName[1],
                        predictordata = cachedDMFilesFullName[2])
    } else {
      dm <- ADMDatamart(modeldata = cachedDMFilesFullName[1],
                        predictordata = FALSE)
    }
  }

  return(dm)
}


