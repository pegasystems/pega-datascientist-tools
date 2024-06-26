% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adm.R
\name{ADMDatamart}
\alias{ADMDatamart}
\title{Generic method to read ADM Datamart data.}
\usage{
ADMDatamart(
  modeldata = NULL,
  predictordata = NULL,
  folder = ".",
  cleanupHookModelData = identity,
  cleanupHookPredictorData = identity,
  keepSerializedModelData = F,
  filterModelData = identity,
  filterPredictorData = identity,
  predictorCategorization = defaultPredictorCategorization,
  tmpFolder = tempdir(check = T),
  verbose = F
)
}
\arguments{
\item{modeldata}{Location, reference or the actual model table from the
ADM datamart. If not given defaults to the name of the dataset export file
for the datamart model table. To not use it at all, set to FALSE.}

\item{predictordata}{Location, reference or the actual predictor binning
table from the ADM datamart. If not given defaults to the name of the dataset export file
for the datamart predictor binning table. To not use at all, set to FALSE.}

\item{folder}{Optional path for the folder in which to look for the model
and predictor data. If first two arguments are not given will try to
interpret the modeldata argument as the folder name and use the default
dataset export file names.}

\item{cleanupHookModelData}{Optional cleanup function to call directly after the
raw model data has been read from the source file. This is especially helpful
to clean up data that has been custom exported from a database into an
Excel or CSV file.}

\item{cleanupHookPredictorData}{Optional cleanup function to call directly after the
raw predictor data has been read from the source file. This is especially helpful
to clean up data that has been custom exported from a database into an
Excel or CSV file.}

\item{keepSerializedModelData}{By default the serialized model data is left
out from the model data as this typically is large and only needed for
specific use cases.}

\item{filterModelData}{Post processing filter on the models, defaults to no
filtering but a custom function can be provided to drop e.g. certain
channels or issues or perform other clean up activities.}

\item{filterPredictorData}{Post processing filter on the predictor data. Defaults
to no filtering. Filtering on model ID's present in the model data will be
performed regardless. Potentially useful is to supply \code{filterLatestSnapshotOnly},
so to keep only the predictor data of the last snapshot - if multiple are
present.}

\item{predictorCategorization}{When passed in, a function that determines
the category of a predictor. This is then used to set an extra field in
the predictor data: "PredictorCategory" that holds the predictor category.
This is useful to globally set the predictor categories instead of having
to pass it to every individual plot function. The function does not need
to be vectorized. It is just applied to the levels of a factor and should
take just a character as an argument. Defaults to a function that returns the text
before the first dot.}

\item{tmpFolder}{Optional folder to store unzipped data (defaults to a temp folder)}

\item{verbose}{Flag for verbose logging, defaults to FALSE.}
}
\value{
A \code{list}) with two elements: "modeldata" contains a clean
\code{data.table} with the ADM model data, and "predictordata" contains a
clean \code{data.table} with the ADM predictor snapshots.
}
\description{
Method is very flexible in the arguments. It can take a \code{data.table}
for the model data and the predictor data, but the arguments can also be
pointing to files in CSV, zipped CSV, (LD)JSON, parquet or dataset export formats.
}
\examples{
\dontrun{
  datamart <- ADMDatamart("~/Downloads")

  datamart <- ADMDatamart("models.csv", "predictors.csv", folder="adm")
}
}
