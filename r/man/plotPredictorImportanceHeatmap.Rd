% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plots.R
\name{plotPredictorImportanceHeatmap}
\alias{plotPredictorImportanceHeatmap}
\title{Performance of predictors per proposition}
\usage{
plotPredictorImportanceHeatmap(
  datamart,
  aggregation = intersect(c("Issue", "Group", "Name", "Treatment"), names(modeldata)),
  facets = "ConfigurationName",
  limit = .Machine$integer.max,
  maxNameLength = .Machine$integer.max,
  filter = identity
)
}
\arguments{
\item{datamart}{ADM Datamart data.}

\item{aggregation}{Vector of field names to aggregate by. Defaults to
"Issue","Group","Name","Treatment" (if present).}

\item{facets}{Vector of additional fields for faceting. Defaults to
"ConfigurationName".}

\item{limit}{Optional limit of number of predictors. Useful if there are
many predictors to keep the plot readable.}

\item{maxNameLength}{Max length of predictor names. Above this length
they will be abbreviated. NOTE: currently not supported anymore.}

\item{filter}{Optional filter to apply on the model data}
}
\value{
A \code{ggplot} object that can either be printed directly, or to
which additional decoration (e.g. coloring or labels) can be added first.
}
\description{
Creates a visual grid of predictor names vs propositions (configurable) with
colors indicating the strength of the predictor for that proposition.
}
\examples{
\dontrun{
    plt <- plotPredictorImportanceHeatmap(modelPredictorBins, mdls, limit=50) +
    theme(axis.text.y = element_text(size=8),
    axis.text.x = element_text(size=8, angle = 45, hjust = 1),
    strip.text = element_text(size=8))
    print(plt)
}
}
