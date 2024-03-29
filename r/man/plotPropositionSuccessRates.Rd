% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plots.R
\name{plotPropositionSuccessRates}
\alias{plotPropositionSuccessRates}
\title{Plot ADM Proposition Success Rates}
\usage{
plotPropositionSuccessRates(
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

\item{limit}{Optional limit of number of propositions.}

\item{maxNameLength}{Max length of predictor names. Above this length
they will be abbreviated.}

\item{filter}{Optional filter to apply on the model data}
}
\value{
A \code{ggplot} object that can either be printed directly, or to
which additional decoration (e.g. coloring or labels) can be added first.
}
\description{
Creates a bar char of proposition success rates. Most useful if done per
channel or other field.
}
\examples{
\dontrun{
plotPropositionSuccessRates(mdls[Positives > 10], limit=20, facets="Channel") +
scale_fill_continuous_divergingx()
}
}
