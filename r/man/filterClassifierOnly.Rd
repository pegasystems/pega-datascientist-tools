% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/adm.R
\name{filterClassifierOnly}
\alias{filterClassifierOnly}
\title{Subset the datamart predictor data to only the classifier binning.}
\usage{
filterClassifierOnly(dt, reverse = F)
}
\arguments{
\item{dt}{Predictor datamart table}

\item{reverse}{Boolean to flip the operation, so setting
this to TRUE results in all rows except the classifier.}
}
\value{
A \code{data.table}.
}
\description{
Subset the datamart predictor data to only the classifier binning.
}
\examples{
data(adm_datamart)
filterClassifierOnly(adm_datamart$preditordata)
}
