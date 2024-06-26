% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cdh_utils.R
\name{auc2GINI}
\alias{auc2GINI}
\title{Convert AUC performance metric to GINI.}
\usage{
auc2GINI(auc)
}
\arguments{
\item{auc}{The ROC AUC (number between 0.5 and 1)}
}
\value{
GINI metric, a number between 0 and 1
}
\description{
Convert AUC performance metric to GINI.
}
\examples{
auc2GINI(0.8232)
}
