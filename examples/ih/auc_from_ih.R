library(cdhtools)
library(data.table)
library(lubridate)
library(ggplot2)

# Read an export of IH data. IH is usually (very) large so in reality
# this may be filtered for the time range and models we're interested in
# and probably sampled as well.

# ih <- readDSExport("Data-pxStrategyResult_pxInteractionHistory", "~/Downloads")
ih <- readDSExport("Data-pxStrategyResult_pxInteractionHistory.zip", "../data")

# standardized camel casing of fields
applyUniformPegaFieldCasing(ih)

# convert date/time fields
ih[, OutcomeTime := fromPRPCDateTime(OutcomeTime)]

# Granularity is days. Calculate AUC from the predicted propensities and
# the actual outcomes stored in IH.

ih[, period := day(OutcomeTime)]
performance <- ih[Issue=="Sales", 
                  .(auc = auc_from_probs(Outcome=="Accept"|Outcome=="Accepted", Propensity),
                    n = .N,
                    date = first(OutcomeTime)), 
                  by=c("period", "Name", "Group", "Issue")]

# Plot

ggplot(performance, aes(date, auc, color=Name)) +
  geom_smooth(size=1,se=F) + geom_point() + theme_minimal() +
  scale_y_continuous(limits=c(0.5,NA))

