# Example preprocessing to create off-line model reports for
# CDH Sample, for CreditCards offers in Outbound channels

library(cdhtools)
library(data.table)

# Read the latest ADM Model export files from the Downloads folder
dm <- ADMDatamart("~/Downloads", 
                  filterModelData = function(mdls) {
                    return(mdls[ConfigurationName == "OmniAdaptiveModel" & Group == "CreditCards" & Direction == "Outbound"])
                  })

# Write back a CSV with only the model data of interest
write.csv(dm$modeldata, "~/Downloads/models.csv", row.names = F)

# Write back a CSV with only the predictor data for the models of interest
write.csv(dm$predictordata, "~/Downloads/predictors.csv", row.names = F)
