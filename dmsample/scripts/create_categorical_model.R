################################################################
# Create a PMML model with a categorical outcome for DMSample
# Ivar Siccama
################################################################
library(data.table)

datadir <- "data"
modeldir <- "models"
input <- paste(datadir,"data_dms_customer.csv",sep="/")
train <- fread(input, header = T, sep = ",", stringsAsFactors=T)

scoreCreditLimit     <- rep(0, nrow(train))
scoreComplaint       <- rep(0, nrow(train))
scoreCustomerService <- rep(0, nrow(train))
scoreOther           <- rep(0, nrow(train))

# Simulate a behavior score for the call context Credit Limit
sel <- train$credithistory  == "Past arrears"
scoreCreditLimit[sel] <- scoreCreditLimit[sel] + 2
sel <- train$paymenthistory  == "Same month"
scoreCreditLimit[sel] <- scoreCreditLimit[sel] + 1
sel <- train$income < 30000
scoreCreditLimit[sel] <- scoreCreditLimit[sel] + 1
sel <- train$income < 40000
scoreCreditLimit[sel] <- scoreCreditLimit[sel] + 1
sel <- train$income < 60000
scoreCreditLimit[sel] <- scoreCreditLimit[sel] + 1

# Simulate a behavior score for the call context Complaint
sel <- train$frequentcaller == "Yes"
scoreComplaint[sel] <- scoreComplaint[sel] + 1
sel <- train$startingmember == "Yes"
scoreComplaint[sel] <- scoreComplaint[sel] + 2
sel <- train$age < 25
scoreComplaint[sel] <- scoreComplaint[sel] + 2
sel <- train$age < 40
scoreComplaint[sel] <- scoreComplaint[sel] + 1

# Simulate a behavior score for the call context Customer Service
sel <- train$startingmember == "Yes"
scoreCustomerService[sel] <- scoreCustomerService[sel] + 2
sel <- train$frequentcaller == "Yes"
scoreCustomerService[sel] <- scoreCustomerService[sel] + 1
sel <- train$creditstatus == "Due payment"
scoreCustomerService[sel] <- scoreCustomerService[sel] + 1
sel <- train$age > 40
scoreCustomerService[sel] <- scoreCustomerService[sel] + 1
sel <- train$credithistory == "Past arrears"
scoreCustomerService[sel] <- scoreCustomerService[sel] + 0.5

###
# Function to simulate categorical outcomes for Call Context
###
setCallContext <- function(df) {
  cat(df[1], df[2], df[3],"\n")
  tot <- df[1] + df[2] + df[3]
  cat1 <- 0
  cat2 <- 0
  cat3 <- 0
  if (tot > 0) {
    cat1 <- df[1] / tot
    cat2 <- df[2] / tot
    cat3 <- df[3] / tot
  }
  category <- "Other"
  cat(cat1, cat2, cat3, "\n")
  if (cat1 > cat2 && cat1 > cat3) {
    if (runif(1) > 0.10) {category <- "Credit Limit"}
    else {category <- "Complaint"}
  }
  if (cat2 > cat1 && cat2 > cat3) {
    if (runif(1) > 0.05) {category <- "Complaint"}
    else {category <- "Credit Limit"}
  }
  if (cat3 > cat1 && cat3 > cat2) {
    if(runif(1) > 0.02) {category <- "Customer Service"}
    else {category <- "Other"}
  }
  cat(category, "\n")
  category
}
scores <- data.frame(scoreCreditLimit, scoreComplaint, scoreCustomerService)
callcontext <- apply(scores, 1, setCallContext)

####
# Create the model
####
library("r2pmml")
library("randomForest")
library("gbm")
library("caret")


preds <- train[, c("age","credithistory","creditstatus","paymenthistory",
                   "frequentcaller","gender","income","startingmember")]
trainIndex <- createDataPartition(callcontext, p=0.80, list=FALSE)
xDev <- as.data.frame(preds)[ trainIndex,] # c(1:500,1934)]
xVal <- as.data.frame(preds)[-trainIndex,] # c(1:500,1934)]
yDev <- callcontext[trainIndex]
yVal <- callcontext[-trainIndex]

gbm = gbm.fit(x = xDev, y = yDev, distribution="multinomial",
              interaction.depth = 3, shrinkage = 0.1, n.trees = 100,
              response.name = "CallContext")
print(gbm)
valPredictions = predict(gbm,newdata = xVal, n.trees=100, type="response")
categ <- apply(valPredictions, 1, which.max)
print(data.frame(valPredictions, categ, yVal))
table(categ, yVal)

# Export the model to PMML
output <- paste(modeldir,"PredictCallContext.pmml",sep="/")
r2pmml(gbm, output)

##############
# As a separate task, generate an additional 'value' column, that can be used to
# build a model that predicts customer value
##############
val <- rep(0, nrow(train))

sel <- train$startingmember == "No"
val[sel] <- val[sel] + 3

sel <- train$frequentcaller == "Yes"
val[sel] <- val[sel] + 5

sel <- train$creditstatus == "Due payment"
val[sel] <- val[sel] - 1

sel <- train$age > 30
val[sel] <- val[sel] + 2
sel <- train$age > 55
val[sel] <- val[sel] + 1

sel <- train$credithistory  == "Paid on time"
val[sel] <- val[sel] + 2

sel <- train$creditstatus  == "Clear"
val[sel] <- val[sel] + 2
sel <- train$creditstatus  == "Due payment"
val[sel] <- val[sel] - 1

sel <- train$paymenthistory  == "Same day"
val[sel] <- val[sel] + 1
sel <- train$paymenthistory  == "Same week"
val[sel] <- val[sel] + 2

sel <- train$gender  == "f"
val[sel] <- val[sel] + 0.5

sel <- train$income > 30000
val[sel] <- val[sel] + 1
sel <- train$income > 40000
val[sel] <- val[sel] + 2
sel <- train$income > 60000
val[sel] <- val[sel] + 3
val <- val * 100

yDev <- val[trainIndex] 
yVal <- val[-trainIndex]
gbmValue = gbm.fit(x = xDev, y = yDev, distribution= 'gaussian',
              interaction.depth = 3, shrinkage = 0.1, n.trees = 200,
              response.name = "Value")
predictedVals = predict(gbmValue, newdata = xVal, n.trees=100, type="response")
print(data.frame(predictedVals, yVal, yVal-predictedVals))
rmse <- sqrt(mean((yVal-predictedVals)^2))
cat("RMSE=", rmse, "\n")
# Export to PMML
output <- paste(modeldir,"PredictCustomerValue.pmml",sep="/")
r2pmml(gbmValue, output)

# Export the training data to csv so that we can also build a spectrum model in PAD
preds$value <- val
csv_file <- paste(datadir, "customer_data_with_value.csv", sep="/")
write.table(preds, file=csv_file, row.names=F, col.names=T, sep=",")
