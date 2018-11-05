# Creates RISK PMML model

library(lubridate)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(pmml)

datadir <- "data"
modeldir <- "models"

# The base date should be the same as the "DemoDate" set in the global
# control parameters of the DMSample application.
BaseDateString <- 'Mon Nov 5 16:00:00 CET 2018' # copy/paste from PRPC
BaseDateAsDate <- as.POSIXct(as.Date(BaseDateString, format="%a %b %d %H:%M:%S CET %Y", tz="CET"))

testCustomers <- data.frame(
  ID = c('CE-1', 
         'CE-10', 
         'CE-2',
         'CE-3',
         'CE-4',
         'CE-5',
         'CE-6',
         'CE-7',
         'CE-8',
         'CE-9'),
  Age = c(23, 
          35, 
          45,
          33,
          55,
          18,
          45,
          27,
          39,
          30),
  Gender = c('f', 
             'm', 
             'f',
             'm',
             'm',
             'm',
             'f',
             'f',
             'm',
             'f'),
  ContractEndDate = c("20160803T230000.000 GMT",
                      "20170901T220000.000 GMT",
                      "20160127T230000.000 GMT",
                      "20160514T050000.000 GMT",
                      "20160113T230000.000 GMT",
                      "20170312T230000.000 GMT",
                      "20160201T230000.000 GMT",
                      "20160616T050000.000 GMT",
                      "20170403T220000.000 GMT",
                      "20160913T220000.000 GMT"),
  DataUsage = c(500,
                1500,
                300,
                1000,
                300,
                100,
                200,
                300,
                1000,
                300))

rnormclip <- function(n, mean = 0, sd = 1, cliplo, cliphi) {
  r <- rnorm(n, mean, sd)
  return (ifelse(r < cliplo | r > cliphi, runif(n, cliplo, cliphi), r))
}

ncust <- 20000
custAge <- round(rnormclip(ncust, 40, 20, 15, 80))
# custGender <- sample(c('F','M','f','m','Male','Female'),ncust,replace=T) # purposefully not clean but using multiple representations for the same
custGender <- sample(c('f','m'),ncust,replace=T) # purposefully not clean but using multiple representations for the same
customers <- data.frame(ID=paste("Cust_",1:ncust,sep=""),Age=custAge,Gender=custGender)

# Contract end months and Usage data represent product usage aggregates
# Eventually these should be proper datasets of their own and aggregated
# in DSM. For the purpose of modelling, we can only use pre-aggregated values
# currently.
endDateLo <- BaseDateAsDate + months(5) + days(1) # ymd("2016-06-01"), high was ymd("2017-12-01")
endDate <- endDateLo + days(round(runif(ncust, 0,  548)))

# Generate in Pega Date fmt "20160616T050000.000 GMT"
customers$ContractEndDate <- format(endDate, format="%Y%m%dT%H%M%S.000 GMT", tz="GMT")

customers$DataUsage <- round(runif(ncust, 300, 50000)) # some indication of usage

# outcomes:
# Risk : Reject / Defer
# Churn : Yes/No, will be mapped to High / Medium / Low
for (c in 1:ncust) {
  pChurn <- runif(1, 0.1, 0.2)
  pRisk <- rnormclip(1, 0.5, 0.5, 0, 1)

  pChurn <- pChurn * 
    (1 - (as.double(endDate[c] - min(endDate)) / as.double(max(endDate) - min(endDate)))) * 
    (1 + 0.8* (customers$DataUsage[c] / max(customers$DataUsage))) *
    (1 + 0.7* (1-((customers$Age[c] - 25)^2) / max((customers$Age - 25)^2))) *
    ifelse(customers$Gender[c] %in% c('f','F','Female'), 0.7, 1.3)
  pChurn <- min(pChurn, 1)
  
  pRisk <- pRisk * 
    (1 + 0.2* (customers$DataUsage[c] / max(customers$DataUsage))) *
    (1 + 0.1* (1-((customers$Age[c] - 40)^2) / max((customers$Age - 40)^2))) *
    ifelse(customers$Gender[c] %in% c('f','F','Female'), 1.2, 0.8)
  pRisk <- min(pRisk, 1)
  
  customers$pRisk[c] <- pRisk
  customers$pChurn[c] <- pChurn
  
  customers$Risk[c] <- sample(c(T,F), prob=c(pRisk, 1-pRisk), replace=T)
  customers$Churn[c] <- sample(c(T,F), prob=c(pChurn, 1-pChurn), replace=T)
}

# Plots to review distributions

# customers$set <- "Train"
# 
# testCustomers$pRisk <- NA
# testCustomers$pChurn <- NA
# testCustomers$Risk <- NA
# testCustomers$Churn <- NA
# testCustomers$set <- "Test"
# 
# #customers <- rbind(customers, testCustomers)

customers$endDate <- as.POSIXct(as.Date(customers$ContractEndDate, format="%Y%m%dT%H%M%S.000 GMT"), tz="GMT")
customers$ageBinned <- cut(customers$Age,breaks=seq(10,100,by=5))
customers$dataBinned <- cut(customers$DataUsage,20)

# Reject vs end date / age / data
p<- ggplot(group_by(customers, endDate) %>% summarise(pReject = sum(Risk) / n()), 
       aes(endDate, pReject)) +
  geom_line() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Risk by date")
print(p)
        
p <- ggplot(data=group_by(customers, ageBinned) %>% summarise(pReject = sum(Risk) / n()),
             aes(x=ageBinned, y=pReject))+geom_point()+
        theme(axis.text.x = element_text(angle = 45, hjust = 1))+
        geom_smooth(method = "loess", se=TRUE, color="blue", aes(group=1))+
  ggtitle("Risk by Age")
print(p)

p<- ggplot(group_by(customers, dataBinned) %>% summarise(pReject = sum(Risk) / n()), 
           aes(dataBinned, pReject)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Risk by data usage")
print(p)

p<- ggplot(group_by(customers,  Gender, Age) %>% summarise(pReject = sum(Risk) / n()), 
           aes(Age, fill=Gender, pReject)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Risk by Gender and Age")
print(p)

# similar for churn
p <- ggplot(data=group_by(customers, ageBinned) %>% summarise(pChurn = sum(Churn) / n()),
            aes(x=ageBinned, y=pChurn))+geom_point()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  geom_smooth(method = "loess", se=TRUE, color="blue", aes(group=1))+
  ggtitle("Churn by date")
print(p)

p<- ggplot(group_by(customers, dataBinned) %>% summarise(pChurn = sum(Churn) / n()), 
           aes(dataBinned, pChurn)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Churn by data usage")
print(p)

p<- ggplot(group_by(customers, dataBinned) %>% summarise(pChurn = sum(Churn) / n()), 
           aes(dataBinned, pChurn)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Churn by data usage")
print(p)

p<- ggplot(group_by(customers,  Gender, Age) %>% summarise(pChurn = sum(Churn) / n()), 
           aes(Age, fill=Gender, pChurn)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Churn by Gender and Age")
print(p)

customers$Risk <- ifelse(customers$Risk, "Reject", "Accept")

# Set the base date for modelling. Handy if this matches up with the demo date in the shipped application.
# In DMSample we somehow got to this weird format "Sun Oct 04 16:00:00 CEST 2015"
customers$BaseDate <- BaseDateString
customers <- select(customers, -ageBinned, -dataBinned, -endDate, -pRisk)

print(head(customers,10))
write.csv(customers, paste(datadir, "dmsample_customers.csv", sep="/"), row.names=F)

print(summary(customers))

# Now, create a simple forest model for Risk
customers$Risk <- as.factor(customers$Risk)
set.seed(1234)
riskModel = randomForest(Risk ~ ., 
                         # NB 'end date' needs pre-processing which we can't easily do in PMML
                         # NB PMML support is problematic for larger forests: BUG-209053
                         data = select(customers, Gender, DataUsage, Age, Risk), ntree = 20)
print(riskModel)
print(summary(customers$Risk))

pmmlModel <- pmml(riskModel)
saveXML(pmmlModel, paste(modeldir, "riskModel.pmml", sep="/"))

results <- testCustomers
results$RiskPrediction <- predict(riskModel, newdata=testCustomers) #, type="prob")
print(results)


