# Export ADM Datamart models to PMML directly from DB

library(XML) # not sure why this is necessary...
library(RJDBC)
library(cdhtools)

drv <- JDBC("org.postgresql.Driver", "~/Documents/tools/jdbc-drivers/postgresql-9.2-1002.jdbc4.jar")
pg_host <- "localhost:5432"
pg_db <- "pega731"
pg_user <- "pega"
pg_pwd <- "pega"

conn <- dbConnect(drv, paste("jdbc:postgresql://", pg_host,  "/", pg_db, sep=""), pg_user, pg_pwd)

adm2pmml(dbConn=conn, forceUseDM = T, verbose=T, ruleNameFilter = "BannerModel")

dbDisconnect(conn)
