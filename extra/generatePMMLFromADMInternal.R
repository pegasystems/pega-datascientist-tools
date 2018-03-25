# Read ADM factory JSON and turn into PMML

library(XML) # not sure why this is necessary...
library(jsonlite) # not sure why this is necessary...
library(RJDBC)
library(cdhtools)

# Change the driver path, host, db name, username and password to run on different environments.

drv <- JDBC("org.postgresql.Driver", "~/Documents/tools/jdbc-drivers/postgresql-9.2-1002.jdbc4.jar")

pg_user <- "pega"
pg_pwd <- "pega"

conn <- dbConnect(drv, "jdbc:postgresql://localhost:5432/pega731", pg_user, pg_pwd)

adm2pmml(dbConn=conn, verbose=T)

dbDisconnect(conn)


