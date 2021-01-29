# Read ADM factory JSON and turn into PMML

# just for rJava on OSX:
if (Sys.info()['sysname'] == 'Darwin') {
  dyn.load(paste0(system2('/usr/libexec/java_home', stdout = TRUE), '/jre/lib/server/libjvm.dylib'))
}

library(XML)
library(jsonlite)
library(RJDBC)
library(cdhtools)

# Change the driver path, host, db name, username and password to run on different environments.

drv <- JDBC("org.postgresql.Driver", "~/Documents/tools/jdbc-drivers/postgresql-9.2-1002.jdbc4.jar")

pg_user <- "<USERNAME>"
pg_pwd <- "<PASSWORD>"

conn <- dbConnect(drv, "jdbc:postgresql://localhost:5432/pega731", pg_user, pg_pwd)

adm2pmml(dbConn=conn, verbose=T)

dbDisconnect(conn)


