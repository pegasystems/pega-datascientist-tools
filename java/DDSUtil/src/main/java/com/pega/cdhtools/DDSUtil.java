package com.pega.cdhtools;

import com.datastax.driver.core.*;
import com.google.devtools.common.options.OptionsParser;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

// Reads the data from a DDS dataset and exports it in the same format as
// the manual dataset export would do, as a zipped multi-line JSON file.
public class DDSUtil {
    public static void main(String[] args) throws IOException {

        OptionsParser parser = OptionsParser.newOptionsParser(DDSUtilOptions.class);
        parser.parseAndExitUponError(args);
        DDSUtilOptions options = parser.getOptions(DDSUtilOptions.class);

        if (options.help | !(options.cmd.equals("TRUNCATE") | options.cmd.equals("EXPORT")) | options.dataset.equals("")) {
            printUsage(parser);
            System.exit(0);
        }

        System.out.println("Cassandra: ".concat(options.host).concat(" keyspace: ".concat(options.keyspace)));
        final Session session = getSession(options);
        final String tablename = getCassandraTableName(session, options.keyspace, options.dataset);

        if (tablename.equals("")) {
            System.out.println("DDS table for dataset ".concat(options.dataset).concat(" not found."));
            System.exit(-1);
        }

        switch (options.cmd) {
            case "TRUNCATE":
                truncateTable(session, options.keyspace, tablename);
                break;
            case "EXPORT":
                final String outFile = options.dir.concat("//").concat(options.dataset).concat(".zip"); // for testing!
                exportTable(session, options.keyspace, tablename, new File(outFile));
                break;
            default:
        }

        // close Session
        session.close();
        System.exit(0);
    }

    private static Session getSession(DDSUtilOptions options) {
        Cluster cluster = Cluster.builder().addContactPoint(options.host).withCredentials("dnode_ext", "dnode_ext").build();
        Session session = cluster.connect();
        return (session);
    }

    private static String getCassandraTableName(Session session, String keyspace, String ddsname) {
        final String tablename_prefix = ddsname.substring(0, Math.min(15, ddsname.length())).toLowerCase().concat("_");
        Metadata metadata = session.getCluster().getMetadata();

        String tablename = "";
        Collection<TableMetadata> tablesMetadata = metadata.getKeyspace(keyspace).getTables();
        for (TableMetadata tm : tablesMetadata) {
            if (tm.getName().startsWith(tablename_prefix)) {
                tablename = tm.getName();
                break;
            }
        }

        return (tablename);
    }

    private static void truncateTable(Session session, String keyspace, String tablename) {
        final String fullyQualifiedTableName = keyspace.concat(".").concat(tablename);

        System.out.println("Truncate ".concat(fullyQualifiedTableName));

        final String query = "truncate ".concat(fullyQualifiedTableName);
        System.out.println("query: ".concat(query));

        session.execute(query);
    }

    private static void exportTable(Session session, String keyspace, String tablename, File destination) throws java.io.IOException {
        final String fullyQualifiedTableName = keyspace.concat(".").concat(tablename);

        System.out.println("Export ".concat(fullyQualifiedTableName).concat(" to ").concat(destination.getCanonicalPath()));

        Metadata metadata = session.getCluster().getMetadata();
        TableMetadata tableMetadata = metadata.getKeyspace(keyspace).getTable(tablename);

        ArrayList<String> columnnames = new ArrayList<String>();
        Collection<ColumnMetadata> columnsMetadata = tableMetadata.getColumns();
        for (ColumnMetadata cm : columnsMetadata) {
            columnnames.add(cm.getName());
        }
        final String cols = String.join(",", columnnames);
        System.out.println("columns: ".concat(cols));

        FileOutputStream fout = new FileOutputStream(destination);
        ZipOutputStream zout = new ZipOutputStream(fout);
        ZipEntry ze = new ZipEntry("data.json"); // name of internal JSON file is hardcoded to be same as manual DS export
        zout.putNextEntry(ze);

        final String query = "select ".concat(cols).concat(" from ").concat(fullyQualifiedTableName);
        System.out.println("query: ".concat(query));

        int nrow = 0;
        ResultSet rows = session.execute(query);
        for (Row row : rows) {
            // Assert the structure is simple
            String level = row.getString("ds_");
            if (!level.equals("top_")) throw new RuntimeException("Nested structure detected - not supported");

            StringBuffer resultRowAsString = new StringBuffer("{");
            boolean isFirst = true;

            // Key/exposed columns first
            for (int i = 0; i < columnnames.size(); i++) {
                String colName = columnnames.get(i);
                if (!colName.endsWith("_")) {
                    Object value = row.getObject(i);
                    if (!isFirst) resultRowAsString.append(",");
                    resultRowAsString.append("\"").append(colName).append("\" : ");
                    if (value.getClass().getName().equals("java.lang.String")) { // quote String values
                        resultRowAsString.append("\"").append(value.toString()).append("\"");
                    } else {
                        resultRowAsString.append(value.toString());
                    }
                    isFirst = false;
                }
            }

            // Add the embedded JSON data as-is (skipping the opening bracket)
            final String jsonData = row.getString("data_");
            resultRowAsString.append(", ").append(jsonData.substring(1)).append("\n");
            zout.write(resultRowAsString.toString().getBytes());
            nrow++;
        }
        zout.closeEntry();
        zout.close();
        System.out.println("Done, written ".concat(String.valueOf(nrow)).concat(" records to ").concat(destination.getCanonicalPath()));
    }

    private static void printUsage(OptionsParser parser) {
        System.out.println("Usage: java -jar DDSUtil-1.0-SNAPSHOT.jar OPTIONS");
        System.out.println(parser.describeOptions(Collections.<String, String>emptyMap(),
                OptionsParser.HelpVerbosity.LONG));
    }
}


