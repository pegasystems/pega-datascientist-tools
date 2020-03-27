package com.pega.cdhtools;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

/**
 * Command-line options definition for example server.
 */
public class DDSUtilOptions extends OptionsBase {

    @Option(
            name = "help",
            abbrev = 'h',
            help = "Prints usage info.",
            category = "startup",
            defaultValue = "false"
    )
    public boolean help;

    //public enum ddsCommands {TRUNCATE, DUMP};

    @Option(
            name = "cmd",
            abbrev = 'c',
            help = "The operation to perform: TRUNCATE or EXPORT.",
            category = "startup",
            defaultValue = ""
    )
    public String cmd;

    @Option(
            name = "dataset",
            abbrev = 's',
            help = "Pega name of the dataset.",
            category = "startup",
            defaultValue = ""
    )
    public String dataset;

    @Option(
            name = "keyspace",
            abbrev = 'k',
            help = "Cassandra dataset keyspace.",
            category = "startup",
            defaultValue = "data"
    )
    public String keyspace;

    @Option(
            name = "host",
            abbrev = 'o',
            help = "IP name/address of the Cassandra instance.",
            category = "startup",
            defaultValue = "localhost"
    )
    public String host;

    /**
    @Option(
            name = "port",
            abbrev = 'p',
            help = "The port of the Cassandra instance.",
            defaultValue = "8080"
    )
    public int port;
    **/

    @Option(
            name = "dir",
            abbrev = 'd',
            help = "Name of directory to write the export to.",
            category = "startup",
            allowMultiple = false,
            defaultValue = "."
    )
    public String dir;

}