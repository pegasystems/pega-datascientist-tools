# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pdstools[app]>=4.0.3",
# ]
# ///

import argparse
import difflib
import logging
import os
import sys
from importlib import resources

from pdstools import __version__

# App configuration with display names and paths
APPS = {
    "health_check": {
        "display_name": "Adaptive Model Health Check",
        "path": "pdstools.app.health_check",
    },
    "decision_analyzer": {
        "display_name": "Decision Analysis",
        "path": "pdstools.app.decision_analyzer",
    },
    "impact_analyzer": {
        "display_name": "Impact Analyzer",
        "path": "pdstools.app.impact_analyzer",
    },
}

# Aliases for app names
ALIASES = {
    "hc": "health_check",
    "da": "decision_analyzer",
    "ia": "impact_analyzer",
}


def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line utility to run pdstools apps.",
    )

    parser.add_argument("--version", action="version", version=f"pdstools {__version__}")

    # Create help text with display names and aliases
    app_choices = list(APPS.keys()) + list(ALIASES.keys())
    help_parts = []
    for key in APPS.keys():
        # Find aliases for this app
        app_aliases = [alias for alias, full_name in ALIASES.items() if full_name == key]
        alias_text = f" (alias: {', '.join(app_aliases)})" if app_aliases else ""
        help_parts.append(f'"{key}"{alias_text}')

    help_text = "The app to run: " + " | ".join(help_parts)

    parser.add_argument(
        "app",
        choices=app_choices,
        help=help_text,
        nargs="?",  # This makes the 'app' argument optional
        default=None,  # Explicitly set default to None
    )
    parser.add_argument(
        "--data-path",
        dest="data_path",
        default=None,
        help=(
            "Path to a data file or directory to load on startup. "
            "Supports parquet, csv, json, arrow, zip, and partitioned folders. "
            "Exposed to the app as the PDSTOOLS_DATA_PATH env var."
        ),
    )
    parser.add_argument(
        "--sample",
        dest="sample",
        default=None,
        help=(
            "Pre-ingestion interaction sampling for large datasets. "
            "Specify an absolute count (e.g. '100000', '100k', '1M') or a percentage "
            "(e.g. '10%%'). All rows for each sampled interaction are kept. "
            "Exposed to the app as the PDSTOOLS_SAMPLE_LIMIT env var."
        ),
    )
    parser.add_argument(
        "--temp-dir",
        dest="temp_dir",
        default=None,
        help=(
            "Directory for temporary files such as the sampled data parquet. "
            "Defaults to the current working directory. "
            "Exposed to the app as the PDSTOOLS_TEMP_DIR env var."
        ),
    )
    return parser


def check_for_typos(unknown_args, known_args):
    """Check if unknown arguments might be typos of known pdstools arguments.

    Args:
        unknown_args: List of unknown arguments from parse_known_args
        known_args: List of known pdstools argument names (with --)

    Returns:
        List of (typo, suggestion, similarity) tuples for likely typos
    """
    likely_typos = []

    # Extract arguments that start with -- (not streamlit args or values)
    unknown_flags = [arg for arg in unknown_args if arg.startswith("--")]

    for unknown in unknown_flags:
        # Find the most similar known argument
        # Use cutoff=0.6 to only suggest reasonably similar names
        matches = difflib.get_close_matches(unknown, known_args, n=1, cutoff=0.6)

        if matches:
            # Calculate similarity ratio for reporting
            similarity = difflib.SequenceMatcher(None, unknown, matches[0]).ratio()
            likely_typos.append((unknown, matches[0], similarity))

    return likely_typos


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        del sys.argv[1]
    parser = create_parser()
    args, unknown = parser.parse_known_args()

    # Resolve alias to full app name
    if args.app and args.app in ALIASES:
        args.app = ALIASES[args.app]

    # Check for likely typos in pdstools arguments
    known_pdstools_args = ["--version", "--data-path", "--sample", "--temp-dir"]
    typos = check_for_typos(unknown, known_pdstools_args)

    if typos:
        print("\n⚠️  Warning: Possible typo(s) in pdstools arguments:\n", file=sys.stderr)
        for typo, suggestion, similarity in typos:
            print(f"  '{typo}' → Did you mean '{suggestion}'?", file=sys.stderr)

        print("\nAvailable pdstools arguments:", file=sys.stderr)
        for arg in known_pdstools_args:
            print(f"  {arg}", file=sys.stderr)

        print("\nNote: Unknown arguments are passed to Streamlit, which may not recognize them.", file=sys.stderr)
        print("Use --help to see all available options.\n", file=sys.stderr)

    run(args, unknown)


def run(args, unknown):
    # Configure logging based on environment variable
    log_level = os.environ.get("PDSTOOLS_LOG_LEVEL", "WARNING").upper()
    numeric_level = getattr(logging, log_level, logging.WARNING)
    logging.basicConfig(
        level=numeric_level,
        format="%(levelname)s [%(name)s]: %(message)s",
        force=True,  # Override any existing config
    )

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            "Error: streamlit is not installed. Try installing the optionall dependency group 'app'.\n"
            "If you are using uvx, try running uvx 'pdstools[app]' instead.",
        )
        sys.exit(1)

    # If no app is specified, prompt the user to choose
    if args.app is None:
        app_list = list(APPS.keys())
        print("Available pdstools apps:", flush=True)
        for i, app_key in enumerate(app_list, 1):
            display_name = APPS[app_key]["display_name"]
            print(f"  {i}. {display_name}", flush=True)

        while True:
            try:
                choice = input(
                    f"\nPlease select an app to run (1-{len(app_list)}): ",
                ).strip()

                # Check if it's a number
                if choice.isdigit() and 1 <= int(choice) <= len(app_list):
                    args.app = app_list[int(choice) - 1]
                    break

                # Check if it's an internal app name
                if choice.lower() in APPS:
                    args.app = choice.lower()
                    break

                # Check if it's an alias
                if choice.lower() in ALIASES:
                    args.app = ALIASES[choice.lower()]
                    break

                # Check if it's a display name (case insensitive)
                found = False
                for app_key, app_info in APPS.items():
                    if choice.lower() == app_info["display_name"].lower():
                        args.app = app_key
                        found = True
                        break
                if found:
                    break

                # If we get here, invalid input
                valid_options = []
                valid_options.extend([str(i) for i in range(1, len(app_list) + 1)])
                valid_options.extend(APPS.keys())
                valid_options.extend(ALIASES.keys())
                valid_options.extend(
                    [app_info["display_name"] for app_info in APPS.values()],
                )
                print(
                    f"Invalid choice. Please enter: {', '.join(valid_options[:4])}...",
                )

            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                sys.exit(0)
            except Exception:
                print("Invalid input. Please try again.")

    # Propagate CLI flags to the Streamlit process via env vars
    if args.data_path:
        os.environ["PDSTOOLS_DATA_PATH"] = args.data_path
    if args.sample:
        os.environ["PDSTOOLS_SAMPLE_LIMIT"] = args.sample
        # Check if polars 64-bit runtime is installed; hint if not
        try:
            from importlib.metadata import distribution

            distribution("polars-rt64")
            has_rt64 = True
        except Exception:
            has_rt64 = False
        if not has_rt64:
            print(
                "\n💡 Tip: For very large datasets (>2 billion elements), "
                "install the polars 64-bit runtime:\n"
                "   uv pip install 'polars[rt64]'\n",
                file=sys.stderr,
            )
    if args.temp_dir:
        os.environ["PDSTOOLS_TEMP_DIR"] = args.temp_dir

    display_name = APPS[args.app]["display_name"]
    print(f"Running {display_name} app...")

    app_path = APPS[args.app]["path"]
    with resources.path(app_path, "Home.py") as filepath:
        filename = str(filepath)

    if args.app == "decision_analyzer":
        sys.argv = [
            "streamlit",
            "run",
            filename,
            "--server.enableXsrfProtection",
            "false",
        ]
    else:  # health_check
        sys.argv = ["streamlit", "run", filename]

    if unknown:
        sys.argv.extend(unknown)
    if "--server.maxUploadSize" not in sys.argv:
        sys.argv.extend(["--server.maxUploadSize", "2000"])
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
