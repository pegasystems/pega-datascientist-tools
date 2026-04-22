# /// script
# requires-python = ">=3.10"
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

# Subcommands recognised by the pre-parser. Anything else (including a
# bare app name) is routed to ``run`` for backwards compatibility.
_SUBCOMMANDS = {"run", "doctor", "list"}

# App configuration with display names and paths.
# ``launcher`` is listed first so it's the default selection in the
# interactive picker — it hosts all three tools in one process.
APPS = {
    "launcher": {
        "display_name": "All tools (launcher)",
        "path": "pdstools.app.launcher",
    },
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
    "all": "launcher",
    "hc": "health_check",
    "da": "decision_analyzer",
    "ia": "impact_analyzer",
}


def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line utility to run pdstools apps.",
    )

    parser.add_argument("--version", action="version", version=f"pdstools {__version__}")
    parser.add_argument(
        "--list",
        dest="list_apps",
        action="store_true",
        default=False,
        help="List available apps (one per line, tab-separated key/name/aliases) and exit.",
    )

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
            "Exposed to the app as the PDSTOOLS_SAMPLE_LIMIT env var. "
            "To sample programmatically without the app, see "
            "pdstools.decision_analyzer.utils.sample_interactions() and "
            "pdstools.decision_analyzer.utils.prepare_and_save()."
        ),
    )
    parser.add_argument(
        "--filter",
        dest="filter",
        action="append",
        default=None,
        help=(
            "Pre-ingestion row filter for extracting specific data from large files. "
            "Syntax options: "
            "'Column=value1,value2,...' (categorical, exact match), "
            "'Column>=N' / 'Column<=N' / 'Column>N' / 'Column<N' (numeric), "
            "'Column=YYYY-MM-DD..YYYY-MM-DD' (date range, inclusive). "
            "Column names use display names (e.g. 'Channel', 'Decision Time', "
            "'ModelPositives'). Multiple --filter flags are ANDed together. "
            "Can be combined with --sample (filter is applied first)."
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
    parser.add_argument(
        "--full-embed",
        dest="full_embed",
        action="store_true",
        default=False,
        help=(
            "Bundle all JS/CSS libraries (Plotly, itables, etc.) directly into the "
            "generated HTML report, producing a fully self-contained file that works "
            "offline and in air-gapped environments. "
            "The file will be larger and esbuild is required. "
            "Without this flag (default) libraries are loaded from CDN — "
            "smaller file, but requires an internet connection at viewing time. "
            "Exposed to the app as the PDSTOOLS_FULL_EMBED env var."
        ),
    )
    parser.add_argument(
        "--no-full-embed",
        dest="full_embed",
        action="store_false",
        help=(
            "Load JS/CSS libraries from CDN (default). "
            "Produces a smaller report file but requires internet access when viewing. "
            "Use --full-embed for offline/air-gapped environments."
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


def _aliases_for(app_key: str) -> list[str]:
    return [a for a, full in ALIASES.items() if full == app_key]


def list_apps() -> None:
    """Print one line per app: ``<key>\\t<display_name>\\t<aliases>``."""
    for key, info in APPS.items():
        aliases = ",".join(_aliases_for(key))
        print(f"{key}\t{info['display_name']}\t{aliases}")


def doctor() -> None:
    """Print environment health information for support diagnostics.

    Thin wrapper over :func:`pdstools.show_versions` with all
    diagnostic flags enabled. Exposed as the ``pdstools doctor`` CLI
    subcommand so users can share their environment without writing
    Python.
    """
    from .utils.show_versions import show_versions

    show_versions(include_runtime_diagnostics=True)


def main():
    # Formalised subcommand shape. Backwards-compat: ``pdstools [app]`` and
    # bare ``pdstools`` (interactive) still work — anything that isn't a
    # known subcommand is routed to ``run``.
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCOMMANDS:
        sub = sys.argv[1]
        if sub == "doctor":
            doctor()
            return
        if sub == "list":
            list_apps()
            return
        # ``run`` -> just strip and fall through to argparse
        del sys.argv[1]

    parser = create_parser()
    args, unknown = parser.parse_known_args()

    if args.list_apps:
        list_apps()
        return

    # Resolve alias to full app name
    if args.app and args.app in ALIASES:
        args.app = ALIASES[args.app]

    # Check for likely typos in pdstools arguments
    known_pdstools_args = [
        "--version",
        "--list",
        "--data-path",
        "--sample",
        "--filter",
        "--temp-dir",
        "--full-embed",
        "--no-full-embed",
    ]
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

    # Validate --data-path early so users get a clean error rather than
    # an opaque crash inside Streamlit.
    if args.data_path and not os.path.exists(args.data_path):
        print(
            f"Error: --data-path '{args.data_path}' does not exist.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            "Error: streamlit is not installed. Try installing the optional dependency group 'app'.\n"
            "If you are using uvx, try running uvx 'pdstools[app]' instead.",
        )
        sys.exit(1)

    # If no app is specified, prompt the user to choose. Prefer the
    # arrow-key picker (questionary, ships with the [app] extra); fall
    # back to the numeric prompt only when questionary isn't installed
    # or stdin isn't a TTY.
    if args.app is None and sys.stdin.isatty():
        try:
            import questionary
        except ImportError:
            questionary = None  # type: ignore[assignment]
        if questionary is not None:
            choice = questionary.select(
                "Select an app to run:",
                choices=[questionary.Choice(title=APPS[k]["display_name"], value=k) for k in APPS],
            ).ask()
            # ask() returns None on Ctrl+C / Esc — treat that as "user
            # asked to leave" rather than falling through to the numeric
            # prompt (which would just re-ask the same question).
            if choice is None:
                print("Exiting...", flush=True)
                sys.exit(0)
            args.app = choice

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
            except ValueError:
                print("Invalid input. Please try again.")

    # Propagate CLI flags to the Streamlit process via env vars
    if args.data_path:
        os.environ["PDSTOOLS_DATA_PATH"] = args.data_path
    if args.sample:
        os.environ["PDSTOOLS_SAMPLE_LIMIT"] = args.sample
        # Check if polars 64-bit index runtime is active
        try:
            import polars as _pl

            has_rt64 = _pl.get_index_type() == _pl.UInt64
        except (ImportError, AttributeError):
            has_rt64 = False
        if not has_rt64:
            print(
                "\n💡 Tip: For very large datasets (>2 billion elements), "
                "install the polars 64-bit runtime:\n"
                "   uv pip install 'polars[rt64]'\n",
                file=sys.stderr,
            )
    if args.filter:
        import json

        os.environ["PDSTOOLS_FILTER"] = json.dumps(args.filter)
    if args.temp_dir:
        os.environ["PDSTOOLS_TEMP_DIR"] = args.temp_dir
    # Propagate full_embed; only set the env var when explicitly provided
    # so the Streamlit app can keep its own default when the flag is absent.
    if args.full_embed:
        os.environ["PDSTOOLS_FULL_EMBED"] = "true"

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
    elif args.app == "launcher":
        # The launcher hosts the DA pages, so it inherits DA's XSRF
        # exemption (DA uses the file-uploader workaround that breaks
        # under XSRF). Standalone HC / IA launches keep XSRF on.
        sys.argv = [
            "streamlit",
            "run",
            filename,
            "--server.enableXsrfProtection",
            "false",
        ]
    else:  # health_check, impact_analyzer
        sys.argv = ["streamlit", "run", filename]

    if unknown:
        sys.argv.extend(unknown)
    if "--server.maxUploadSize" not in sys.argv:
        sys.argv.extend(["--server.maxUploadSize", "2000"])
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
