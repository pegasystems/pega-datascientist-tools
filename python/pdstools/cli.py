# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pdstools[app]>=4.0.3",
# ]
# ///

import argparse
import sys
from importlib import resources

# App configuration with display names and paths
APPS = {
    "health_check": {
        "display_name": "Health Check",
        "path": "pdstools.app.health_check",
    },
    "decision_analyzer": {
        "display_name": "Decision Analyzer",
        "path": "pdstools.app.decision_analyzer",
    },
    "impact_analyzer": {
        "display_name": "Impact Analyzer",
        "path": "pdstools.app.impact_analyzer",
    },
}


def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line utility to run pdstools apps."
    )

    # Create help text with display names
    app_choices = list(APPS.keys())
    help_text = "The app to run: " + " | ".join(
        [f'"{key}" ({APPS[key]["display_name"]})' for key in app_choices]
    )

    parser.add_argument(
        "app",
        choices=app_choices,
        help=help_text,
        nargs="?",  # This makes the 'app' argument optional
        default=None,  # Explicitly set default to None
    )
    return parser


def main():
    parser = create_parser()
    args, unknown = parser.parse_known_args()

    run(args, unknown)


def run(args, unknown):
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            "Error: streamlit is not installed. Try installing the optionall dependency group 'app'.\n"
            "If you are using uvx, try running uvx 'pdstools[app]' instead."
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
                    f"\nPlease select an app to run (1-{len(app_list)}): "
                ).strip()

                # Check if it's a number
                if choice.isdigit() and 1 <= int(choice) <= len(app_list):
                    args.app = app_list[int(choice) - 1]
                    break

                # Check if it's an internal app name
                elif choice.lower() in APPS:
                    args.app = choice.lower()
                    break

                # Check if it's a display name (case insensitive)
                else:
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
                    valid_options.extend(
                        [app_info["display_name"] for app_info in APPS.values()]
                    )
                    print(
                        f"Invalid choice. Please enter: {', '.join(valid_options[:4])}..."
                    )

            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                sys.exit(0)
            except Exception:
                print("Invalid input. Please try again.")

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
