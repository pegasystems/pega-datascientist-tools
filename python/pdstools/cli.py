# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pdstools[app]>=4.0.3",
# ]
# ///

import argparse
import sys
from importlib import resources


def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line utility to run pdstools apps."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for the 'run' command
    run_parser = subparsers.add_parser("run", help="Run the specified pdstools app")
    run_parser.add_argument(
        "app",
        choices=["health_check", "decision_analyzer"],
        help='The app to run: "health_check" or "decision_analyzer"',
        nargs="?",  # This makes the 'app' argument optional
    )
    run_parser.set_defaults(func=run)

    # Set 'run' as the default subcommand
    parser.set_defaults(command="run", func=run)

    return parser


def main():
    parser = create_parser()
    args, unknown = parser.parse_known_args()

    # Manually handle the default command if none is provided
    if args.command is None:
        args.command = "run"

    args.func(args, unknown)


def run(args, unknown):
    from streamlit.web import cli as stcli

    # If no app is specified, prompt the user to choose
    if not hasattr(args, 'app') or args.app is None:
        available_apps = ["health_check", "decision_analyzer"]
        print("Available pdstools apps:")
        for i, app in enumerate(available_apps, 1):
            print(f"  {i}. {app}")
        
        while True:
            try:
                choice = input("\nPlease select an app to run (1-2): ").strip()
                if choice in ['1', '2']:
                    args.app = available_apps[int(choice) - 1]
                    break
                elif choice.lower() in available_apps:
                    args.app = choice.lower()
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or the app name.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                sys.exit(0)
            except Exception:
                print("Invalid input. Please try again.")

    print(f"Running {args.app} app.")
    print(unknown)
    
    if args.app == "decision_analyzer":
        with resources.path("pdstools.app.decision_analyzer", "Home.py") as filepath:
            filename = str(filepath)
        sys.argv = [
            "streamlit",
            "run",
            filename,
            "--server.enableXsrfProtection",
            "false",
        ]
    else:  # health_check
        with resources.path("pdstools.app.health_check", "Home.py") as filepath:
            filename = str(filepath)
        sys.argv = ["streamlit", "run", filename]

    if unknown:
        sys.argv.extend(unknown)
    if "--server.maxUploadSize" not in sys.argv:
        sys.argv.extend(["--server.maxUploadSize", "2000"])
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
