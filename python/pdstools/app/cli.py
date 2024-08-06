def main():
    import sys

    if len(sys.argv) == 1:
        print("No arguments provided. Please see `pdstools help`.")
    elif sys.argv[1] == "run":
        run(*sys.argv[1:])
    elif sys.argv[1] == "help":
        help(*sys.argv[1:])
    else:
        print(f"Unknown argument {sys.argv[1]}. Please See `pdstools help` for help")


def run(*args):
    from streamlit.web import cli as stcli
    import sys
    import os

    print("Running app.")
    print(args)
    if len(args) > 1 and args[1] == "decision_analyzer":
        filename = os.path.join(
            os.path.dirname(__file__), "decision_analyzer", "home.py"
        )
        sys.argv = [
            "streamlit",
            "run",
            filename,
            "--server.enableXsrfProtection",
            "false",
        ]

    else:
        filename = os.path.join(os.path.dirname(__file__), "health_check", "Home.py")
        sys.argv = ["streamlit", "run", filename]
    if len(args) > 2:
        sys.argv.extend(args[2:])
    if "--server.maxUploadSize" not in sys.argv:
        sys.argv.extend(["--server.maxUploadSize", "2000"])
    sys.exit(stcli.main())


def help(*args):
    msg = (
        "Command line utility to run pdstools apps. ",
        "To run the healthcheck, enter the following command: \n",
        "`pdstools run`",
        "To run the Decision Analyzer(Explainability Extract), enter the following command: \n",
        "`pdstools run decision_analyzer`",
    )
    print("".join(msg))
