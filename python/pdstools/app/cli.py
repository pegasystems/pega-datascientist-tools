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
    from . import Home

    print("Running app.")
    print(args)
    filename = Home.__file__
    sys.argv = ["streamlit", "run", filename]
    if len(args) > 1:
        sys.argv.extend(args[1:])
    if "--server.maxUploadSize" not in sys.argv:
        sys.argv.extend(["--server.maxUploadSize", "2000"])
    sys.exit(stcli.main())


def help(*args):
    msg = (
        "Command line utility to run pdstools apps. ",
        "Currently, only the health check is provided, ",
        "but this will be expanded in the future. \n\n",
        "To run the healthcheck, enter the following command: \n",
        "`pdstools run`",
    )
    print("".join(msg))
