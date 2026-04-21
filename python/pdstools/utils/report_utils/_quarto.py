"""Quarto execution helpers: run, version detection, callouts, credits."""

import datetime
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path

from ._common import logger
from ._html import _inline_css


def _write_params_files(
    temp_dir: Path,
    params: dict | None = None,
    project: dict = {"type": "default"},
    analysis: dict | None = None,
    full_embed: bool = False,
) -> None:
    """Write parameters to YAML files for Quarto processing.

    Parameters
    ----------
    temp_dir : Path
        Directory where YAML files will be written
    params : dict, optional
        Parameters to write to params.yml, by default None
    project : dict, optional
        Project configuration to write to _quarto.yml, by default {"type": "default"}
    analysis : dict, optional
        Analysis configuration to write to _quarto.yml, by default None
    full_embed : bool, default=False
        When True, embeds all resources (JavaScript libraries like Plotly,
        itables, etc.) for a fully standalone HTML (larger output).
        When False, loads JavaScript libraries from CDN and skips esbuild
        bundling (smaller output, but requires internet).

    Returns
    -------
    None

    """
    import yaml  # type: ignore[import-untyped]  # types-PyYAML not in project deps

    params = params or {}
    analysis = analysis or {}

    # Parameters to python code
    with open(temp_dir / "params.yml", "w") as f:
        yaml.dump(
            params,
            f,
        )

    # When not using full_embed, disable embed-resources so Quarto does not
    # invoke esbuild to bundle JavaScript libraries (Plotly, itables, etc.).
    # This avoids failures in environments where esbuild is unavailable
    # (e.g. DJS Docker images that removed it due to CVE issues).
    # See GitHub issue #620.
    # plotly-connected: false = load Plotly from CDN (smaller file)
    # plotly-connected: true = embed Plotly (larger file)
    embed = full_embed
    html_format: dict = {
        "embed-resources": embed,
        "plotly-connected": embed,
    }

    quarto_config: dict = {
        "project": project,
        "analysis": analysis,
        "format": {
            "html": html_format,
        },
    }

    with open(temp_dir / "_quarto.yml", "w") as f:
        yaml.dump(quarto_config, f)


def run_quarto(
    qmd_file: str | None = None,
    output_filename: str | None = None,
    output_type: str | None = "html",
    params: dict | None = None,
    project: dict = {"type": "default"},
    analysis: dict | None = None,
    temp_dir: Path = Path(),
    *,
    full_embed: bool = False,
) -> int:
    """Run the Quarto command to generate the report.

    Parameters
    ----------
    qmd_file : str, optional
        Path to the Quarto markdown file to render, by default None
    output_filename : str, optional
        Name of the output file, by default None
    output_type : str, optional
        Type of output format (html, pdf, etc.), by default "html"
    params : dict, optional
        Parameters to pass to Quarto execution, by default None
    project : dict, optional
        Project configuration settings, by default {"type": "default"}
    analysis : dict, optional
        Analysis configuration settings, by default None
    temp_dir : Path, optional
        Temporary directory for processing, by default Path(".")
    full_embed : bool, default=False
        When True, fully embeds all JavaScript libraries (Plotly, itables,
        etc.) into the HTML output (larger file).
        When False, loads JavaScript libraries from CDN and skips esbuild
        bundling, avoiding the need for esbuild (see issue #620).

    Returns
    -------
    int
        Return code from the Quarto process (0 for success)

    Raises
    ------
    RuntimeError
        If the Quarto process fails (non-zero return code), includes captured output
    subprocess.SubprocessError
        If the Quarto command fails to execute
    FileNotFoundError
        If required files are not found

    """

    def get_command() -> list[str]:
        quarto_exec, _ = get_quarto_with_version()
        _command = [str(quarto_exec), "render"]

        if qmd_file is not None:
            _command.append(qmd_file)

        options = _set_command_options(
            output_type=output_type,
            output_filename=output_filename,
            execute_params=params is not None,
        )

        _command.extend(options)
        return _command

    if params is not None:
        _write_params_files(
            temp_dir,
            params=params,
            project=project,
            analysis=analysis,
            full_embed=full_embed,
        )

    # render file or render project with options
    command = get_command()

    logger.info("Executing: %s in temp directory %s", " ".join(command), temp_dir)

    # Set QUARTO_PYTHON to ensure Quarto uses the same Python that's running pdstools.
    # This is critical for isolated environments (uv tool, pipx) where the default
    # system Python may not have the required dependencies like ipykernel.
    import sys

    env = os.environ.copy()
    env["QUARTO_PYTHON"] = sys.executable
    logger.info(f"Setting QUARTO_PYTHON to: {sys.executable}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        cwd=temp_dir,
        env=env,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Capture all output for potential error reporting
    output_lines: list[str] = []
    if process.stdout is not None:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            output_lines.append(line)
            logger.info(line)
    else:  # pragma: no cover
        logger.warning("subprocess.stdout is None, unable to read output")

    return_code = process.wait()
    message = f"Quarto process exited with return code {return_code}"
    logger.info(message)

    # Raise an exception with captured output if Quarto failed
    if return_code != 0:
        captured_output = "\n".join(output_lines)
        # Surface common, actionable failure modes at the top of the message
        # so they're not buried in hundreds of lines of Quarto output.
        hints = []
        permission_lines = [
            ln for ln in output_lines if "permission denied" in ln.lower() or "access is denied" in ln.lower()
        ]
        if permission_lines:
            hints.append(
                "Permission denied while running Quarto. Common causes: the "
                "output directory or temp directory is not writable, the Quarto "
                "binary is locked by an antivirus scanner, or a previous run "
                "left files open. Offending log lines:\n  - " + "\n  - ".join(permission_lines[:5])
            )
        hint_block = ("\nHint: " + "\n\nHint: ".join(hints) + "\n") if hints else ""
        raise RuntimeError(
            f"Quarto rendering failed with return code {return_code}.{hint_block}\nOutput:\n{captured_output}",
        )

    if not full_embed and output_type == "html" and output_filename is not None:
        html_path = temp_dir / output_filename
        if html_path.is_file():
            n_inlined = _inline_css(html_path, temp_dir)
            logger.info("Inlined %d CSS stylesheet(s) into %s", n_inlined, output_filename)

    return return_code


def _set_command_options(
    output_type: str | None = None,
    output_filename: str | None = None,
    execute_params: bool = False,
) -> list[str]:
    """Set the options for the Quarto command.

    Parameters
    ----------
    output_type : str, optional
        Output format type (html, pdf, etc.), by default None
    output_filename : str, optional
        Name of the output file, by default None
    execute_params : bool, optional
        Whether to include parameter execution flag, by default False

    Returns
    -------
    list[str]
        list of command line options for Quarto

    """
    options = []
    if output_type is not None:
        options.append("--to")
        options.append(output_type)
    if output_filename is not None:
        options.append("--output")
        options.append(output_filename)
    if execute_params:
        options.append("--execute-params")
        options.append("params.yml")
    return options


def _get_cmd_output(args: list[str]) -> list[str]:
    """Get command output in an OS-agnostic way."""
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.split("\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run command {' '.join(args)}: {e}")
        raise FileNotFoundError(
            f"Command failed. Make sure {args[0]} is installed and in the system PATH.",
        )


def _get_version_only(versionstr: str) -> str:
    """Extract version number from version string."""
    match = re.search(r"(\d+(?:\.\d+)*)", versionstr)
    return match.group(1) if match else ""


def get_quarto_with_version() -> tuple[Path, str]:
    """Get Quarto executable path and version."""
    try:
        quarto_path = shutil.which("quarto")
        if not quarto_path:
            raise FileNotFoundError(
                "Quarto executable not found. Please ensure Quarto is installed and in the system PATH.",
            )
        executable = Path(quarto_path)

        version_string = _get_version_only(_get_cmd_output(["quarto", "--version"])[0])
        logger.info("quarto version: %s", version_string)
        return executable, version_string
    except Exception as e:
        logger.error(f"Error getting quarto version: {e}")
        raise


def get_pandoc_with_version() -> tuple[Path, str]:
    """Get Pandoc executable path and version."""
    try:
        pandoc_path = shutil.which("pandoc")
        if not pandoc_path:
            raise FileNotFoundError(
                "Pandoc executable not found. Please ensure Pandoc is installed and in the system PATH.",
            )
        executable = Path(pandoc_path)
        if not executable:
            raise FileNotFoundError(
                "Pandoc executable not found. Please ensure Pandoc is installed and in the system PATH.",
            )

        version_string = _get_version_only(_get_cmd_output(["pandoc", "--version"])[0])
        logger.info("pandoc version: %s", version_string)
        return executable, version_string
    except Exception as e:
        logger.error(f"Error getting pandoc version: {e}")
        raise


def quarto_print(text):
    from IPython.display import Markdown, display

    display(Markdown(text))


def quarto_callout_info(info):
    quarto_print(
        """
::: {.callout-note}
%s
:::
"""
        % info,
    )


def quarto_callout_important(info):
    quarto_print(
        """
::: {.callout-important}
%s
:::
"""
        % info,
    )


def quarto_plot_exception(plot_name: str, e: Exception):
    quarto_print(
        """
::: {.callout-important collapse="true"}
## Error rendering %s plot: %s

%s
:::
"""
        % (plot_name, e, traceback.format_exc()),
    )


def quarto_callout_no_prediction_data_warning(extra=""):
    quarto_callout_important(f"Prediction Data is not available. {extra}")


def quarto_callout_no_predictor_data_warning(extra=""):
    quarto_callout_important(f"Predictor Data is not available. {extra}")


def show_credits(quarto_source: str | None = None):
    """Display a credits section with build metadata at the end of a report.

    Prints a formatted block containing the generation timestamp, Quarto and
    Pandoc versions, and optionally the source notebook path.

    Parameters
    ----------
    quarto_source : str, optional
        Path or identifier of the source .qmd file. Include this for
        standalone reports where knowing the source is useful. Omit for
        Quarto website projects where pages are generated from templates.
    """
    _, quarto_version = get_quarto_with_version()
    _, pandoc_version = get_pandoc_with_version()

    timestamp_str = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    lines = [
        f"Document created at: {timestamp_str}",
    ]
    if quarto_source:
        lines.append(f"This notebook: {quarto_source}")
    lines.extend(
        [
            f"Quarto runtime: {quarto_version}",
            f"Pandoc: {pandoc_version}",
        ]
    )

    quarto_print("\n\n    ".join([""] + lines) + "\n\n    ")
