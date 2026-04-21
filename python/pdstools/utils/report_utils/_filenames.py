"""Filename generation and resource copying for report rendering."""

import os
import shutil
from pathlib import Path


def get_output_filename(
    name: str | None,
    report_type: str,
    model_id: str | None = None,
    output_type: str = "html",
) -> str:
    """Generate the output filename based on the report parameters.

    If ``name`` already ends with ``.{output_type}`` (case-insensitive), it
    is treated as a full filename and returned verbatim (after replacing
    spaces with underscores). Otherwise the filename is composed from
    ``report_type``, ``name``, and ``model_id`` with the extension appended.
    """
    if name:
        name = name.replace(" ", "_")
        if name.lower().endswith(f".{output_type.lower()}"):
            return name
    if report_type == "ModelReport":
        return f"{report_type}_{name}_{model_id}.{output_type}" if name else f"{report_type}_{model_id}.{output_type}"
    return f"{report_type}_{name}.{output_type}" if name else f"{report_type}.{output_type}"


def copy_quarto_file(qmd_file: str, temp_dir: Path) -> None:
    """Copy the report quarto file to the temporary directory.

    Parameters
    ----------
    qmd_file : str
        Name of the Quarto markdown file to copy
    temp_dir : Path
        Destination directory to copy files to

    Returns
    -------
    None

    """
    from pdstools import __reports__

    shutil.copy(__reports__ / qmd_file, temp_dir)
    shutil.copytree(__reports__ / "assets", temp_dir / "assets", dirs_exist_ok=True)


def copy_report_resources(resource_dict: list[tuple[str, str]]):
    """Copy report resources from the reports directory to specified destinations.

    Parameters
    ----------
    resource_dict : list[tuple[str, str]]
        list of tuples containing (source_path, destination_path) pairs

    Returns
    -------
    None

    """
    from pdstools import __reports__

    for src, dest in resource_dict:
        source_path = __reports__ / src
        destination_path = dest

        if destination_path == "":
            destination_path = "./"

        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy(source_path, destination_path)
