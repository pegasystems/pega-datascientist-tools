import os
import shutil
import subprocess
import logging

from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

class QuartoHelper:
    
    @staticmethod
    def copy_report_resources(resource_dict: list[tuple[str, str]]):
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
                
    @staticmethod
    def run_quarto(
        qmd_file: str,
        output_filename: str,
        output_type: str = "html",
        params: Dict = {},
        project: Dict = {"type": "default"},
        analysis: Dict = {},
        temp_dir: Path = Path("."),
        verbose: bool = False,
    ) -> int:
        """Run the Quarto command to generate the report."""

        Reports._write_params_files(
            temp_dir,
            params=params,
            project=project,
            analysis=analysis,
        )

        quarto_exec, _ = get_quarto_with_version(verbose)

        command = [
            str(quarto_exec),
            "render",
            qmd_file,
            "--to",
            output_type,
            "--output",
            output_filename,
            "--execute-params",
            "params.yml",
        ]

        if verbose:
            print(f"Executing: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            cwd=temp_dir,
            text=True,
            bufsize=1,  # Line buffered
        )

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if verbose:
                    print(line)
                logger.info(line)
        else:  # pragma: no cover
            logger.warning("subprocess.stdout is None, unable to read output")

        return_code = process.wait()
        message = f"Quarto process exited with return code {return_code}"
        logger.info(message)

        return return_code
        
        