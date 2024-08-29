import math
import os
from glob import glob
from typing import Dict, List, Optional

import polars as pl
import polars.selectors as cs
from tqdm.auto import tqdm


class Anonymization:
    def __init__(
        self,
        path_to_files: str,
        temporary_path: str = "/tmp/anonymisation",
        output_file: str = "anonymised.parquet",
        skip_columns_with_prefix: Optional[List[str]] = None,
        batch_size: int = 500,
        file_limit: Optional[int] = None,
    ):
        """
        Initialize the Anonymization object.

        Once this class is initialised, simply run `.anonymize` to get started.

        Parameters
        ----------
        path_to_files : str
            The `glob pattern` towards the files to read in.
            For instance, if you have a number of `json` files
            in your ~/Downloads folder, you would use the following value:
            "~/Downloads/*.json"
        temporary_path : str, optional
            The temporary path to store intermediate files.
            Defaults to "/tmp/anonymisation".
        output_file : str, optional
            The name of the output file. Defaults to "anonymised.parquet".
        skip_columns_with_prefix : List[str], optional
            A list of column prefixes to skip during anonymization.
            Leave empty to use the default values: `Context_` and `Decision_`.
        batch_size : int, optional
            The batch size for processing the data. Defaults to 500.
        file_limit : int, optional
            The maximum number of files to process. Defaults to None.

        Examples
        --------
        >>> anonymizer = Anonymization(
        ...     path_to_files="~/Downloads/*.json",
        ...     batch_size=1000,
        ...     file_limit=10
        ... )
        >>> anonymizer.anonymize()
        """
        self.path_to_files = path_to_files
        self.temp_path = temporary_path
        self.output_file = output_file
        self.skip_col_prefix = skip_columns_with_prefix or ("Context_", "Decision_")
        self.batch_size = batch_size
        self.file_limit = file_limit

        try:
            os.mkdir(temporary_path)
        except FileExistsError:
            pass

    def anonymize(self, verbose: bool = True):
        """Anonymize the data.

        This method performs the anonymization process on the data files specified
        during initialization. It writes temporary parquet files, processes and
        writes the parquet files to a single file, and outputs the anonymized data
        to the specified output file.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose output during the anonymization process.
            Defaults to True.
        """
        if verbose:
            print("Writing temporary parquet files")
        chunked_files = self.preprocess(verbose=verbose)

        if verbose:
            print("Processing and writing parquet files to single file")
        self.process(chunked_files, verbose=verbose)
        if verbose:
            print(f"Succesfully anonymized data to {self.output_file}")

    @staticmethod
    def min_max(column_name: str, range: List[Dict[str, float]]) -> pl.Expr:
        """Normalize the values in a column using the min-max scaling method.

        Parameters
        ----------
        column_name : str
            The name of the column to be normalized.
        range : List[Dict[str, float]]
            A list of dictionaries containing the minimum and maximum values for the column.

        Returns
        -------
        pl.Expr
            A Polars expression representing the normalized column.

        Examples
        --------
        >>> range = [{"min": 0.0, "max": 100.0}]
        >>> min_max("age", range)
        Column "age" normalized using min-max scaling.
        """
        if range[0]["min"] == range[0]["max"]:  # pragma: no cover
            print(f"Column {column_name} only contains one value, so returning 0")
            return pl.lit(0.0).alias(column_name)
        return (pl.col(column_name) - pl.lit(range[0]["min"])) / (
            pl.lit(range[0]["max"] - range[0]["min"])
        )

    @staticmethod
    def _infer_types(df: pl.DataFrame):
        """Infers the types of columns in a DataFrame.

        Parameters
        ----------
        df (pl.DataFrame):
            The DataFrame for which to infer column types.

        Returns
        -------
        dict:
            A dictionary mapping column names to their inferred types.
            The inferred types can be either "numeric" or "symbolic".
        """
        types = dict()
        for col in df.columns:
            try:
                ser = df.get_column(col)
                try:
                    ser = ser.replace("", None)
                except Exception:
                    pass
                ser.cast(pl.Float64)
                types[col] = "numeric"
            except Exception:
                types[col] = "symbolic"

        return types

    @staticmethod
    def chunker(files: List[str], size: int):
        """Split a list of files into chunks of a specified size.

        Parameters
        ----------
        files (List[str]):
            A list of file names.
        size (int):
            The size of each chunk.

        Returns
        -------
        generator:
            A generator that yields chunks of files.

        Examples
        --------
            >>> files = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt', 'file5.txt']
            >>> chunks = chunker(files, 2)
            >>> for chunk in chunks:
            ...     print(chunk)
            ['file1.txt', 'file2.txt']
            ['file3.txt', 'file4.txt']
            ['file5.txt']
        """

        return (files[pos : pos + size] for pos in range(0, len(files), size))

    def chunk_to_parquet(self, files: List[str], i) -> str:
        """Convert a chunk of files to Parquet format.

        Parameters:
        files (List[str]):
            List of file paths to be converted.
        temp_path (str):
            Path to the temporary directory where the Parquet file will be saved.
        i:
            Index of the chunk.

        Returns:
        str: File path of the converted Parquet file.
        """

        init_df = pl.concat([pl.read_ndjson(n) for n in files], how="diagonal_relaxed")
        df = init_df.select(pl.all().exclude(pl.Null))
        types = self._infer_types(df)
        for n, t in types.items():
            if t == "numeric" and not df.schema[n].is_numeric():  # pragma: no cover
                df = df.with_columns(
                    pl.col(n).replace(pl.lit(""), None).cast(pl.Float64)
                )

        filename = os.path.join(self.temp_path, f"{i}.parquet")
        df.write_parquet(filename)
        return filename

    def preprocess(self, verbose: bool) -> List[str]:
        """
        Preprocesses the files in the specified path.

        Parameters
        ----------
            verbose (bool):
                Set to True to get a progress bar for the file count

        Returns
        -------
            list[str]: A list of the temporary bundled parquet files
        """

        files = glob(self.path_to_files)
        files.sort(key=os.path.getmtime)

        if self.file_limit:
            files = files[: self.file_limit]

        chunked_files = []

        length = math.ceil(len(files) / self.batch_size)
        for i, file_chunk in enumerate(
            tqdm(
                self.chunker(files, self.batch_size), total=length, disable=not verbose
            )
        ):
            chunked_files.append(self.chunk_to_parquet(file_chunk, i))

        return chunked_files

    def process(
        self,
        chunked_files: List[str],
        verbose: bool = True,
    ):
        """
        Process the data for anonymization.

        Parameters
        ----------
        chunked_files (list[str]):
            A list of the bundled temporary parquet files to process

        verbose (bool):
            Whether to print verbose output. Default is True.

        Raises
        ------
        ImportError:
            If polars-hash is not installed.

        Returns:
            None
        """
        try:  # to make it optional
            import polars_hash as plh
        except ImportError:  # pragma: no cover
            raise ImportError(
                "Polars-hash not installed. Please install using pip install polars-hash"
            )

        df = pl.concat(
            [pl.scan_parquet(f) for f in chunked_files], how="diagonal_relaxed"
        )

        symb_nonanonymised = [
            key for key in df.columns if key.startswith(tuple(self.skip_col_prefix))
        ]
        nums = [
            key
            for key, value in df.schema.items()
            if (value in cs.NUMERIC_DTYPES and key not in symb_nonanonymised)
        ]
        symb = [
            key
            for key in df.columns
            if (key not in nums and key not in symb_nonanonymised)
        ]
        if verbose:
            print(
                "Context_* and Decision_* columns (not anonymized): ",
                symb_nonanonymised,
            )
            print("Numeric columns:", nums)
            print("Symbolic columns:", symb)

        min_max_map = (
            df.select(
                [
                    pl.struct(
                        pl.col(num).min().alias("min"), pl.col(num).max().alias("max")
                    ).alias(num)
                    for num in nums
                ]
            )
            .collect()
            .to_dict(as_series=False)
        )

        anonymised_df = df.select(
            pl.col(symb_nonanonymised),
            plh.col(symb).chash.sha2_256(),
            *[self.min_max(c, min_max_map[c]) for c in nums],
        )

        anonymised_df.sink_parquet(self.output_file)
