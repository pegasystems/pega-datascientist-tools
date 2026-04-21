"""Hash-based anonymisation of Pega Historical Datasets."""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Iterator
from glob import glob

import polars as pl

logger = logging.getLogger(__name__)


class Anonymization:
    """Anonymise Pega datasets (in particular, the Historical Dataset).

    Numeric columns are min-max scaled to ``[0, 1]``.  Symbolic columns
    are hashed with SHA-256.  Columns whose name starts with one of the
    ``skip_columns_with_prefix`` values are passed through unchanged
    (by default ``Context_*`` and ``Decision_*``).

    Once constructed, call :meth:`anonymize` to run the pipeline.  All
    file system work happens then; ``__init__`` is pure.

    Parameters
    ----------
    path_to_files : str
        Glob pattern matching the input files, e.g. ``"~/Downloads/*.json"``.
    temporary_path : str, optional
        Directory used for intermediate parquet chunks.  Defaults to a
        fresh ``tempfile.mkdtemp`` directory created on first use.
    output_file : str, default="anonymised.parquet"
        Path to write the final anonymised parquet file.
    skip_columns_with_prefix : list[str], optional
        Column-name prefixes to leave unchanged. Defaults to
        ``("Context_", "Decision_")``.
    batch_size : int, default=500
        Number of input files combined per intermediate parquet chunk.
    file_limit : int, optional
        Process at most this many files (useful for testing).

    Examples
    --------
    >>> Anonymization(
    ...     path_to_files="~/Downloads/*.json",
    ...     batch_size=1000,
    ...     file_limit=10,
    ... ).anonymize()
    """

    def __init__(
        self,
        path_to_files: str,
        temporary_path: str | None = None,
        output_file: str = "anonymised.parquet",
        skip_columns_with_prefix: list[str] | tuple[str, ...] | None = None,
        batch_size: int = 500,
        file_limit: int | None = None,
    ):
        self.path_to_files = path_to_files
        self._temp_path: str | None = temporary_path
        self.output_file = output_file
        self.skip_col_prefix: tuple[str, ...] = tuple(skip_columns_with_prefix or ("Context_", "Decision_"))
        self.batch_size = batch_size
        self.file_limit = file_limit

    @property
    def temp_path(self) -> str:
        """Lazily create (and cache) the temp directory."""
        if self._temp_path is None:
            self._temp_path = tempfile.mkdtemp(prefix="pdstools_anonymise_")
        else:
            os.makedirs(self._temp_path, exist_ok=True)
        return self._temp_path

    def anonymize(self, verbose: bool = True) -> None:
        """Run the full anonymisation pipeline.

        Parameters
        ----------
        verbose : bool, default=True
            Print progress messages between stages.
        """
        if verbose:
            print("Writing temporary parquet files")
        chunked_files = self.preprocess(verbose=verbose)

        if verbose:
            print("Processing and writing parquet files to single file")
        self.process(chunked_files, verbose=verbose)
        if verbose:
            print(f"Successfully anonymized data to {self.output_file}")

    @staticmethod
    def min_max(column_name: str, value_range: list[dict[str, float]]) -> pl.Expr:
        """Return a min-max scaling expression for ``column_name``.

        Parameters
        ----------
        column_name : str
            Column to normalise.
        value_range : list[dict[str, float]]
            Single-element list whose dict has ``"min"`` and ``"max"``
            keys, matching the shape produced by Polars when collecting
            a struct of ``min``/``max`` aggregations.

        Returns
        -------
        pl.Expr
            ``(col - min) / (max - min)``, or the literal ``0.0`` when
            min == max.
        """
        lo = value_range[0]["min"]
        hi = value_range[0]["max"]
        if lo == hi:  # pragma: no cover
            logger.info("Column %s only contains one value, returning 0", column_name)
            return pl.lit(0.0).alias(column_name)
        return (pl.col(column_name) - pl.lit(lo)) / (pl.lit(hi - lo))

    @staticmethod
    def _infer_types(df: pl.DataFrame) -> dict[str, str]:
        """Classify each column as ``"numeric"`` or ``"symbolic"``.

        A column is considered numeric if its values can be cast to
        ``Float64`` (after replacing empty strings with null).
        """
        types: dict[str, str] = {}
        for col in df.collect_schema().names():
            try:
                ser = df.get_column(col)
                try:
                    ser = ser.replace("", None)
                except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError):
                    pass
                ser.cast(pl.Float64)
                types[col] = "numeric"
            except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError, ValueError) as exc:
                logger.debug("Column %s defaulted to symbolic: %s", col, exc)
                types[col] = "symbolic"
        return types

    @staticmethod
    def chunker(files: list[str], size: int) -> Iterator[list[str]]:
        """Yield successive ``size``-element slices of ``files``."""
        return (files[pos : pos + size] for pos in range(0, len(files), size))

    def chunk_to_parquet(self, files: list[str], i: int) -> str:
        """Read a chunk of NDJSON files and write them as a parquet file.

        Parameters
        ----------
        files : list[str]
            NDJSON file paths to combine.
        i : int
            Chunk index (used in the output filename).

        Returns
        -------
        str
            Path to the parquet file produced.
        """
        init_df = pl.concat([pl.read_ndjson(n) for n in files], how="diagonal_relaxed")
        df = init_df.select(pl.all().exclude(pl.Null))
        types = self._infer_types(df)
        for n, t in types.items():
            if t == "numeric" and not df.schema[n].is_numeric():  # pragma: no cover
                df = df.with_columns(
                    pl.col(n).replace(pl.lit(""), None).cast(pl.Float64),
                )

        filename = os.path.join(self.temp_path, f"{i}.parquet")
        df.write_parquet(filename)
        return filename

    def preprocess(self, verbose: bool) -> list[str]:
        """Convert input files into intermediate parquet chunks.

        Parameters
        ----------
        verbose : bool
            Show a tqdm progress bar over chunks (if installed).

        Returns
        -------
        list[str]
            Paths to the temporary chunked parquet files.
        """
        files = glob(self.path_to_files)
        files.sort(key=os.path.getmtime)

        if self.file_limit:
            files = files[: self.file_limit]

        chunked_files: list[str] = []
        length = math.ceil(len(files) / self.batch_size)

        try:
            from tqdm.auto import tqdm

            iterable = tqdm(
                self.chunker(files, self.batch_size),
                total=length,
                disable=not verbose,
            )
        except ImportError:
            iterable = self.chunker(files, self.batch_size)

        for i, file_chunk in enumerate(iterable):
            chunked_files.append(self.chunk_to_parquet(file_chunk, i))

        return chunked_files

    def process(self, chunked_files: list[str], verbose: bool = True) -> None:
        """Hash, scale, and write the final anonymised parquet file.

        Parameters
        ----------
        chunked_files : list[str]
            Intermediate parquet files produced by :meth:`preprocess`.
        verbose : bool, default=True
            Print which columns will be hashed / scaled / preserved.

        Raises
        ------
        MissingDependenciesException
            When ``polars-hash`` is not installed.
        """
        try:
            import polars_hash as plh
        except ImportError:  # pragma: no cover
            from ..utils.namespaces import MissingDependenciesException

            raise MissingDependenciesException(
                ["polars-hash"],
                namespace="Anonymization",
                deps_group="pega_io",
            )

        df: pl.LazyFrame = pl.concat(
            [pl.scan_parquet(f) for f in chunked_files],
            how="diagonal_relaxed",
        )
        schema = df.collect_schema()

        skipped = [key for key in schema.names() if key.startswith(self.skip_col_prefix)]
        nums = [key for key, value in schema.items() if value.is_numeric() and key not in skipped]
        symb = [key for key in schema.names() if key not in nums and key not in skipped]
        if verbose:
            print("Context_* and Decision_* columns (not anonymized):", skipped)
            print("Numeric columns:", nums)
            print("Symbolic columns:", symb)

        min_max_df = df.select(
            [
                pl.struct(
                    pl.col(num).min().alias("min"),
                    pl.col(num).max().alias("max"),
                ).alias(num)
                for num in nums
            ],
        ).collect()
        min_max_map = min_max_df.to_dict(as_series=False)

        anonymised_df = df.select(
            pl.col(skipped),
            plh.col(symb).chash.sha2_256(),
            *[self.min_max(c, min_max_map[c]) for c in nums],
        )

        anonymised_df.sink_parquet(self.output_file)
