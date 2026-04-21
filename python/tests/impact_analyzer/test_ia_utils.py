"""Tests for Impact Analyzer utility functions and shared sampling utilities."""

import polars as pl
import pytest

from pdstools.utils.streamlit_utils import parse_sample_spec


class TestParseSampleSpec:
    """Test the shared parse_sample_spec function."""

    def test_absolute_count(self):
        assert parse_sample_spec("100000") == {"n": 100000}

    def test_k_suffix(self):
        assert parse_sample_spec("100k") == {"n": 100000}
        assert parse_sample_spec("100K") == {"n": 100000}

    def test_m_suffix(self):
        assert parse_sample_spec("1M") == {"n": 1000000}
        assert parse_sample_spec("1m") == {"n": 1000000}

    def test_fractional_k(self):
        assert parse_sample_spec("1.5k") == {"n": 1500}

    def test_percentage(self):
        assert parse_sample_spec("10%") == {"fraction": 0.1}
        assert parse_sample_spec("50%") == {"fraction": 0.5}
        assert parse_sample_spec("100%") == {"fraction": 1.0}

    def test_whitespace_stripped(self):
        assert parse_sample_spec("  100k  ") == {"n": 100000}

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_sample_spec("invalid")

    def test_zero_count(self):
        with pytest.raises(ValueError, match="positive"):
            parse_sample_spec("0")

    def test_negative_count(self):
        with pytest.raises(ValueError, match="positive"):
            parse_sample_spec("-100")

    def test_zero_percentage(self):
        with pytest.raises(ValueError, match="Percentage"):
            parse_sample_spec("0%")

    def test_over_100_percentage(self):
        with pytest.raises(ValueError, match="Percentage"):
            parse_sample_spec("150%")


class TestPrepareAndSaveRandom:
    """Test IA random sampling function."""

    def test_sample_by_count(self, tmp_path):
        from pdstools.app.impact_analyzer.ia_streamlit_utils import (
            prepare_and_save_random,
        )

        data = pl.DataFrame({"a": range(1000), "b": range(1000)}).lazy()

        sampled, path = prepare_and_save_random(data, n=100, output_dir=str(tmp_path))

        assert path is not None
        assert sampled.collect().height == 100
        assert pl.read_parquet(path).height == 100

    def test_sample_by_fraction(self, tmp_path):
        from pdstools.app.impact_analyzer.ia_streamlit_utils import (
            prepare_and_save_random,
        )

        data = pl.DataFrame({"a": range(1000), "b": range(1000)}).lazy()

        sampled, path = prepare_and_save_random(data, fraction=0.1, output_dir=str(tmp_path))

        assert path is not None
        assert sampled.collect().height == 100

    def test_no_sampling_when_within_limit(self, tmp_path):
        from pdstools.app.impact_analyzer.ia_streamlit_utils import (
            prepare_and_save_random,
        )

        data = pl.DataFrame({"a": range(50)}).lazy()

        sampled, path = prepare_and_save_random(data, n=100, output_dir=str(tmp_path))

        assert path is None
        assert sampled.collect().height == 50

    def test_requires_n_or_fraction(self, tmp_path):
        from pdstools.app.impact_analyzer.ia_streamlit_utils import (
            prepare_and_save_random,
        )

        data = pl.DataFrame({"a": range(100)}).lazy()

        with pytest.raises(ValueError, match="n or fraction"):
            prepare_and_save_random(data, output_dir=str(tmp_path))
