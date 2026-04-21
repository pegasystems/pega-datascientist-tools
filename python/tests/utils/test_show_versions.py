"""Tests for pdstools.utils.show_versions."""

from unittest.mock import patch

import polars as pl
import pytest

import pdstools
from pdstools import __version__
from pdstools.utils import show_versions as sv_module


def test_show_versions_print(capsys):
    pdstools.show_versions()
    captured = capsys.readouterr()
    assert "Version info" in captured.out
    assert "pdstools" in captured.out
    assert __version__ in captured.out
    assert "Dependencies" in captured.out


def test_show_versions_return_string():
    result = pdstools.show_versions(False)
    assert isinstance(result, str)
    assert "Version info" in result
    assert __version__ in result
    assert "Dependencies" in result


def test_show_versions_no_dependencies():
    result = sv_module.show_versions(print_output=False, include_dependencies=False)
    assert isinstance(result, str)
    assert "Version info" in result
    assert "Dependencies" not in result


class TestGetDependencyVersion:
    def test_existing_module(self):
        version = sv_module._get_dependency_version("polars")
        assert version != "<not installed>"

    def test_strips_version_specifier(self):
        version = sv_module._get_dependency_version("polars>=1.0")
        assert version != "<not installed>"

    def test_missing_module(self):
        version = sv_module._get_dependency_version("definitely_not_a_real_module_xyz")
        assert version == "<not installed>"

    def test_unexpected_import_error_handled(self):
        with patch.object(
            sv_module.importlib,
            "import_module",
            side_effect=RuntimeError("boom"),
        ):
            version = sv_module._get_dependency_version("polars")
            assert version == "<not installed>"


class TestExpandNestedDeps:
    def test_no_nested(self):
        extras = {"required": {"polars>=1.0"}, "extra1": {"plotly"}}
        out = sv_module.expand_nested_deps(extras)
        assert out["required"] == {"polars>=1.0"}
        assert out["extra1"] == {"plotly"}

    def test_nested_resolved(self):
        extras = {
            "adm": {"plotly"},
            "all": {"pdstools[adm]"},
        }
        out = sv_module.expand_nested_deps(extras)
        assert out["adm"] == {"plotly"}
        assert out["all"] == {"plotly"}

    def test_unresolvable_nested_falls_back(self):
        extras = {"x": {"pdstools[nonexistent]"}}
        out = sv_module.expand_nested_deps(extras)
        assert out["x"] == {"pdstools[nonexistent]"}


def test_grouped_dependencies_has_required():
    deps = sv_module.grouped_dependencies()
    assert "required" in deps
    assert isinstance(deps["required"], set)


class TestDependencyTable:
    def test_returns_polars_dataframe(self):
        table = sv_module._dependency_table(public_only=False)
        assert isinstance(table, pl.DataFrame)
        assert table.columns[0] == "group"
        assert table.height > 0

    def test_public_only_excludes_private_groups(self):
        table_all = sv_module._dependency_table(public_only=False)
        table_pub = sv_module._dependency_table(public_only=True)
        all_groups = set(table_all["group"].to_list())
        pub_groups = set(table_pub["group"].to_list())
        for private in ("dev", "docs", "tests"):
            assert private in all_groups
            assert private not in pub_groups

    def test_marks_membership_with_check_or_x(self):
        table = sv_module._dependency_table(public_only=False)
        for col in table.columns:
            if col == "group":
                continue
            values = set(table[col].to_list())
            assert values.issubset({"√", "X"})


class TestDependencyGreatTable:
    def test_public(self):
        try:
            from great_tables import GT
        except ImportError:
            pytest.skip("great_tables not installed")
        tab = sv_module.dependency_great_table(public_only=True)
        assert isinstance(tab, GT)

    def test_full(self):
        try:
            from great_tables import GT
        except ImportError:
            pytest.skip("great_tables not installed")
        tab = sv_module.dependency_great_table(public_only=False)
        assert isinstance(tab, GT)
