"""Tests for pdstools.cli — parser creation and argument parsing only.

Does NOT test run() or main() since they require streamlit.
"""

import pytest

from pdstools.cli import APPS, create_parser


# ---------------------------------------------------------------------------
# APPS dict
# ---------------------------------------------------------------------------


class TestAppsDict:
    def test_expected_keys(self):
        assert set(APPS.keys()) == {"health_check", "decision_analyzer", "impact_analyzer"}

    def test_each_entry_has_display_name_and_path(self):
        for key, value in APPS.items():
            assert "display_name" in value, f"{key} missing 'display_name'"
            assert "path" in value, f"{key} missing 'path'"

    def test_display_names_are_strings(self):
        for key, value in APPS.items():
            assert isinstance(value["display_name"], str)
            assert len(value["display_name"]) > 0

    def test_paths_are_dotted_module_paths(self):
        for key, value in APPS.items():
            path = value["path"]
            assert isinstance(path, str)
            assert path.startswith("pdstools.app.")
            assert path.count(".") >= 2


# ---------------------------------------------------------------------------
# create_parser
# ---------------------------------------------------------------------------


class TestCreateParser:
    def test_parser_is_created(self):
        parser = create_parser()
        assert parser is not None

    def test_parser_choices(self):
        parser = create_parser()
        # The 'app' positional argument should have correct choices (including aliases)
        app_action = None
        expected_choices = {"health_check", "decision_analyzer", "impact_analyzer", "hc", "da", "ia"}
        for action in parser._actions:
            if hasattr(action, "choices") and action.choices is not None:
                if set(action.choices) == expected_choices:
                    app_action = action
                    break
        assert app_action is not None, "Parser should have an 'app' argument with the app choices and aliases"

    def test_parser_description(self):
        parser = create_parser()
        assert "pdstools" in parser.description.lower()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_default_values_no_args(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.app is None
        assert args.data_path is None
        assert args.sample is None
        assert args.temp_dir is None

    def test_app_argument(self):
        parser = create_parser()
        for app_name in ("health_check", "decision_analyzer", "impact_analyzer"):
            args = parser.parse_args([app_name])
            assert args.app == app_name

    def test_data_path_flag(self):
        parser = create_parser()
        args = parser.parse_args(["decision_analyzer", "--data-path", "/tmp/my_data.parquet"])
        assert args.data_path == "/tmp/my_data.parquet"

    def test_sample_flag(self):
        parser = create_parser()
        args = parser.parse_args(["health_check", "--sample", "100000"])
        assert args.sample == "100000"

    def test_sample_flag_percentage(self):
        parser = create_parser()
        args = parser.parse_args(["health_check", "--sample", "10%"])
        assert args.sample == "10%"

    def test_temp_dir_flag(self):
        parser = create_parser()
        args = parser.parse_args(["impact_analyzer", "--temp-dir", "/tmp/scratch"])
        assert args.temp_dir == "/tmp/scratch"

    def test_all_flags_together(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "decision_analyzer",
                "--data-path",
                "/data/extract",
                "--sample",
                "50000",
                "--temp-dir",
                "/tmp/work",
            ]
        )
        assert args.app == "decision_analyzer"
        assert args.data_path == "/data/extract"
        assert args.sample == "50000"
        assert args.temp_dir == "/tmp/work"

    def test_invalid_app_raises_error(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["nonexistent_app"])

    def test_parse_known_args_passes_unknown(self):
        """Unknown args should be captured separately (like streamlit flags)."""
        parser = create_parser()
        args, unknown = parser.parse_known_args(
            [
                "health_check",
                "--server.maxUploadSize",
                "5000",
            ]
        )
        assert args.app == "health_check"
        assert "--server.maxUploadSize" in unknown
        assert "5000" in unknown
