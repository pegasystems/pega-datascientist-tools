"""Tests for pdstools.cli."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from pdstools import cli as cli_module
from pdstools.cli import ALIASES, APPS, check_for_typos, create_parser, main, run


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
        assert args.full_embed is False

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

    def test_full_embed_flag(self):
        parser = create_parser()
        args = parser.parse_args(["health_check", "--full-embed"])
        assert args.full_embed is True

    def test_no_full_embed_flag(self):
        parser = create_parser()
        args = parser.parse_args(["health_check", "--no-full-embed"])
        assert args.full_embed is False

    def test_full_embed_default_is_false(self):
        parser = create_parser()
        args = parser.parse_args(["health_check"])
        assert args.full_embed is False

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


# ---------------------------------------------------------------------------
# check_for_typos
# ---------------------------------------------------------------------------


class TestCheckForTypos:
    KNOWN = ["--version", "--data-path", "--sample", "--filter", "--temp-dir"]

    def test_no_unknown_args_returns_empty(self):
        assert check_for_typos([], self.KNOWN) == []

    def test_no_typo_unknown_returns_empty(self):
        # Streamlit-style args (no close match) shouldn't be flagged
        result = check_for_typos(["--server.port", "8501"], self.KNOWN)
        assert result == []

    def test_typo_detected(self):
        # "--data_path" → "--data-path"
        result = check_for_typos(["--data_path"], self.KNOWN)
        assert len(result) == 1
        typo, suggestion, similarity = result[0]
        assert typo == "--data_path"
        assert suggestion == "--data-path"
        assert 0 < similarity <= 1

    def test_only_double_dash_args_checked(self):
        # Single-dash args and bare values should be ignored
        result = check_for_typos(["-x", "value", "--sampel"], self.KNOWN)
        assert len(result) == 1
        assert result[0][0] == "--sampel"
        assert result[0][1] == "--sample"

    def test_multiple_typos(self):
        result = check_for_typos(["--versoin", "--temp_dir"], self.KNOWN)
        typos = {t[0]: t[1] for t in result}
        assert typos.get("--versoin") == "--version"
        assert typos.get("--temp_dir") == "--temp-dir"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def _patch_run(self):
        return patch.object(cli_module, "run")

    def test_resolves_alias(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pdstools", "hc"])
        with self._patch_run() as mock_run:
            main()
        args, unknown = mock_run.call_args[0]
        assert args.app == "health_check"
        assert unknown == []

    def test_strips_run_subcommand(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pdstools", "run", "decision_analyzer"])
        with self._patch_run() as mock_run:
            main()
        args, _ = mock_run.call_args[0]
        assert args.app == "decision_analyzer"

    def test_typo_warning_printed(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pdstools", "health_check", "--data_path", "/x"])
        with self._patch_run():
            main()
        captured = capsys.readouterr()
        assert "Possible typo" in captured.err
        assert "--data-path" in captured.err

    def test_no_typo_no_warning(self, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            ["pdstools", "health_check", "--server.port", "8501"],
        )
        with self._patch_run():
            main()
        captured = capsys.readouterr()
        assert "Possible typo" not in captured.err

    def test_passes_unknown_to_run(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["pdstools", "decision_analyzer", "--server.port", "8501"],
        )
        with self._patch_run() as mock_run:
            main()
        _, unknown = mock_run.call_args[0]
        assert "--server.port" in unknown
        assert "8501" in unknown


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    defaults = {
        "app": "health_check",
        "data_path": None,
        "sample": None,
        "filter": None,
        "temp_dir": None,
        "full_embed": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestRunStreamlitMissing:
    def test_exits_when_streamlit_missing(self, capsys):
        # Hide streamlit.web.cli so the import fails
        with patch.dict(sys.modules, {"streamlit.web": None, "streamlit.web.cli": None}):
            with pytest.raises(SystemExit) as exc:
                run(_make_args(), [])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "streamlit is not installed" in captured.out


class TestRunEnvVarPropagation:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        for var in (
            "PDSTOOLS_DATA_PATH",
            "PDSTOOLS_SAMPLE_LIMIT",
            "PDSTOOLS_FILTER",
            "PDSTOOLS_TEMP_DIR",
            "PDSTOOLS_FULL_EMBED",
        ):
            monkeypatch.delenv(var, raising=False)

    def _run(self, args):
        # Mock streamlit so we never actually launch it
        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_module = MagicMock()
        fake_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_module},
        ):
            with patch.object(sys, "exit") as mock_exit:
                run(args, [])
        return mock_exit, fake_stcli

    def test_data_path_env_var(self):
        import os

        self._run(_make_args(data_path="/tmp/data.parquet"))
        assert os.environ["PDSTOOLS_DATA_PATH"] == "/tmp/data.parquet"

    def test_sample_env_var(self):
        import os

        self._run(_make_args(sample="50k"))
        assert os.environ["PDSTOOLS_SAMPLE_LIMIT"] == "50k"

    def test_temp_dir_env_var(self):
        import os

        self._run(_make_args(temp_dir="/tmp/work"))
        assert os.environ["PDSTOOLS_TEMP_DIR"] == "/tmp/work"

    def test_filter_env_var_is_json(self):
        import json
        import os

        self._run(_make_args(filter=["Channel=Web", "Score>=0.5"]))
        loaded = json.loads(os.environ["PDSTOOLS_FILTER"])
        assert loaded == ["Channel=Web", "Score>=0.5"]

    def test_full_embed_true_sets_env_var(self):
        import os

        self._run(_make_args(full_embed=True))
        assert os.environ.get("PDSTOOLS_FULL_EMBED") == "true"

    def test_full_embed_false_does_not_set_env_var(self):
        """When --full-embed is absent (default False), env var must not be set."""
        import os

        self._run(_make_args(full_embed=False))
        assert "PDSTOOLS_FULL_EMBED" not in os.environ

    def test_decision_analyzer_disables_xsrf(self):
        _, fake_stcli = self._run(_make_args(app="decision_analyzer"))
        argv = sys.argv
        assert "--server.enableXsrfProtection" in argv
        assert "false" in argv

    def test_default_max_upload_size_appended(self):
        self._run(_make_args(app="health_check"))
        assert "--server.maxUploadSize" in sys.argv
        assert "2000" in sys.argv

    def test_unknown_args_appended(self):
        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_module = MagicMock()
        fake_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_module},
        ):
            with patch.object(sys, "exit"):
                run(_make_args(app="health_check"), ["--server.port", "9000"])
        assert "--server.port" in sys.argv
        assert "9000" in sys.argv


class TestRunInteractivePrompt:
    def _run_with_input(self, user_inputs, app=None):
        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_module = MagicMock()
        fake_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_module},
        ):
            with patch("builtins.input", side_effect=user_inputs):
                with patch.object(sys, "exit"):
                    args = _make_args(app=app)
                    run(args, [])
                    return args

    def test_select_by_number(self):
        args = self._run_with_input(["1"])
        # First entry in APPS is health_check
        assert args.app == list(APPS.keys())[0]

    def test_select_by_internal_name(self):
        self._run_with_input(["decision_analyzer"])

    def test_select_by_alias(self):
        args = self._run_with_input(["da"])
        assert args.app == ALIASES["da"]

    def test_select_by_display_name(self):
        # "Adaptive Model Health Check"
        args = self._run_with_input([APPS["health_check"]["display_name"]])
        assert args.app == "health_check"

    def test_invalid_then_valid(self, capsys):
        args = self._run_with_input(["bogus", "0", "999", "hc"])
        assert args.app == "health_check"
        captured = capsys.readouterr()
        assert "Invalid choice" in captured.out

    def test_keyboard_interrupt_exits(self):
        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_module = MagicMock()
        fake_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_module},
        ):
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                with pytest.raises(SystemExit) as exc:
                    run(_make_args(app=None), [])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Questionary picker integration in run()
# ---------------------------------------------------------------------------


class TestRunQuestionaryPicker:
    """Picker integration on TTY stdin uses questionary; non-TTY falls back."""

    def test_falls_through_to_numeric_prompt_when_not_tty(self, monkeypatch):
        # Force isatty False; the existing input() prompt must be used.
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_module = MagicMock()
        fake_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_module},
        ):
            with patch("builtins.input", return_value="1"):
                with patch.object(sys, "exit"):
                    args = _make_args(app=None)
                    run(args, [])
        assert args.app == list(APPS.keys())[0]

    def test_questionary_choice_sets_app(self, monkeypatch):
        """When questionary returns a choice, run() launches that app."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        fake_questionary = MagicMock()
        fake_select = MagicMock()
        fake_select.ask.return_value = "decision_analyzer"
        fake_questionary.select.return_value = fake_select
        fake_questionary.Choice = MagicMock(side_effect=lambda title, value: value)

        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_st_module = MagicMock()
        fake_st_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {
                "questionary": fake_questionary,
                "streamlit": MagicMock(),
                "streamlit.web": fake_st_module,
            },
        ):
            with patch.object(sys, "exit"):
                args = _make_args(app=None)
                run(args, [])
        assert args.app == "decision_analyzer"

    def test_questionary_cancel_exits(self, monkeypatch):
        """When questionary returns None (Ctrl+C / Esc), CLI exits cleanly."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        fake_questionary = MagicMock()
        fake_select = MagicMock()
        fake_select.ask.return_value = None
        fake_questionary.select.return_value = fake_select
        fake_questionary.Choice = MagicMock(side_effect=lambda title, value: value)

        with patch.dict(sys.modules, {"questionary": fake_questionary}):
            with pytest.raises(SystemExit) as exc:
                run(_make_args(app=None), [])
        assert exc.value.code == 0

    def test_questionary_missing_falls_back_to_numeric(self, monkeypatch):
        """When questionary isn't installed, the numeric prompt runs."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        # Make `import questionary` raise ImportError.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "questionary":
                raise ImportError("simulated missing questionary")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_st_module = MagicMock()
        fake_st_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_st_module},
        ):
            with patch("builtins.input", return_value="1"):
                with patch.object(sys, "exit"):
                    args = _make_args(app=None)
                    run(args, [])
        assert args.app == list(APPS.keys())[0]


# ---------------------------------------------------------------------------
# --data-path validation
# ---------------------------------------------------------------------------


class TestDataPathValidation:
    def test_nonexistent_data_path_exits_2(self, capsys):
        args = _make_args(data_path="/nonexistent/path/should/never/exist.parquet")
        fake_stcli = MagicMock()
        fake_module = MagicMock()
        fake_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_module},
        ):
            with pytest.raises(SystemExit) as exc:
                run(args, [])
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_existing_data_path_passes(self, tmp_path):
        # Create a real file so the validation passes
        f = tmp_path / "data.parquet"
        f.write_text("dummy")
        fake_stcli = MagicMock()
        fake_stcli.main.return_value = 0
        fake_module = MagicMock()
        fake_module.cli = fake_stcli
        with patch.dict(
            sys.modules,
            {"streamlit": MagicMock(), "streamlit.web": fake_module},
        ):
            with patch.object(sys, "exit"):
                run(_make_args(data_path=str(f)), [])
        # If we got here, no SystemExit(2) was raised.


# ---------------------------------------------------------------------------
# --list flag and 'list' subcommand
# ---------------------------------------------------------------------------


class TestListApps:
    def test_list_flag_prints_apps(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pdstools", "--list"])
        with patch.object(cli_module, "run") as mock_run:
            main()
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        for key, info in APPS.items():
            assert key in captured.out
            assert info["display_name"] in captured.out
        # Tab-separated
        for line in captured.out.strip().splitlines():
            assert line.count("\t") == 2

    def test_list_subcommand(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pdstools", "list"])
        with patch.object(cli_module, "run") as mock_run:
            main()
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "health_check" in captured.out

    def test_aliases_in_list_output(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pdstools", "list"])
        with patch.object(cli_module, "run"):
            main()
        captured = capsys.readouterr()
        # 'hc' is the alias for health_check
        assert "hc" in captured.out


# ---------------------------------------------------------------------------
# doctor subcommand
# ---------------------------------------------------------------------------


class TestDoctor:
    def test_doctor_subcommand_runs(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pdstools", "doctor"])
        with patch.object(cli_module, "run") as mock_run:
            main()
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "--- Version info ---" in captured.out
        assert "pdstools:" in captured.out
        assert "--- Dependencies ---" in captured.out
        assert "--- Polars runtime ---" in captured.out
        assert "rt64 runtime active:" in captured.out
        assert "--- External tools ---" in captured.out
        # At least one of the documented dependency groups should render.
        assert "--- Dependency group:" in captured.out

    def test_doctor_renders_not_installed_when_which_returns_none(self, monkeypatch, capsys):
        from pdstools import cli as cli_mod

        # Force shutil.which to return None so both quarto and pandoc are
        # reported as 'not installed'.
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _name: None)
        cli_mod.doctor()
        captured = capsys.readouterr()
        assert "quarto: not installed" in captured.out
        assert "pandoc: not installed" in captured.out


# ---------------------------------------------------------------------------
# Subcommand routing
# ---------------------------------------------------------------------------


class TestSubcommandRouting:
    def test_run_subcommand_with_app(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pdstools", "run", "hc"])
        with patch.object(cli_module, "run") as mock_run:
            main()
        args, _ = mock_run.call_args[0]
        assert args.app == "health_check"

    def test_bare_app_still_works(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pdstools", "decision_analyzer"])
        with patch.object(cli_module, "run") as mock_run:
            main()
        args, _ = mock_run.call_args[0]
        assert args.app == "decision_analyzer"

    def test_bare_pdstools_routes_to_run(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pdstools"])
        with patch.object(cli_module, "run") as mock_run:
            main()
        args, _ = mock_run.call_args[0]
        assert args.app is None
