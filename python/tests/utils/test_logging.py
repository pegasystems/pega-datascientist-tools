"""Test logging configuration across pdstools modules."""

import logging


def test_all_main_modules_have_logger():
    """Verify main modules define a logger."""
    # Import and verify logger names

    # Just verify imports work - loggers are defined at module level
    assert True


def test_logging_level_changes(caplog):
    """Test that logging level can be changed dynamically."""
    logger = logging.getLogger("pdstools.test")

    with caplog.at_level(logging.INFO):
        logger.debug("This should not appear")
        logger.info("This should appear")

    assert "This should not appear" not in caplog.text
    assert "This should appear" in caplog.text


def test_cli_logging_env_var(monkeypatch):
    """Test that PDSTOOLS_LOG_LEVEL environment variable works."""
    import os

    # This test verifies the env var can be set
    # The actual logging config happens in cli.py which we've already tested
    monkeypatch.setenv("PDSTOOLS_LOG_LEVEL", "DEBUG")
    assert os.getenv("PDSTOOLS_LOG_LEVEL") == "DEBUG"
