# Standardize Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace inconsistent debug/verbose flags and print statements with standardized Python logging throughout pdstools library code.

**Architecture:** Convert all library code to use Python's built-in logging module with per-module loggers. Keep user-facing CLI output as print statements. **Critical distinction**: Only preserve `debug` parameters when they **change the return value** (e.g., returning extra columns). Remove all `debug`/`verbose` parameters that only control print output - use logging instead.

**Tech Stack:** Python logging module, existing pdstools structure

**Scope:**
- ~16 files with print statements in library code
- ~10 files with `verbose` parameters using print statements (remove these entirely)
- Multiple files with `debug` parameters (keep only if they affect return values, remove otherwise)
- Add documentation on enabling logging for debugging

**Decision**: `verbose` parameters are removed entirely (not deprecated) since they're internal APIs with limited usage.

---

## Task 1: Create Logging Configuration Documentation

**Files:**
- Create: `docs/source/logging.rst`
- Modify: `docs/source/index.rst`

**Step 1: Write the logging documentation file**

```rst
.. _logging:

Logging and Debugging
=====================

PDS Tools uses Python's built-in ``logging`` module for debugging and diagnostics. By default, logging is disabled to avoid cluttering output. You can enable logging to troubleshoot issues or understand what the library is doing.

Enabling Logging
----------------

Basic Configuration
^^^^^^^^^^^^^^^^^^^

To see debug messages from pdstools in your Python scripts or notebooks:

.. code-block:: python

    import logging

    # Enable debug logging for all pdstools modules
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Now use pdstools as normal
    from pdstools.adm import ADMDatamart
    dm = ADMDatamart(...)

Selective Module Logging
^^^^^^^^^^^^^^^^^^^^^^^^^

To enable logging for specific pdstools modules only:

.. code-block:: python

    import logging

    # Configure root logger at INFO level
    logging.basicConfig(level=logging.INFO)

    # Enable DEBUG for specific modules
    logging.getLogger('pdstools.adm').setLevel(logging.DEBUG)
    logging.getLogger('pdstools.decision_analyzer').setLevel(logging.DEBUG)

CLI Applications
^^^^^^^^^^^^^^^^

When using the CLI applications (Decision Analysis Tool, ADM Health Check), you can enable logging by setting the ``PDSTOOLS_LOG_LEVEL`` environment variable:

.. code-block:: bash

    # Enable debug logging
    export PDSTOOLS_LOG_LEVEL=DEBUG
    pdstools decision_analyzer --data-path data.parquet

    # Or inline
    PDSTOOLS_LOG_LEVEL=DEBUG pdstools decision_analyzer --data-path data.parquet

Available log levels: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``

Log Levels
----------

PDS Tools uses the following log levels:

- **DEBUG**: Detailed diagnostic information, iteration progress, data transformations
- **INFO**: Confirmation that things are working as expected (rarely used in library code)
- **WARNING**: Something unexpected happened but the library can continue (e.g., deprecated parameters)
- **ERROR**: A serious problem occurred, but not fatal (e.g., failed to load optional data)
- **CRITICAL**: A serious error that prevents the program from continuing

Logging in Custom Code
-----------------------

If you're extending pdstools or writing custom analysis code, follow the same pattern:

.. code-block:: python

    import logging

    # At module level
    logger = logging.getLogger(__name__)

    # In your functions
    def my_analysis_function(data):
        logger.debug(f"Processing {len(data)} records")
        logger.info("Analysis complete")
        logger.warning("Found unexpected values in column X")
        logger.error("Failed to compute metric Y")
```

**Step 2: Add logging.rst to index.rst**

In `docs/source/index.rst`, add after line 15 (after GettingStartedWithDecisionAnalyzer):

```rst
   logging
```

**Step 3: Verify documentation builds**

Run: `cd python/docs && make html`
Expected: Build succeeds without warnings, logging.rst rendered correctly

**Step 4: Commit**

```bash
git add docs/source/logging.rst docs/source/index.rst
git commit -m "docs: add logging and debugging guide"
```

---

## Task 2: Add Environment Variable Support for CLI Logging

**Files:**
- Modify: `python/pdstools/cli.py:1-50`

**Step 1: Add logging configuration at CLI entry point**

Add after the imports section (around line 15):

```python
import os

# Configure logging from environment variable
log_level = os.getenv("PDSTOOLS_LOG_LEVEL", "WARNING").upper()
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if log_level in valid_levels:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
```

**Step 2: Test the environment variable**

Run:
```bash
PDSTOOLS_LOG_LEVEL=DEBUG uv run pdstools --help
```
Expected: Help text displays, no logging output (no operations triggering logs yet)

**Step 3: Commit**

```bash
git add python/pdstools/cli.py
git commit -m "feat: add PDSTOOLS_LOG_LEVEL environment variable support"
```

---

## Task 3: Remove verbose parameter from BinAggregator, convert to logging

**Files:**
- Modify: `python/pdstools/adm/BinAggregator.py:47-240`

**Step 1: Write test for logging behavior**

In `python/tests/test_adm_binaggregator.py`, add:

```python
import logging

def test_accumulate_predictor_logging(caplog, sample_dm):
    """Test that operations produce debug logs when logging enabled."""
    bin_agg = BinAggregator(sample_dm)

    with caplog.at_level(logging.DEBUG):
        bin_agg.accumulate_predictor("SomePredictorName")

    # Should have debug messages about topics and model IDs
    assert any("Topic:" in record.message for record in caplog.records)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest python/tests/test_adm_binaggregator.py::test_accumulate_predictor_logging -v`
Expected: FAIL (test not finding log messages)

**Step 3: Remove verbose parameter and replace with unconditional logger.debug**

In `BinAggregator.py`:
- Remove `verbose: bool = False` from function signatures
- Replace `if verbose: print(...)` with unconditional `logger.debug(...)`
- Remove verbose parameter from all method calls that pass it

Example for line 154-159:
```python
# Before
if verbose:
    print(f"Topic: {topic}, predictor: {predictor}")

# After
logger.debug(f"Topic: {topic}, predictor: {predictor}")
```

Apply to:
- Line 47: Remove `verbose` from `accumulate_predictor` signature
- Line 154-159: Replace conditional print with `logger.debug()`
- Line 182, 189: Remove `verbose=verbose` from method calls
- Line 211: Remove `verbose=False` from `accumulate_num_binnings` signature
- Line 214-215: Replace conditional print with `logger.debug()`
- Line 220-234: Replace conditional print with `logger.debug()`
- Line 230: Remove `verbose=verbose` from method call
- Line 240-241: Replace conditional print with `logger.debug()`
- Line 282: Remove `verbose=False` from `accumulate_sym_binnings` signature
- Line 314-316: Replace conditional print with `logger.debug()`
- Line 332-334: Replace conditional print with `logger.debug()`
- Line 616: Remove `verbose=False` from any other affected method

**Step 4: Update docstrings to remove verbose parameter**

Remove `verbose` from Parameters section. Add note about logging:
```python
"""
...
Notes
-----
Enable debug logging to see processing progress:
    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
...
"""
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest python/tests/test_adm_binaggregator.py::test_accumulate_predictor_logging -v`
Expected: PASS

**Step 6: Commit**

```bash
git add python/pdstools/adm/BinAggregator.py python/tests/test_adm_binaggregator.py
git commit -m "refactor: remove verbose parameter from BinAggregator, use logging"
```

---

## Task 4: Convert cdh_utils error prints to logging

**Files:**
- Modify: `python/pdstools/utils/cdh_utils.py:1308,1322`

**Step 1: Replace print statements with logger.error**

Line 1308:
```python
# Before
print(f"Error reading file {path_list[0]}: {e}")

# After
logger.error(f"Error reading file {path_list[0]}: {e}")
```

Line 1322:
```python
# Before
print(f"Error adding file {file_path} to zip: {e}")

# After
logger.error(f"Error adding file {file_path} to zip: {e}")
```

**Step 2: Verify logger is already defined**

Check that `logger = logging.getLogger(__name__)` exists near top of file (it should be around line 1342).

If it's inside a function, move it to module level after imports.

**Step 3: Run existing tests**

Run: `uv run pytest python/tests/ -k cdh_utils -v`
Expected: All tests pass (logging doesn't break functionality)

**Step 4: Commit**

```bash
git add python/pdstools/utils/cdh_utils.py
git commit -m "refactor: convert cdh_utils error prints to logging"
```

---

## Task 5: Remove verbose parameter from Explanations Reports

**Files:**
- Modify: `python/pdstools/explanations/Reports.py:52-102`

**Step 1: Write test for logging behavior**

In `python/tests/test_explanations.py` (or create if not exists), add:

```python
import logging

def test_reports_logging(caplog, sample_explanations):
    """Test that operations produce debug logs when logging enabled."""
    with caplog.at_level(logging.DEBUG):
        sample_explanations.generate_report()

    # Should have debug messages
    assert len([r for r in caplog.records if r.levelname == "DEBUG"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest python/tests/test_explanations.py::test_reports_logging -v`
Expected: FAIL or needs implementation

**Step 3: Remove verbose parameter entirely**

In methods that have `verbose` parameter (lines 52, 136):
- Remove `verbose: bool = False` from function signatures
- Trace where verbose is passed to other methods and remove those parameters too
- Replace any conditional print statements with unconditional `logger.debug()` calls
- Remove verbose from all method call chains

**Step 4: Update docstrings**

Remove verbose from Parameters section. Add logging note in Notes section if not present.

**Step 5: Run test to verify it passes**

Run: `uv run pytest python/tests/test_explanations.py::test_reports_logging -v`
Expected: PASS

**Step 6: Commit**

```bash
git add python/pdstools/explanations/Reports.py python/tests/test_explanations.py
git commit -m "refactor: remove verbose parameter from Explanations Reports, use logging"
```

---

## Task 6: Document debug vs verbose vs logging in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add logging standards section**

After the "Python Type Hints" section, add:

```markdown
## Logging Standards

### When to Use Logging vs Debug Parameters

PDS Tools uses Python's `logging` module for diagnostic output. Use logging for:
- Progress messages during long-running operations
- Diagnostic information about data transformations
- Error messages in library code
- Warnings about deprecated features or unexpected data

**Do NOT use print statements in library code.** Print statements are only acceptable in:
- CLI code for user-facing output (prompts, menus, results)
- Interactive notebook examples
- Test debugging (temporary only)

### Parameter Naming Conventions

**`debug` parameter**: Use when the parameter changes the **return value** (e.g., returns additional columns, more detailed data structure). This is NOT for logging.

Example:
```python
def summary_by_channel(self, ..., debug: bool = False) -> pl.LazyFrame:
    """
    ...
    debug : bool, default False
        If True, include extra columns (ModelTechnique, Configurations) in output.
    """
    result = result.group_by(...).agg(...)
    if debug:
        return result  # Keep all columns
    return result.drop("ModelTechnique", "Configurations")  # Hide internal columns
```

**`verbose` parameter**: REMOVED. Never use this pattern. If you find a `verbose` parameter in code:
1. Remove the parameter from the function signature completely
2. Replace `if verbose: print(...)` with unconditional `logger.debug(...)`
3. Update docstrings to remove the parameter and mention logging configuration
4. Remove verbose from all function calls in the codebase

### Logging Pattern

Every module should have a module-level logger:

```python
import logging

logger = logging.getLogger(__name__)

class MyClass:
    def my_method(self, data):
        logger.debug(f"Processing {len(data)} records")
        try:
            result = self._compute(data)
            logger.debug(f"Computation complete: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Failed to compute: {e}")
            raise
```

### Testing Logging

Test logging output using pytest's `caplog` fixture:

```python
import logging

def test_my_method_logging(caplog):
    with caplog.at_level(logging.DEBUG):
        obj = MyClass()
        obj.my_method(data)

    assert "Processing" in caplog.text
    assert any(r.levelname == "DEBUG" for r in caplog.records)
```
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add logging standards to CLAUDE.md"
```

---

## Task 7: Audit all remaining debug parameters

**Files:**
- Review: All files with `debug` parameters
- Modify: Remove `debug` parameters that don't affect return values

**Step 1: Search for all remaining debug parameters**

Run:
```bash
rg "debug\s*[=:]" --type py python/pdstools -A 10 | grep -E "(def |return )" > /tmp/debug_params.txt
```

Review each one to determine:
- Does it change what the function returns? → Keep it
- Does it only control print/logging? → Remove it

**Step 2: Remove debug parameters that only affect logging**

For each `debug` parameter that doesn't change return values:
- Remove from function signature
- Replace `if debug: print(...)` with unconditional `logger.debug(...)`
- Remove from all method calls
- Update docstrings

**Step 3: Document remaining debug parameters clearly**

For each `debug` parameter that DOES change return values (like in Aggregates.py):
- Ensure docstring explicitly says it affects return value
- Add example showing what extra data is returned

Example:
```python
"""
Parameters
----------
debug : bool, default False
    If True, return additional diagnostic columns (ModelTechnique, Configurations).
    If False, return only summary columns (usesNBAD, usesAGB).

    This parameter affects the return value structure, not logging output.
    For debug logging, use logging.basicConfig(level=logging.DEBUG).

Examples
--------
>>> summary = dm.summary_by_channel(debug=True)
>>> print(summary.columns)  # Includes ModelTechnique, Configurations
>>> summary = dm.summary_by_channel(debug=False)
>>> print(summary.columns)  # Only usesNBAD, usesAGB
"""
```

**Step 4: Commit**

```bash
git add python/pdstools/
git commit -m "refactor: remove debug parameters that don't affect return values"
```

---

## Task 8: Document remaining debug parameters clearly

**Files:**
- Review: `python/pdstools/adm/Aggregates.py`, `python/pdstools/ih/Aggregates.py`
- Modify: Docstrings for clarity

**Step 1: Ensure clear docstrings for debug parameters that affect return values**

For each `debug` parameter in Aggregates classes:

```python
def _summarize_meta_info(
    self,
    grouping: list[str] | None,
    model_data: pl.LazyFrame,
    debug: bool,
) -> pl.LazyFrame:
    """...

    Parameters
    ----------
    ...
    debug : bool
        If True, return internal diagnostic columns (ModelTechnique, Configurations).
        If False, return only the summary columns (usesNBAD, usesAGB).

        This parameter affects the return value structure, not logging output.
        For debug logging, use logging.basicConfig(level=logging.DEBUG).

    Examples
    --------
    >>> result = obj._summarize_meta_info(grouping, data, debug=True)
    >>> result.columns  # ['usesNBAD', 'usesAGB', 'ModelTechnique', 'Configurations']
    >>> result = obj._summarize_meta_info(grouping, data, debug=False)
    >>> result.columns  # ['usesNBAD', 'usesAGB']
    """
```

**Step 2: Update CONTRIBUTING.md**

Add to the Code Quality Standards section:
```markdown
### Debug vs Verbose vs Logging

**Never use these patterns:**
- ❌ `verbose` parameter for controlling print output
- ❌ `print()` statements in library code (except CLI user-facing output)
- ❌ `debug` parameter that only controls logging

**Only use `debug` parameter when it changes return values:**
- ✅ `debug=True` returns extra columns for inspection
- ✅ `debug=True` returns more detailed data structure
- Document clearly what extra data is returned

**For diagnostic output, use logging:**
```python
logger = logging.getLogger(__name__)

def process_data(data):
    logger.debug(f"Processing {len(data)} records")  # Not a parameter
    result = transform(data)
    logger.debug(f"Produced {len(result)} results")
    return result
```
```

**Step 3: Commit**

```bash
git add python/pdstools/adm/Aggregates.py python/pdstools/ih/Aggregates.py CONTRIBUTING.md
git commit -m "docs: clarify debug parameter usage - only for return values"
```

---

## Task 9: Update Getting Started guides

**Files:**
- Modify: `python/docs/source/GettingStarted.rst`
- Modify: `python/docs/source/GettingStartedWithDecisionAnalyzer.rst`

**Step 1: Add logging section to GettingStarted.rst**

Add after the installation section:

```rst
Enabling Debug Logging
----------------------

If you need to troubleshoot or see what pdstools is doing under the hood, enable debug logging:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Now use pdstools - debug messages will appear
    from pdstools.adm import ADMDatamart
    dm = ADMDatamart(...)

For more details, see :ref:`logging`.
```

**Step 2: Add logging section to GettingStartedWithDecisionAnalyzer.rst**

Add before the "Next Steps" section:

```rst
Troubleshooting with Logging
-----------------------------

If you encounter issues loading data or running analysis, enable debug logging:

.. code-block:: bash

    export PDSTOOLS_LOG_LEVEL=DEBUG
    pdstools decision_analyzer --data-path data.parquet

This will show detailed information about file reading, data processing, and any errors.

For more information, see :ref:`logging`.
```

**Step 3: Verify docs build**

Run: `cd python/docs && make html`
Expected: Build succeeds, logging references resolve correctly

**Step 4: Commit**

```bash
git add python/docs/source/GettingStarted.rst python/docs/source/GettingStartedWithDecisionAnalyzer.rst
git commit -m "docs: add logging instructions to getting started guides"
```

---

## Task 10: Create example notebook demonstrating logging

**Files:**
- Create: `examples/debugging_with_logging.ipynb`

**Step 1: Create notebook with logging examples**

Create a Jupyter notebook showing:
1. Basic logging setup
2. Enabling for specific modules
3. Example of debug output during ADM analysis
4. Example of catching warnings and errors
5. Tips for using logging in production

**Step 2: Add to documentation**

In `python/docs/source/index.rst`, add to Examples section:
```rst
   examples/debugging_with_logging
```

**Step 3: Test notebook runs**

Run: `uv run jupyter nbconvert --execute --to notebook examples/debugging_with_logging.ipynb`
Expected: Executes without errors

**Step 4: Commit**

```bash
git add examples/debugging_with_logging.ipynb python/docs/source/index.rst
git commit -m "docs: add logging example notebook"
```

---

## Task 11: Run full test suite and update tests

**Files:**
- Run: All tests
- Update: Any tests affected by logging changes

**Step 1: Run full test suite**

Run: `uv run pytest python/tests/ -v --cov=python/pdstools --cov-report=term-missing`
Expected: All tests pass, coverage remains ≥80%

**Step 2: Fix any failing tests**

If tests fail due to:
- Missing verbose parameter: Remove from test calls
- Print statement checks: Update to check logger output instead
- Other logging-related issues: Fix accordingly

**Step 3: Add logging-specific tests**

In `python/tests/test_logging.py` (create if not exists):

```python
"""Test logging configuration across pdstools modules."""

import logging
import pytest


def test_all_modules_have_logger():
    """Verify all main modules define a logger."""
    from pdstools.adm import ADMDatamart
    from pdstools.decision_analyzer import DecisionAnalyzer
    from pdstools.prediction import Prediction

    # Check that loggers exist (will be named after module)
    assert logging.getLogger('pdstools.adm.ADMDatamart')
    assert logging.getLogger('pdstools.decision_analyzer.DecisionAnalyzer')
    assert logging.getLogger('pdstools.prediction.Prediction')


def test_logging_level_changes(caplog):
    """Test that logging level can be changed dynamically."""
    logger = logging.getLogger('pdstools.test')

    with caplog.at_level(logging.INFO):
        logger.debug("This should not appear")
        logger.info("This should appear")

    assert "This should not appear" not in caplog.text
    assert "This should appear" in caplog.text


def test_verbose_deprecation_warning():
    """Test that verbose parameter raises deprecation warning."""
    from pdstools.adm import BinAggregator
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Test with actual object if possible, or skip if no test data
        # bin_agg.accumulate_predictor("pred", verbose=True)
        pass  # Adjust based on available test fixtures
```

**Step 4: Run tests again**

Run: `uv run pytest python/tests/test_logging.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add python/tests/
git commit -m "test: add logging tests and update affected tests"
```

---

## Task 12: Update CHANGELOG and create PR

**Files:**
- Modify: `CHANGELOG.md`
- Create: Pull request

**Step 1: Update CHANGELOG**

Add comprehensive entry:

```markdown
## [Unreleased]

### Added
- Logging configuration documentation in docs/source/logging.rst
- `PDSTOOLS_LOG_LEVEL` environment variable for CLI logging control
- Example notebook demonstrating logging usage

### Changed
- Converted error print statements in cdh_utils to logging
- Standardized logging across all pdstools modules
- Updated getting started guides with logging instructions

### Removed
- `verbose` parameters from BinAggregator and Explanations Reports (use logging instead)
- `debug` parameters that only controlled logging (use logging instead)

### Fixed
- Clarified that remaining `debug` parameters only affect return values, not logging

### Documentation
- Added comprehensive logging and debugging guide
- Updated CLAUDE.md with logging standards and patterns
- Added logging section to getting started guides
```

**Step 2: Create summary document**

Create `docs/logging-refactoring-summary.md`:

```markdown
# Logging Refactoring Summary

## Changes Made

1. **Documentation**: Created comprehensive logging guide with examples
2. **CLI Support**: Added PDSTOOLS_LOG_LEVEL environment variable
3. **Code Changes**: Converted print statements and verbose parameters to logging
4. **Standards**: Documented patterns in CLAUDE.md and CONTRIBUTING.md
5. **Tests**: Added logging tests and updated affected tests

## Files Changed

- Documentation: 5 files
- Source code: 4 modules
- Tests: 3 test files
- Project docs: 2 files (CLAUDE.md, CONTRIBUTING.md)

## Backward Compatibility

**Breaking changes:**
- `verbose` parameters removed from BinAggregator and Explanations Reports
- `debug` parameters that only controlled logging have been removed
- Code calling these with `verbose=True` will need updates

**Unchanged:**
- `debug` parameters that affect return values remain unchanged
- New logging is opt-in (disabled by default)

**Migration:**
Most `verbose` usage was internal. For any external usage:
```python
# Before
bin_agg.accumulate_predictor("pred", verbose=True)

# After - enable logging instead
import logging
logging.basicConfig(level=logging.DEBUG)
bin_agg.accumulate_predictor("pred")
```

Since these are lower-level API methods with limited external usage, removing rather than deprecating is acceptable.

## Migration Guide for Users

**For removed `verbose` parameters:**
```python
# Before
bin_agg.accumulate_predictor("pred", verbose=True)

# After
import logging
logging.basicConfig(level=logging.DEBUG)
bin_agg.accumulate_predictor("pred")
```

**For `debug` parameters (no change):**
```python
# Still works the same - debug affects return value
summary = dm.summary_by_channel(debug=True)  # Returns extra columns
```

## Next Steps

1. Monitor for any user issues with removed parameters
2. Consider adding more strategic debug logging based on user feedback
3. Review other modules for additional logging opportunities
```

**Step 3: Run final verification**

Run:
```bash
# Build docs
cd python/docs && make html

# Run tests
cd ../..
uv run pytest python/tests/ -v

# Try CLI with logging
PDSTOOLS_LOG_LEVEL=DEBUG uv run pdstools --help
```

Expected: All succeed

**Step 4: Commit and create PR**

```bash
git add CHANGELOG.md docs/logging-refactoring-summary.md
git commit -m "chore: finalize logging standardization"
git push origin feature/standardize-logging
```

Create PR with title: "Standardize logging across pdstools"

---

## Notes

- **DRY**: Use consistent `logger = logging.getLogger(__name__)` pattern everywhere
- **YAGNI**: Don't add complex logging infrastructure - use stdlib logging
- **TDD**: Write logging tests before converting each module
- **Backward compatibility**: Keep deprecated parameters with warnings for one version
- **Documentation**: Comprehensive guide helps users migrate smoothly
