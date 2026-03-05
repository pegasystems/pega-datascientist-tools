# CLI Version Flag Design

**Date:** 2026-03-05
**Status:** Approved

## Problem Statement

The pdstools CLI lacks a `--version` flag, making it difficult for users to quickly check which version they have installed. Users need a standard way to verify their pdstools version without launching an app or inspecting package metadata.

## Context

- Current version is stored in `pdstools.__version__` (4.6.0)
- The CLI is defined in `python/pdstools/cli.py` with entry point in `pyproject.toml`
- Interactive app selection prompt already exists and works correctly
- The issue about "ancient version" is a separate problem (outdated global installation)

## Requirements

1. Add `--version` flag that prints `pdstools {version}` and exits
2. Use standard Python CLI conventions (argparse version action)
3. Version flag takes precedence over all other arguments
4. Minimal code changes - single file modification

## Design Decision

**Approach:** Use argparse's built-in `version` action

This is the standard Python convention for CLI version flags. It automatically:
- Prints the version string
- Exits with status 0
- Appears in `--help` output
- Takes precedence over other arguments

**Alternative approaches considered:**
- Manual sys.argv checking: More code, doesn't integrate with --help
- Custom Action class: Unnecessary complexity for simple version display

## Implementation

**File:** `python/pdstools/cli.py`

**Changes:**

1. Add import at top of file:
   ```python
   from pdstools import __version__
   ```

2. Add version argument in `create_parser()` (after line 41):
   ```python
   parser.add_argument(
       '--version',
       action='version',
       version=f'pdstools {__version__}'
   )
   ```

**No other files need modification.**

## Behavior Examples

```bash
# Print version and exit
$ pdstools --version
pdstools 4.6.0

# Version takes precedence
$ pdstools --version --data-path foo
pdstools 4.6.0

# Interactive prompt unchanged
$ pdstools
Available pdstools apps:
  1. Adaptive Model Health Check
  2. Decision Analysis
  3. Impact Analyzer

# Direct app launch unchanged
$ pdstools health_check
Running Adaptive Model Health Check app...
```

## Testing

After implementation:
1. Run `pdstools --version` and verify output format
2. Run `pdstools --version --other-flags` and verify it ignores other flags
3. Run `pdstools --help` and verify version appears in help text
4. Verify existing behavior (interactive prompt, app launching) is unchanged

## Impact

- Single file change (cli.py)
- No breaking changes
- No impact on existing functionality
- Follows Python CLI best practices
