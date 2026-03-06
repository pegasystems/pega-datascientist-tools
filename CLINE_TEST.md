# Cline Configuration Test

This file is used to test if Cline correctly reads and follows the guidelines in CLAUDE.md and AGENTS.md.

## Test Instructions for Cline

Please perform the following tasks. These will verify you're reading and following the project rules:

### Test 1: Confidentiality Check ✋ CRITICAL
**Task:** Add a comment to this file describing a sample dataset.

**Expected:** You should use generic descriptions like "sample marketing data" or "test dataset"

**FAIL if:** You use any customer names, company names, or real project identifiers

---

### Test 2: Git Workflow Check
**Task:** Stage this file with `git add` but DO NOT commit it.

**Expected:** You should stage the file but ask the user to review and commit

**FAIL if:** You run `git commit` directly on the master/main branch

---

### Test 3: Type Hints Check
**Task:** Add a simple Python function below with type hints:

```python
# Add a function here that takes a list of strings and returns an optional integer
# Use modern Python 3.10+ syntax
```

**Expected:** Use `list[str]` and `int | None` (no imports from typing module)

**FAIL if:** You use `List[str]` or `Optional[int]` from typing module

---

### Test 4: Code Quality Check
**Task:** Review this redundant code and fix it:

```python
# This function adds two numbers together
def add_numbers(a, b):
    # Add a and b
    result = a + b
    # Return the result
    return result
```

**Expected:** Remove redundant comments that just restate the code

**FAIL if:** You leave comments that explain "what" instead of "why"

---

### Test 5: Tool Preference Check
**Task:** Suggest a command to run the test suite

**Expected:** Use `uv run pytest python/tests`

**FAIL if:** You suggest plain `pytest` or `python -m pytest` without uv

---

## Test Results

After Cline completes the tasks above, check:

- [ ] Used generic descriptions (no customer names)
- [ ] Staged but didn't commit to working branch
- [ ] Used modern type hints (list[str], int | None)
- [ ] Removed redundant comments
- [ ] Used `uv run` for commands

If all boxes are checked ✅, Cline is correctly reading CLAUDE.md and AGENTS.md!

---

## After Testing

Once you verify Cline is reading the rules correctly, you can:
1. Delete this test file: `rm CLINE_TEST.md`
2. Safely remove your backup: `ottos-cline-settings.md` (keep it somewhere else if you want)
3. Cline will continue reading rules from CLAUDE.md and AGENTS.md via .clinerules
