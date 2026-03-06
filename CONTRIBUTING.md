# Contribution Guide
We love feedback and contributions. Thank you for helping improve Pega Data Scientist Tools!

## Getting Help
- **Bugs / Feature Requests:** Open an [Issue](https://github.com/pegasystems/pega-datascientist-tools/issues).
- **Questions / Discussions:** Use the [Discussions](https://github.com/pegasystems/pega-datascientist-tools/discussions) tab.


## Contributing workflow
1. Fork this repository (creating a fork is only necessary when you are not part of the Pega organization)
2. Clone your fork.
3. Create a new branch for your change (`git checkout -b amazing-feature`).
4. Commit your work with clear messages (`git commit -m 'Added an amazing feature'`).
5. Push to the branch (`git push origin amazing-feature`)
6. Open a Pull Request (PR).


## Testing
Pull requests trigger continuous integration via [GitHub Actions](https://docs.github.com/en/actions). These workflows automatically run tests and check on your changes. You can also enable Actions in your fork to test before submitting. To manually and locally run the tests with pytest, use either your IDE or the command line:

```bash
[uv run] pytest python/tests
```

To test with code coverage analysis run the following (make sure to have installed the `codecov` and `pytest-codecov` packages). Code coverage analysis is part of the GitHub continuous integration too.

```bash
[uv run] pytest python/tests --cov=./python/pdstools --cov-report=xml
```

## Documentation and articles
- The Python documentation uses `Sphinx` to generate the docs, `nbsphinx` to convert the jupyter notebooks to markdown, and `Furo` as the Sphinx template. These dependencies can be installed (from `python/docs`) with `[uv] pip install -r docs-requirements.txt`. `Pandoc` is a requirement too and needs to be installed separately.
- A Makefile is provided to create the documentation. The docs are automatically generated on any push commits in the master branch, but you can build them locally by following the steps below

### Create documentation locally
To generate a new version of the docs, simply navigate to the top-level folder and run the following command:
```sh
cd python/docs && [uv run] make html
```

### Edit or Add new content
To edit or add content, make changes / new files in the examples folder. The articles should be jupyter notebook files, and **should be empty, not pre-run**.

1. Create an article somewhere in the top-level `examples` folder. These articles should be jupyter notebook files, and should be **empty, not pre-run**.
2. Edit the cp command in `python/docs/Makefile` where the articles are copied to a temporary to include your article.
3. Update `python/docs/source/index.rst` and add your entry to the included notebooks. Since they will be moved to the `articles` folder, simply enter the name of the notebook file, without extension.
4. Be aware that the H1 headers of your notebook will be used as the entries in the doc index. If you have multiple H1's, you get multiple entries.

For example, if we added an article called `Example.ipynb` in the `examples/helloworld` folder, we would add `../../examples/helloworld/Example.ipynb` to the cp line in the Makefile (right before `source/articles`, of course), and add `articles/Example` to the `python/docs/source/index.rst` file. To test that it works, we recommend building the docs (`make html`) locally before creating a pull request.

## Style Guide

### Product Naming Conventions

When referring to pdstools applications, use consistent naming:

| Tool | User-facing name | Code/CLI reference |
|------|------------------|-------------------|
| Decision Analyzer | Decision Analysis Tool | `decision_analyzer` |
| ADM Health Check | ADM Health Check | `adm_healthcheck` |

**Rules:**
- **Documentation, UI headers, help text** → Use user-facing name
- **Code, CLI commands, file paths** → Use code reference
- Be consistent within each context

**Examples:**

✅ **Good:**
```markdown
# Documentation
The Decision Analysis Tool provides comprehensive insights...

# Code
from pdstools.decision_analyzer import DecisionAnalyzer
```

❌ **Avoid:**
```markdown
# Don't mix contexts
Run the decision_analyzer tool to analyze decisions.
# Should be: "Run the Decision Analysis Tool..."
```

### Version and Upgrade Text

**Do:**
- Show current version number
- Show "Keep up to date" message
- Let dynamic version checks show specific upgrade commands when needed

**Don't:**
- Include specific install commands in static UI headers
- Assume user's package manager (uv vs pip vs conda)

**Examples:**

✅ **Good:**
```python
st.caption(f"pdstools {version} · Keep up to date")

# Dynamic warning shown when update available:
if newer_version_available:
    st.warning(f"A newer version is available ({latest}). "
               "Run `uv pip install --upgrade pdstools` to update.")
```

❌ **Avoid:**
```python
# Static header with hardcoded command
st.caption(f"pdstools {version} · Keep up to date: `uv pip install --upgrade pdstools`")
```

### Documentation Updates

When updating analysis documentation:
- Keep analysis lists focused on key features
- Match current page structure
- Update descriptions when page names change
- Spell-check all user-facing text
