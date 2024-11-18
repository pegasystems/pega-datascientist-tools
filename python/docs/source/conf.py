# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime

import plotly.io as pio

pio.renderers.default = "notebook_connected"

# -- Project information -----------------------------------------------------

project = "pdstools"
copyright = f"{datetime.now().year}, Pegasystems"
author = "Pegasystems"

# The full version, including alpha/beta/rc tags
release = "4.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ["sphinx.ext.autodoc", "sphinx_autodoc_typehints", "sphinx.ext.napoleon"]
extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinxarg.ext",
    # "sphinx_search.extension",
    # "jupyter_sphinx",
    # "sphinx_gallery.gen_gallery",
]

source_suffix = [".rst", ".md"]
intersphinx_mapping = {
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "python": ("https://docs.python.org/3", None),
}

add_module_names = False
toc_object_entries_show_parents = "hide"

# -- Autoapi settings --------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../pdstools"]
nbsphinx_allow_errors = True
autodoc_typehints = "both"

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
# html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

html_favicon = "../../../images/pegasystems-inc-vector-logo.svg"
html_logo = "../../../images/pegasystems-inc-vector-logo.svg"
html_title = "pdstools"

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/pegasystems/pega-datascientist-tools",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

suppress_warnings = ["spub.duplicated_toc_entry"]


# Overwriting nbsphinx in order to add remove_input cell tag (remove code cell,
# keep output)
import re  # noqa: E402

import nbsphinx  # noqa: E402
from nbsphinx import RST_TEMPLATE  # noqa: E402
from nbsphinx import setup as nbsphinx_setup  # noqa: E402

BLOCK_REGEX = r"(({{% block {block} -%}}\n)(.*?)({{% endblock {block} %}}\n))"
PATCH_TEMPLATE = r"{{% block {block} -%}}\n{patch}{{% endblock {block} %}}\n"


def search(block, template):
    pattern = BLOCK_REGEX.format(block=block)
    m = re.search(pattern, template, re.DOTALL)
    assert m is not None, f"Block {block} is not found"
    return m.group(3)


def patch(block, template, patch):
    pattern = BLOCK_REGEX.format(block=block)
    sub = PATCH_TEMPLATE.format(block=block, patch=patch)
    return re.sub(pattern, sub, template, flags=re.DOTALL)


def remove_block_on_tag(block, tags, template):
    content = search(block, RST_TEMPLATE)
    conditions = [f"{t!r} in cell.metadata.tags" for t in tags]
    content1 = f"""\
{{%- if {" or ".join(conditions)} -%}}
{{%- else -%}}
{content}
{{%- endif -%}}
"""
    return patch(block, template, content1)


RST_TEMPLATE = remove_block_on_tag(
    "input", ["remove_cell", "remove_input"], RST_TEMPLATE
)
RST_TEMPLATE = remove_block_on_tag(
    "nboutput", ["remove_cell", "remove_output"], RST_TEMPLATE
)
nbsphinx.RST_TEMPLATE = RST_TEMPLATE


# Exclude the logger from the documentation
def skip_member(app, what, name, obj, skip, options):
    if name == "logger":
        return True
    if name.startswith("_"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    # Call the nbsphinx setup function to ensure it is set up correctly
    nbsphinx_setup(app)


__all__ = ["setup"]
