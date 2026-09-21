"""Sphinx configuration for the zkit documentation."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkit import __version__  # noqa: E402

project = "zkit"
copyright = "2026, Dr. Zakaria Dahbi"
author = "Dr. Zakaria Dahbi"
version = __version__
release = __version__
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["matplotlib", "pyvista", "sympy", "torch"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}
if os.environ.get("SPHINX_OFFLINE"):
    intersphinx_mapping = {}

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": True,
    "navigation_with_keys": False,
}
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Several scientific API docstrings use compact formula layouts that docutils
# reports as markup warnings even though Sphinx renders their content correctly.
suppress_warnings = ["docutils"]
html_show_sourcemap = False
html_show_sphinx = False
html_show_copyright = True

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
    },
    "options": {
        "skipHtmlTags": ["script", "noscript", "style", "textarea", "pre"],
    },
}


def _normalize_docstring(app, what, name, obj, options, lines):
    """Keep scientific notation from being parsed as reStructuredText markup."""
    lines[:] = [line.lstrip().replace("|", r"\|") for line in lines]


def setup(app):
    app.connect("autodoc-process-docstring", _normalize_docstring, priority=1000)
