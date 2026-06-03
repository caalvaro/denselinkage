"""Sphinx configuration for the denselinkage documentation.

Build locally with ``make html`` (or ``make.bat html`` on Windows); both pass
``-W`` so a broken cross-reference, an orphan page, or a malformed directive
fails the build exactly as CI does. The API reference is generated from the
package's own docstrings via autodoc + autosummary — the docstrings are the
single source of truth, so the reference cannot drift from the contract.
"""

from importlib.metadata import PackageNotFoundError, version

# -- Project information ----------------------------------------------------

project = "denselinkage"
author = "Alvaro"
copyright = "2026, Alvaro"

try:
    release = version("denselinkage")
except PackageNotFoundError:  # pragma: no cover - editable/source builds
    release = "0.1.0"
version = release

# -- General configuration --------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # pull docstrings from the package
    "sphinx.ext.autosummary",  # generate per-object API pages
    "sphinx.ext.intersphinx",  # cross-link to python / numpy / pandas
    "sphinx.ext.viewcode",  # "[source]" links into the code
    "myst_parser",  # author narrative pages in Markdown
    "sphinx_copybutton",  # copy button on code blocks
    "sphinx_design",  # cards / grids on the landing page
    "sphinxcontrib.mermaid",  # text-based diagrams (rendered client-side)
]

templates_path = ["_templates"]
root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Everything under these directories is the in-repo design record (ADRs,
# development notes, the paper/slides). It is kept in the tree on purpose and is
# NOT part of the published site — `architecture.md` links to it on GitHub.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "ADRs/**",
    "development/**",
    "paper/**",
    "slides/**",
    "misc/**",
]

# -- HTML output ------------------------------------------------------------

html_theme = "furo"
html_title = f"denselinkage {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/caalvaro/denselinkage/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- autodoc / autosummary --------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"  # mirror the source / contract order
autodoc_typehints = "signature"
autodoc_typehints_format = "short"  # `DataFrame`, not `pandas.core.frame.DataFrame`
autodoc_default_options = {
    "show-inheritance": True
}  # show the port an adapter declares
python_use_unqualified_type_names = True

# -- MyST -------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "substitution"]
myst_heading_anchors = 3

# -- intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- sphinx-copybutton ------------------------------------------------------

# Strip REPL / shell prompts when copying so pasted snippets run as-is.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
