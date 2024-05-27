# Configuration file for the Sphinx documentation builder.
import datetime as dt
import pathlib as plib
import sys
import os

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "bssp"
copyright = f"{dt.datetime.now().year}, BIG lab"
author = "BIG lab"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "external_links": [
        {"name": "GitHub", "url": "https://github.com/BIGlab/bssp"},
        {"name": "PyPI", "url": "https://pypi.org/project/bssp/"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BIGlab/bssp",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/bssp/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "BIGlab",  # Replace with your GitHub username or organization
    "github_repo": "bssp",  # Replace with your repository name
    "github_version": "main",  # Replace with your branch name
    "doc_path": "docs",  # Path in the repo to your docs root, e.g., "source/docs"
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.ico"

# -- Extension configuration -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
}

# -- Sphinx toggle options ---------------------------------------------------
todo_include_todos = True
autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
}