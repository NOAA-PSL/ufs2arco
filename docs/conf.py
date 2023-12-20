# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import datetime
sys.path.insert(0, os.path.abspath("../"))

project = 'ufs2arco'
copyright = f"2023-{datetime.datetime.now().year}, ufs2arco developers"
author = 'ufs2arco developers'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.napoleon",
        "nbsphinx",
        ]

numpydoc_show_class_members = False
napolean_google_docstring = True
napolean_numpy_docstring = False

templates_path = ['_templates']
exclude_patterns = []

napoleon_custom_sections = [("Returns", "params_style"),
                            ("Sets Attributes", "params_style"),
                            ("Assumptions", "notes_style"),
                            ("Required Fields in Config", "params_style"),
                            ("Optional Fields in Config", "params_style"),
                            ]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/NOAA-PSL/ufs2arco",
    "use_repository_button": True,
}
