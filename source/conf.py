import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TextKit-Learn"
copyright = "2025, Yasas Senarath"
author = "Yasas Senarath"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_title = "TextKit Learn"

# Add these to the extensions list
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google or NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.autosummary",  # For generating summary tables
    "sphinx_autodoc_typehints",  # Use type annotations
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

sys.path.insert(0, os.path.abspath(".."))
