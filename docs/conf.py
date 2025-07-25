import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IViS'
html_title = 'IViS – ivis documentation'
copyright = '2025, Antoine Marchal'
author = 'Antoine Marchal'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",             # ← enables notebook support
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

html_theme = 'furo'  # or 'sphinx_rtd_theme'
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#e95420",
        "color-brand-content": "#ff6600",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff6600",
        "color-brand-content": "#ffa266",
    },
}

html_logo = "_static/ivis_logo.png"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_show_sourcelink = False

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

autodoc_mock_imports = [
    "torch",
    "casacore",
    "casatools",
    "casatasks",
    "reproject",
    "matplotlib",
    "marchalib",
    "pytorch_finufft",
    "dask",
    "daskms",
    "radio_beam",
    "deconv",
    "psutil",
    "tqdm",
    "joblib"
]
