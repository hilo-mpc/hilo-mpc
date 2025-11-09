# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from datetime import datetime

import sphinx_theme

sys.path.insert(0, os.path.abspath('../..'))

# Auto-detect version from package
def get_version():
    try:
        from importlib.metadata import version
        return version('hilo_mpc')
    except ImportError:
        # Fallback for older Python versions
        try:
            import pkg_resources
            return pkg_resources.get_distribution('hilo_mpc').version
        except:
            pass
    except:
        pass
    # Final fallback - manual version
    return '1.2.0'

release = get_version()

# -- Project information -----------------------------------------------------

project = 'HILO-MPC'
# Auto-update copyright year
copyright = f'{datetime.now().year}, Johannes Pohlodek, Bruno Morabito, Rolf Findeisen'
author = 'Johannes Pohlodek, Bruno Morabito'

# The full version, including alpha/beta/rc tags
# release is set above from pyproject.toml

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    # 'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx.ext.mathjax',
]
# Reference file
bibtex_bibfiles = ['bibliography.bib']
bibtex_reference_style = 'author_year'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Removes type hints from signature
autodoc_typehints = 'none'  # 'description'

autodoc_member_order = 'groupwise'

# Mock heavy/optional imports so autodoc can build API pages on minimal environments (GitHub Pages)
autodoc_mock_imports = [
    'casadi', 'casadi.tools',
    'numpy', 'scipy', 'pandas', 'sklearn',
    'tensorflow', 'keras', 'torch', 'torchvision',
    'matplotlib', 'bokeh',
    'asyncua',
]

# Variable for project name
rst_epilog = f'.. |project_name| replace:: {project}'

# -- Options for Latex -------------------------------------------------------
# FIXME This does not see to work
latex_elements = {
    'preamble':
        r'''
        \newcommand{\tran}{{\mkern-1.5mu\mathsf{T}}}
        ''',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'alabaster'  # Using built-in theme for reliability
# html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]
html_logo = "images/hilo_logo_short_2.png"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # '_static' -> We don't have a folder '_static' at the moment

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
