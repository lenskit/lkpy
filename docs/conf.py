# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.abspath(".."))

import sphinx_rtd_theme  # noqa: F401

# -- Project information -----------------------------------------------------

project = "LensKit"
copyright = "2018–2024 Drexel University, Boise State University, and collaborators"
author = "Michael D. Ekstrand"

release = version("lenskit")
version = ".".join(release.split(".")[:3])


extensions = [
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
highlight_language = "python3"


html_theme = "sphinx_rtd_theme"
html_theme_options = {
    # 'github_user': 'lenskit',
    # 'github_repo': 'lkpy',
    # 'travis_button': False,
    # 'canonical_url': 'https://lkpy.lenskit.org/',
    # 'font_family': 'Charter, serif'
    # 'font_family': '"Source Sans Pro", "Georgia Pro", Georgia, serif',
    # 'font_size': '15px',
    # 'head_font_family': '"Merriweather Sans", "Arial", sans-serif',
    # 'code_font_size': '1em',
    # 'code_font_family': '"Source Code Pro", "Consolas", "Menlo", sans-serif'
}

templates_path = ["_templates"]
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "LensKitdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "scikit": ("https://scikit-learn.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "seedbank": ("https://seedbank.lenskit.org/en/latest/", None),
    "progress_api": ("https://progress-api.readthedocs.io/en/latest/", None),
    "manylog": ("https://manylog.readthedocs.io/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "implicit": ("https://implicit.readthedocs.io/en/latest/", None),
}

autodoc_default_options = {"members": True, "member-order": "bysource", "show-inheritance": True}
autodoc_typehints = "description"

bibtex_bibfiles = ["lenskit.bib"]
jupyter_execute_notebooks = "off"

# -- external links

extlinks = {
    "issue": ("https://github.com/lenskit/lkpy/issues/%s", "🐞 %s"),
    "pr": ("https://github.com/lenskit/lkpy/pull/%s", "⛙ %s"),
    "user": ("https://github.com/%s", "@%s"),
}

# -- Module Canonicalization ------------------------------------------------

# cleanups = {
#     'lenskit': ['Algorithm', 'Recommender', 'Predictor', 'CandidateSelector']
# }

# for module, objects in cleanups.items():
#     mod = import_module(module)
#     for name in objects:
#         obj = getattr(mod, name)
#         obj.__module__ = module
