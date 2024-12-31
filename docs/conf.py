# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import sys
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version

sys.path.append(str((Path(__file__).parent / "_ext").resolve()))

project = "LensKit"
copyright = "2018–2024 Drexel University, Boise State University, and collaborators"
author = "Michael D. Ekstrand"

release = version("lenskit")
version = ".".join(release.split(".")[:3])
_parsed_ver = Version(release)

extensions = [
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx_togglebutton",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "lk_stability",
]

# set up our filenames
# source_suffix = {".rst": "restructuredtext"}
exclude_patterns = [
    "_build",
    "_ext",
    "Thumbs.db",
    ".DS_Store",
    "old/*",
]
nb_execution_mode = "off"

# layout and setup options
pygments_style = "sphinx"
highlight_language = "python3"

html_theme = "pydata_sphinx_theme"
html_logo = "LKLogo2.png"
if _parsed_ver.is_devrelease:
    html_baseurl = "https://lkpy.lenskit.org/en/latest/"
else:
    html_baseurl = "https://lkpy.lenskit.org/en/stable/"
html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "switcher": {
        "json_url": "https://lkpy.lenskit.org/versions.json",
        "version_match": "2024.0dev",
    },
    "show_version_warning_banner": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/lenskit/lkpy",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Mastodon",
            "url": "https://recsys.social/@LensKit",
            "icon": "fa-brands fa-mastodon",
            "type": "fontawesome",
        },
    ],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # "github_user": "lenskit",
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

# how do we want to set up documentation?
autodoc_default_options = {"members": True, "member-order": "bysource", "show-inheritance": True}
autodoc_typehints = "description"
autodoc_type_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
    "SeedLike": "lenskit.types.SeedLike",
    "RNGLike": "lenskit.types.RNGLike",
    "RNGInput": "lenskit.types.RNGInput",
    "IDSequence": "lenskit.data.types.IDSequence",
}
# autosummary_generate_overwrite = False
autosummary_imported_members = False
autosummary_ignore_module_all = True

# customize doc parsing
napoleon_custom_sections = [("Stability", "returns_style")]

nitpicky = True
todo_include_todos = True

# Cross-linking and external references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "structlog": ("https://www.structlog.org/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "implicit": ("https://benfred.github.io/implicit/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

bibtex_bibfiles = ["lenskit.bib"]
nb_execution_mode = "off"

mermaid_d3_zoom = True

# -- external links

extlinks = {
    "issue": ("https://github.com/lenskit/lkpy/issues/%s", "🐞 %s"),
    "pr": ("https://github.com/lenskit/lkpy/pull/%s", "⛙ %s"),
    "user": ("https://github.com/%s", "@%s"),
}

bibtex_bibfiles = ["lenskit.bib"]
