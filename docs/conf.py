# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import doctest
import sys
from os import fspath
from pathlib import Path

from packaging.version import Version

from lenskit._version import lenskit_version

sys.path.append(str((Path(__file__).parent / "_ext").resolve()))

project = "LensKit"
copyright = "2018–2025 Drexel University, Boise State University, and collaborators"
author = "Michael D. Ekstrand"

release = lenskit_version()
_parsed_ver = Version(release)
version = _parsed_ver.base_version

extensions = [
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx_togglebutton",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_new_tab_link",
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
    html_baseurl = "https://lenskit.org/latest/"
else:
    html_baseurl = "https://lenskit.org/stable/"
html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "switcher": {
        "json_url": "https://lenskit.org/versions.json",
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
        {
            "name": "BlueSky",
            "url": "https://bsky.app/profile/lenskit.org",
            "icon": "fa-brands fa-bluesky",
            "type": "fontawesome",
        },
    ],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright", "version", "disclaimer", "counter"],
    "footer_end": [],
    # "github_user": "lenskit",
    # 'github_repo': 'lkpy',
    # 'travis_button': False,
    # 'canonical_url': 'https://lenskit.org/',
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
    "SeedLike": "lenskit.random.SeedLike",
    "RNGLike": "lenskit.random.RNGLike",
    "RNGInput": "lenskit.random.RNGInput",
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
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "implicit": ("https://benfred.github.io/implicit/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
}

bibtex_bibfiles = ["lenskit.bib"]
nb_execution_mode = "off"
doctest_path = [fspath((Path(__file__).parent / "guide" / "examples").resolve())]
doctest_default_flags = (
    doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE
)

mermaid_d3_zoom = True

# -- external links

extlinks = {
    "issue": ("https://github.com/lenskit/lkpy/issues/%s", "🐞 %s"),
    "pr": ("https://github.com/lenskit/lkpy/pull/%s", "⛙ %s"),
    "user": ("https://github.com/%s", "@%s"),
}
