# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lkbuild.tasks import *


@task
def update_env_specs(c):
    "Update environment files."
    c.run("pyproject2conda project")


@task
def docs(c, watch=False, rebuild=False):
    rb = "-a" if rebuild else ""
    if watch:
        c.run(f"sphinx-autobuild {rb} --watch lenskit docs build/doc")
    else:
        c.run(f"sphinx-build {rb} docs build/doc")
