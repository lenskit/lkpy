# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import nox


@nox.session(venv_backend="uv")
@nox.parametrize("torch", ["2.4", "2.5", "2.6", "2.7", "2.8"])
def sweep_torch(session, torch):
    session.install(f"torch ~={torch}.0", "-e", ".", "--group", "test")
    run_pytest(session)


@nox.session(venv_backend="uv")
@nox.parametrize("numpy", ["2.0", "2.1", "2.2", "2.3", "2.4", "2.5"])
def sweep_numpy(session, numpy):
    session.install(f"numpy ~={numpy}.0", "-e", ".", "--group", "test")
    run_pytest(session)


def run_pytest(session):
    opts = session.posargs
    if not opts:
        opts = ["-m", "not slow", "tests"]
    session.run("pytest", *opts)
