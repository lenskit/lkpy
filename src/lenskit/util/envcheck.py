# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Inspect the Python environment setup.  We provide this in LensKit so that people
can inspect (and report) their setups.
"""

import os
import platform
import re
import sys
from importlib.metadata import distributions, version

import threadpoolctl


def inspect_platform():
    print("Python version:", platform.python_version())
    print("Platform:", platform.platform())
    print("Python location:", sys.executable)


def inspect_version():
    print("LensKit version:", version("lenskit"))


def inspect_compute():
    import numpy as np
    import torch

    print("\nNumPy version:", np.__version__)
    print("PyTorch version:", torch.__version__)
    print("PyTorch backends:")
    print("    cpu:", torch.backends.cpu.get_cpu_capability())
    for mod in [torch.cuda, torch.backends.mkl, torch.backends.mps]:
        if mod.is_available():
            stat = "available"
        elif hasattr(mod, "is_built") and mod.is_built():
            stat = "unavailable"
        else:
            stat = "absent"
        name = mod.__name__.split(".")[-1]
        print(f"    {name}: {stat}")

    print("\nThreading layers:")
    for i, pool in enumerate(threadpoolctl.threadpool_info(), 1):
        print(f"    Backend {i}:")
        for k, v in pool.items():
            print("      {}: {}".format(k, v))


def inspect_env():
    print("\nEnvironment variables:")
    for k, v in os.environ.items():
        if re.match(r"^(LK_|OMP_|NUMBA_|MKL_|TORCH_)", k):
            print(f"    {k}: {v}")
    print("\nPython search paths:")
    for path in sys.path:
        print(f"    {path}")
    print("\nExecutable search paths:")
    paths = os.environ["PATH"].split(os.pathsep)
    for path in paths:
        print(f"    {path}")


def inspect_packages():
    dists = sorted(distributions(), key=lambda d: d.name)
    print("\nInstalled packages ({}):".format(len(dists)))
    for dist in dists:
        print("    {:32s}  {:>10s}".format(dist.name, dist.version))


def main():
    inspect_platform()
    inspect_version()
    inspect_compute()
    inspect_env()
    inspect_packages()


if __name__ == "__main__":
    main()
