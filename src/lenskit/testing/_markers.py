"""
LensKit test markers.
"""

import os

import pytest

jit_enabled = True
if "NUMBA_DISABLE_JIT" in os.environ:
    jit_enabled = False
if os.environ.get("PYTORCH_JIT", None) == "0":
    jit_enabled = False

wantjit = pytest.mark.skipif(not jit_enabled, reason="JIT required")
