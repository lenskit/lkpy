import os

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


COVERAGE = os.environ.get("BUILD_FOR_COVER", None)
EXT_SPECS = {"lenskit.util.kvp": None}

CYTHON_OPTIONS = {}
C_DEFINES = []
if COVERAGE:
    print("enabling tracing")
    CYTHON_OPTIONS["linetrace"] = True
    C_DEFINES.append(("CYTHON_TRACE_NOGIL", "1"))


def _make_extension(name: str, opts: None) -> Extension:
    path = name.replace(".", "/")
    if USE_CYTHON:
        path += ".pyx"
    else:
        path += ".c"
    return Extension(name, [path], define_macros=C_DEFINES)


EXTENSIONS = [_make_extension(ext, opts) for (ext, opts) in EXT_SPECS.items()]
if USE_CYTHON:
    EXTENSIONS = cythonize(EXTENSIONS, compiler_directives=CYTHON_OPTIONS)
print(EXTENSIONS[0].__dict__)
setup(ext_modules=EXTENSIONS)
