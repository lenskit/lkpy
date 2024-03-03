from Cython.Build import cythonize
from setuptools import Extension, setup

EXT_SPECS = {"lenskit.util.kvp": None}


def _make_extension(name: str, opts: None) -> Extension:
    path = name.replace(".", "/") + ".pyx"
    return Extension(name, [path])


EXTENSIONS = [_make_extension(ext, opts) for (ext, opts) in EXT_SPECS.items()]
setup(ext_modules=cythonize(EXTENSIONS))
