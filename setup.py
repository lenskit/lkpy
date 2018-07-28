import sys
import os
from setuptools import setup, find_packages
from distutils.extension import Extension


class LazyNPPath(os.PathLike):
    "NumPy include path, but evaluated lazily"
    def __fspath__(self):
        import numpy
        return numpy.get_include()

    def __str__(self):
        return self.__fspath__()


if sys.platform == 'win32':
    # MSVC - we do not yet support mingw or clang
    openmp_cflags = ['/openmp']
    openmp_lflags = []
else:
    # assume we are using GCC or compatible
    openmp_cflags = ['-fopenmp']
    openmp_lflags = ['-fopenmp']

debug_cflags = []
cython_opts = {}
if 'COVERAGE' in os.environ:
    debug_cflags += ['-DCYTHON_TRACE=1', '-DCYTHON_TRACE_NOGIL=1']
    cython_opts['linetrace'] = True


def extmod(name, openmp=False, cflags=[], ldflags=[]):
    "Create an extension object for one of our extension modules."
    comp_args = cflags + debug_cflags
    link_args = list(ldflags)
    if openmp:
        comp_args += openmp_cflags
        link_args += openmp_lflags
    parts = name.split('.')
    parts[-1] += '.pyx'
    path = os.path.join(*parts)
    return Extension(name, [path], include_dirs=[LazyNPPath()],
                     extra_compile_args=comp_args,
                     extra_link_args=link_args)


setup(
    name="lenskit",
    version="0.0.1",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    url="https://lenskit.github.io/lkpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    setup_requires=[
        'pytest-runner',
        'sphinx',
        'cython',
        'numpy',
        'scipy'
    ],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'tables >= 3.2.0'
    ],
    tests_require=[
        'pytest >= 3.5.1',
        'pytest-arraydiff'
    ],

    packages=find_packages(),
    ext_modules=[
        extmod('lenskit.algorithms._item_knn', openmp=True),
        extmod('lenskit.algorithms._funksvd')
    ]
)
