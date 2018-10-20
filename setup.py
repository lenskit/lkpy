import sys
import os
from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    readme = fh.read()

setup(
    name="lenskit",
    version="0.3.0",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    long_description=readme,
    url="https://lkpy.lenskit.org",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],

    setup_requires=[
        'pytest-runner'
    ],
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'numba >= 0.40',
        'fastparquet',
        'python-snappy'
    ],
    tests_require=[
        'pytest >= 3.5.1',
        'pytest-arraydiff'
    ],

    packages=find_packages()
)
