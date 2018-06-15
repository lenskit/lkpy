from setuptools import setup, find_packages

setup(
    name="lenskit",
    version="0.0.1",
    author="Michael Ekstrand",
    author_email="michaelekstrand@boisestate.edu",
    description="Run recommender algorithms and experiments",
    url="https://lenskit.github.io/lkpy",
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),

    install_requires=[
        'pandas',
        'numpy'
    ],
    setup_requires=[
        'pytest-runner',
        'pytest-cov',
        'sphinx'
    ],
    tests_require=[
        'pytest >= 3.5.1',
        'pytest-arraydiff',
        'dask',
        'toolz',
        'cloudpickle'
    ]
)
