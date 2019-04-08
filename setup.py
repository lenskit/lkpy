from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    readme = fh.read()

setup(
    name="lenskit",
    version="0.6.1",
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

    python_requires='>= 3.5',
    setup_requires=[
        'pytest-runner'
    ],
    install_requires=[
        'pandas >= 0.20',
        'numpy',
        'scipy',
        'numba >= 0.38',
        'pyarrow',
        'cffi',
        'joblib'
    ],
    tests_require=[
        'pytest >= 3.9',
        'pytest-doctestplus'
    ],
    extras_require={
        'docs': [
            'sphinx >= 1.8',
            'sphinx_rtd_theme',
            'nbsphinx',
            'recommonmark',
            'ipython'
        ],
        'hpf': [
            'hpfrec'
        ],
        'implicit': [
            'implicit'
        ]
    },
    packages=find_packages()
)
