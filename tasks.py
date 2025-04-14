from invoke import task
from invoke.context import Context


@task
def build_sdist(c: Context):
    "Build source distribution."
    c.run("uv build --sdist")


@task
def build_dist(c: Context):
    "Build packages for the current platform."
    c.run("uv build")


@task(build_sdist)
def build_conda(c: Context):
    "Build Conda packages."
    from setuptools_scm import get_version

    version = get_version()
    print("packaging LensKit version {}", version)
    cmd = "rattler-build build --recipe conda --output-dir dist/conda"
    c.run(cmd, echo=True, env={"LK_PACKAGE_VERSION": version})
