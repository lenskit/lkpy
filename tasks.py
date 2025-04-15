import logging
import os
import os.path
from pathlib import Path
from shutil import rmtree

from invoke.context import Context
from invoke.tasks import task

CACHEDIR_TAG = "Signature: 8a477f597d28d172789f06886806bc55"

_log = logging.getLogger("lenskit.invoke")
root = Path(__file__).parent


def _make_cache_dir(path: str | Path):
    "Create a directory and a CACHEDIR.TAG file."
    path = Path(path)
    path.mkdir(exist_ok=True)
    with open(path / "CACHEDIR.TAG", "wt") as ctf:
        print(CACHEDIR_TAG, file=ctf)
        print("# Cache directory marker for LensKit build", file=ctf)
        print(
            "# For information about cache directory tags see https://bford.info/cachedir/",
            file=ctf,
        )


@task
def setup_dirs(c: Context):
    "Initialize output directories."
    _make_cache_dir("dist")
    _make_cache_dir("build")
    _make_cache_dir("output")


@task(setup_dirs)
def build_sdist(c: Context):
    "Build source distribution."
    c.run("uv build --sdist")


@task(setup_dirs)
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
    if "CI" in os.environ:
        cmd += " --noarch-build-platform linux-64"
    c.run(cmd, echo=True, env={"LK_PACKAGE_VERSION": version})


@task
def clean(c: Context):
    print(c.config)
    for od in ["build", "dist", "output", "target"]:
        odp = root / od
        if odp.exists():
            print(f"🚮 removing {od}/   ")
            if not c.config.run.dry:
                rmtree(odp, ignore_errors=True)

    for glob in ["*.lprof", "*.profraw", "*.prof", "*.log"]:
        print(f"🚮 removing {glob}")
        for file in root.glob(glob):
            _log.info("removing %s", file)
            if not c.config.run.dry:
                file.unlink()

    print("cleaning generated doc files")
    c.run("git clean -xf docs")
