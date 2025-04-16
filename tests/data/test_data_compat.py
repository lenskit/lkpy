import subprocess as sp
from os import fspath
from pathlib import Path

from pytest import mark, skip

from lenskit.data.dataset import Dataset
from lenskit.logging import get_logger
from lenskit.testing import ml_test_dir

# The LensKit versions we want to test backwards compatibility with
LK_VERSIONS = ["2025.1.1"]

_log = get_logger(__name__)
_ml_path = Path("data/ml-20m.zip")


@mark.parametrize("version", LK_VERSIONS)
def test_data_backwards_compat(version, tmpdir: Path):
    "Test that we can load datasets prepared by old versions."
    _log.info("processing ML file", version=version)

    try:
        sp.call(["uvx", "--version"])
    except FileNotFoundError:
        skip("uvx not installed")

    out_path = tmpdir / "ml-small.lk"
    pkg = f"lenskit=={version}"

    sp.check_call(
        ["uvx", pkg, "data", "convert", "--movielens", fspath(ml_test_dir), fspath(out_path)],
        env={"UV_TORCH_BACKEND": "cpu"},
    )

    _log.info("loading dataset")
    ml = Dataset.load(out_path)
    assert ml.schema.version is not None


@mark.realdata
@mark.parametrize("version", LK_VERSIONS)
def test_data_backwards_ml20m(version, tmpdir: Path):
    "Test that we can load datasets prepared by old versions (ML20M)."
    try:
        sp.call(["uvx", "--version"])
    except FileNotFoundError:
        skip("uvx not installed")

    _log.info("processing ML file", version=version)

    out_path = tmpdir / "ml-20m"
    pkg = f"lenskit=={version}"

    sp.check_call(
        ["uvx", pkg, "data", "convert", "--movielens", fspath(_ml_path), fspath(out_path)],
        env={"UV_TORCH_BACKEND": "cpu"},
    )

    _log.info("loading dataset")
    ml = Dataset.load(out_path)
    assert ml.schema.version is not None
