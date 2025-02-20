import json
from os import fspath
from pathlib import Path

from click.testing import CliRunner

from lenskit.cli import lenskit
from lenskit.testing import ml_test_dir


def test_data_describe_ml():
    runner = CliRunner()
    result = runner.invoke(lenskit, ["data", "describe", "--movielens", fspath(ml_test_dir)])

    assert result.exit_code == 0

    summary = result.output
    print(summary)
    assert "Summary of ml-latest-small" in summary
    assert "tag_counts" in summary
