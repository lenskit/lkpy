import json
from os import fspath
from pathlib import Path

from click.testing import CliRunner

from lenskit.cli import lenskit
from lenskit.testing import ml_test_dir


def test_data_convert(tmpdir: Path):
    out_path = tmpdir / "ml-data"
    schema_file = out_path / "schema.json"

    runner = CliRunner()
    result = runner.invoke(
        lenskit, ["data", "convert", "--movielens", fspath(ml_test_dir), fspath(out_path)]
    )

    assert result.exit_code == 0
    assert schema_file.exists()

    data = json.loads(schema_file.read_text("utf8"))
    assert data["name"] == "ml-latest-small"
