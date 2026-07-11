import json
import tomllib
from os import PathLike
from pathlib import Path
from typing import overload

from pydantic import BaseModel, JsonValue


@overload
def load_model_data(path: Path | PathLike[str], model: None = None) -> JsonValue: ...
@overload
def load_model_data[M: BaseModel](path: Path | PathLike[str], model: type[M]) -> M: ...
def load_model_data[M: BaseModel](path: Path | PathLike[str], model: type[M] | None = None):
    """
    General-purpose function to automatically load configuration data and
    optionally validate with a model.

    Args:
        path:
            The path to the configuration file.  The file type is automatically
            detected.
        model:
            The Pydantic model class to validate, or ``None`` to load as
            JSON-compatible data.
    Returns:
        The validated data.
    Stability:
        Internal
    """
    path = Path(path)
    text = path.read_text()

    match path.suffix:
        case ".json" if model is not None:
            return model.model_validate_json(text)
        case ".json":
            data = json.loads(text)
        case ".toml":
            data = tomllib.loads(text)
        case ".yaml" | ".yml":
            import yaml

            data = yaml.load(text, yaml.SafeLoader)

        case _:
            raise ValueError(f"unsupported configuration type for {path}")

    if model is None:
        return data
    else:
        return model.model_validate(data)
