"""Safe YAML configuration loading and serialization."""

from pathlib import Path
import re

import yaml


class _ConfigLoader(yaml.SafeLoader):
    """Accept exponent notation with or without a decimal point."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=_ConfigLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return value


def format_yaml(value):
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def save_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(format_yaml(value), encoding="utf-8")
    temporary.replace(path)
