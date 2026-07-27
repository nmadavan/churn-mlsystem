"""Load config.yaml and resolve project-relative paths.

Every module reads settings through here so paths and thresholds have exactly one
definition. Paths in config.yaml are relative to the project root; resolve_path()
turns them into absolute paths that work no matter where a script is launched from.
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "config.yaml") -> dict:
    with open(PROJECT_ROOT / path) as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Turn a config-relative path (e.g. 'data/raw/telco_churn.csv') into absolute."""
    return PROJECT_ROOT / relative
