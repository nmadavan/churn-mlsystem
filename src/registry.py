"""A minimal file-based model registry.

Each trained model is saved under models/v<N>/ (model.pkl + metadata.json). A
single current_best.json points at the promoted model that serving should load.
This is the lightweight stand-in for a tool like MLflow: versioned artifacts plus
one "which model is live" pointer, which is all the assignment needs.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config import resolve_path


def _registry_dir(cfg: dict) -> Path:
    d = resolve_path(cfg["model"]["registry_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_version(cfg: dict) -> str:
    """Return the next version label, e.g. 'v3' if v1 and v2 already exist."""
    existing = [
        int(p.name[1:])
        for p in _registry_dir(cfg).glob("v*")
        if p.is_dir() and p.name[1:].isdigit()
    ]
    return f"v{(max(existing) + 1) if existing else 1}"


def save_model(pipeline, metadata: dict, cfg: dict) -> str:
    """Persist a fitted pipeline and its metadata under a fresh version folder."""
    version = next_version(cfg)
    version_dir = _registry_dir(cfg) / version
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, version_dir / "model.pkl")
    metadata = {"version": version,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **metadata}
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return version


def promote(version: str, metrics: dict, cfg: dict) -> None:
    """Point current_best.json at `version` (the model serving will load)."""
    pointer = {
        "version": version,
        "model_path": f"{cfg['model']['registry_dir']}/{version}/model.pkl",
        "metrics": metrics,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    resolve_path(cfg["model"]["current_best_file"]).write_text(json.dumps(pointer, indent=2))


def current_best(cfg: dict) -> dict | None:
    """Return the current_best.json contents, or None if nothing is promoted yet."""
    pointer_path = resolve_path(cfg["model"]["current_best_file"])
    if not pointer_path.exists():
        return None
    return json.loads(pointer_path.read_text())


def load_current_model(cfg: dict):
    """Load the promoted pipeline for serving. Returns (pipeline, pointer_dict)."""
    pointer = current_best(cfg)
    if pointer is None:
        raise FileNotFoundError(
            "No promoted model. Run `python -m src.train` before serving."
        )
    pipeline = joblib.load(resolve_path(pointer["model_path"]))
    return pipeline, pointer
