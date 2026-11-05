"""Serialize offline UDA runs that share a target pseudo-label/map cache."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from src.run_paths import normalize_config_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def target_pseudo_cache_lock(config_path, project_root=None):
    project_root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
    config_path = Path(config_path).expanduser()
    config_path = (config_path if config_path.is_absolute() else project_root / config_path).resolve()
    config = normalize_config_paths(json.loads(config_path.read_text(encoding="utf-8")), project_root)
    split = config["split_submission"]
    if split not in {"lightbox", "sunlamp"}:
        raise ValueError("offline UDA requires lightbox or sunlamp")
    root = Path(config["root_dir"]).resolve()
    lock_path = root / f".uda_cache_{split}.lock"
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"Target pseudo cache is locked: {lock_path}. Another UDA loop may be "
            "using its labels/maps. After an interrupted run, remove this file "
            "manually only after confirming no training process uses that cache."
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"token": token, "pid": os.getpid(), "config": str(config_path),
                       "created_utc": datetime.now(timezone.utc).isoformat()}, stream)
        yield lock_path
    finally:
        # Do not delete a lock that was replaced by a different owner.
        if lock_path.exists():
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                lock_path.unlink()
