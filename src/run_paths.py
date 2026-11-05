"""Stable config/output paths shared by the two standalone route commands."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import tempfile


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATH_KEYS = ("root_dir", "path_results", "cad_graph", "graph_path", "path_pretrain", "project_root")
_PATH_PLACEHOLDER = re.compile(r"<(?:DATA_ROOT|OUTPUT_ROOT|PROJECT_ROOT)>")
_ITERATION_TOKEN = re.compile(r"niter_(\d{4})(?!\d)")


def normalize_config_paths(config, project_root):
    """Return a copy with runtime paths resolved against the project, never cwd."""
    root = Path(project_root).expanduser().resolve()
    normalized = dict(config)
    for key in _PATH_KEYS:
        value = config.get(key)
        if not value:
            continue
        if _PATH_PLACEHOLDER.search(str(value)):
            raise ValueError(f"Configure {key} before running: unresolved path placeholder in {value}")
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        normalized[key] = str(path)
        if key == "root_dir":
            normalized[key] = str(path).rstrip(os.sep) + os.sep
    return normalized


def _iteration_match(config_path):
    filename = Path(config_path).name
    matches = list(_ITERATION_TOKEN.finditer(filename))
    if len(matches) != 1 or filename.count("niter_") != 1:
        raise ValueError(f"Expected one unambiguous niter_ followed by four digits in config filename: {filename}")
    return matches[0]


def parse_config_iteration(config_path):
    """Read the unique four-digit round token from the filename only."""
    return int(_iteration_match(config_path).group(1))


def replace_config_iteration(config_path, iteration):
    """Replace the filename's round token, preserving directories and suffixes."""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or not 0 <= iteration <= 9999:
        raise ValueError("iteration must be an integer between 0 and 9999")
    path = Path(config_path)
    match = _iteration_match(path)
    filename = path.name[:match.start()] + f"niter_{iteration:04d}" + path.name[match.end():]
    return str(path.with_name(filename))


def project_config_arg(config_path, project_root):
    root = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Keep config files inside the uploaded project directory") from exc
    if path.suffix != ".json" or not path.is_file():
        raise FileNotFoundError(f"Expected an existing JSON config: {path}")
    return relative.as_posix()


def config_output_dir(config, config_arg, project_root):
    base = Path(normalize_config_paths(config, project_root)["path_results"])
    # config_arg must already have been canonicalized by project_config_arg.
    relative = Path(config_arg)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output paths require a project-relative config path")
    return base / relative


def _load_disk_config(config_arg, project_root):
    path = Path(project_root) / config_arg
    return normalize_config_paths(json.loads(path.read_text(encoding="utf-8")), project_root)


def _config_sha256(config_arg, project_root):
    # Read the original JSON: training adds transient keys and changes pretrained.
    config = _load_disk_config(config_arg, project_root)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(output, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=output.name + ".",
                                         suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def mark_training_complete(directory, config, config_arg, *, parent_checkpoint=None, project_root=None):
    root = Path(project_root or _PROJECT_ROOT).expanduser().resolve()
    config_arg = project_config_arg(config_arg, root)
    disk_config = _load_disk_config(config_arg, root)
    payload = {"config": config_arg, "start_epoch": int(config.get("start_epoch", 0)),
               "total_epochs": int(config["total_epochs"]), "schema_version": 2,
               "config_sha256": _config_sha256(config_arg, root)}
    if disk_config.get("isloop"):
        payload["checkpoint_sha256"] = _file_sha256(Path(directory) / "ckpt" / "init.pth")
        iteration = parse_config_iteration(config_arg)
        if iteration > 0:
            previous_arg = replace_config_iteration(config_arg, iteration - 1)
            expected = config_output_dir(disk_config, previous_arg, root) / "ckpt" / "init.pth"
            if parent_checkpoint is not None:
                supplied = Path(parent_checkpoint).expanduser()
                supplied = (supplied if supplied.is_absolute() else root / supplied).resolve()
                if supplied != expected.resolve():
                    raise ValueError(f"Parent checkpoint disagrees with previous round: {supplied} != {expected}")
            payload["parent_config"] = previous_arg
            payload["parent_checkpoint_sha256"] = _file_sha256(expected)
    _atomic_json(Path(directory) / "training_complete.json", payload)


def validate_training_complete(directory, config, config_arg, project_root):
    """Verify a finished UDA round and every ancestor through the seed checkpoint."""
    root = Path(project_root).expanduser().resolve()
    current_arg = project_config_arg(config_arg, root)
    current_output = Path(directory)
    hashes = {}

    def checkpoint_hash(path):
        path = path.resolve()
        if path not in hashes:
            hashes[path] = _file_sha256(path)
        return hashes[path]

    while parse_config_iteration(current_arg) > 0:
        marker = current_output / "training_complete.json"
        if not marker.is_file():
            raise RuntimeError(f"Previous round has no completion marker: {current_output}; finish/restart that round first")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise RuntimeError(f"Invalid completion marker: {marker}; preserve the results and restart or use a separate experiment") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            raise RuntimeError(f"Completion marker has no verified UDA lineage: {marker}; preserve existing results and use a separate experiment, "
                               "or explicitly restart this round when it has no downstream results")
        if payload.get("config") != current_arg or payload.get("config_sha256") != _config_sha256(current_arg, root):
            raise RuntimeError(f"Completed round configuration changed: {current_output}; use a separate experiment")
        current_checkpoint = current_output / "ckpt" / "init.pth"
        if payload.get("checkpoint_sha256") != checkpoint_hash(current_checkpoint):
            raise RuntimeError(f"Completed round checkpoint changed: {current_checkpoint}; use a separate experiment")
        previous_arg = replace_config_iteration(current_arg, parse_config_iteration(current_arg) - 1)
        previous_arg = project_config_arg(previous_arg, root)
        previous_config = _load_disk_config(previous_arg, root)
        previous_output = config_output_dir(previous_config, previous_arg, root)
        previous_checkpoint = previous_output / "ckpt" / "init.pth"
        if (payload.get("parent_config") != previous_arg
                or payload.get("parent_checkpoint_sha256") != checkpoint_hash(previous_checkpoint)):
            raise RuntimeError(f"Parent checkpoint changed for completed round: {current_output}; use a separate experiment")
        current_arg, current_output = previous_arg, previous_output


def assert_no_downstream_results(config_path, config, project_root):
    """Reject ancestor replacement without deleting any later experiment output."""
    root = Path(project_root).expanduser().resolve()
    current = Path(config_path).resolve()
    current_arg = current.relative_to(root).as_posix()
    iteration = parse_config_iteration(current)
    candidates = set()

    def is_downstream(filename):
        try:
            return (parse_config_iteration(filename) > iteration
                    and Path(replace_config_iteration(filename, iteration)).name == current.name)
        except ValueError:
            return False

    output_parent = config_output_dir(config, current_arg, root).parent
    if output_parent.is_dir():
        candidates.update(path for path in output_parent.iterdir() if is_downstream(path.name))
    for downstream in current.parent.glob("*.json"):
        if is_downstream(downstream.name):
            argument = project_config_arg(downstream, root)
            downstream_config = _load_disk_config(argument, root)
            candidates.add(config_output_dir(downstream_config, argument, root))
    existing = [path for path in sorted(candidates) if path.is_symlink()
                or (path.exists() and (not path.is_dir() or any(path.iterdir())))]
    if existing:
        raise FileExistsError(f"Cannot replace round {iteration:04d}: downstream results already exist at {existing[0]}. "
                              "Keep existing results and use a separate experiment config/output path")
