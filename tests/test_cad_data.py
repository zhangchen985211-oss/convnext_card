"""Data-contract tests that do not require a PyTorch installation."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from scipy.io import savemat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cad_graph import build_speedplus_graph, load_graph, load_kpts_mat, save_graph


def _camera_file(root: Path) -> None:
    camera = {
        "fx": 1.0,
        "fy": 1.0,
        "Nu": 8,
        "Nv": 8,
        "ppx": 1.0,
        "ppy": 1.0,
        "cameraMatrix": [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
        "distCoeffs": [0.0] * 5,
    }
    (root / "camera.json").write_text(json.dumps(camera), encoding="utf-8")


def _load_with_stubs(name: str, path: Path, modules: dict[str, types.ModuleType]):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _torch_stubs():
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    data.Dataset = object
    utils.data = data
    torch.utils = utils
    return {"torch": torch, "torch.utils": utils, "torch.utils.data": data}


def _load_dataset_module():
    return _load_with_stubs(
        "_cad_contract_speedplus_loader",
        PROJECT_ROOT / "loaders/speedplus_segmentation_precomputed.py",
        _torch_stubs(),
    )


def test_load_kpts_mat_normalizes_both_orientations_and_preserves_fallback(tmp_path):
    expected = np.arange(33, dtype=np.float32).reshape(11, 3)
    horizontal = tmp_path / "horizontal.mat"
    vertical = tmp_path / "vertical.mat"
    fallback = tmp_path / "fallback.mat"
    savemat(horizontal, {"corners": expected})
    savemat(vertical, {"corners": expected.T})
    savemat(fallback, {"unique_key": expected.T})

    for path in (horizontal, vertical, fallback):
        actual = load_kpts_mat(path)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (11, 3)
        assert actual.dtype == np.float32
        assert actual.flags.c_contiguous


@pytest.mark.parametrize("shape", [(1, 33), (33, 1), (3, 10), (11, 4), (3, 3, 11)])
def test_load_kpts_mat_rejects_noncontract_shapes(tmp_path, shape):
    path = tmp_path / "bad_shape.mat"
    savemat(path, {"corners": np.zeros(shape, dtype=np.float32)})
    with pytest.raises(ValueError, match="expected CAD points"):
        load_kpts_mat(path)


@pytest.mark.parametrize("orientation", [(11, 3), (3, 11)])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_load_kpts_mat_rejects_nonfinite_values(tmp_path, orientation, bad_value):
    points = np.ones(orientation, dtype=np.float32)
    points.flat[0] = bad_value
    path = tmp_path / "nonfinite.mat"
    savemat(path, {"corners": points})
    with pytest.raises(ValueError, match="finite"):
        load_kpts_mat(path)


def test_load_kpts_mat_keeps_unique_shape_fallback_unambiguous(tmp_path):
    points = np.arange(33, dtype=np.float32).reshape(11, 3)
    ambiguous = tmp_path / "ambiguous.mat"
    savemat(ambiguous, {"first": points, "second": points.T})
    with pytest.raises(KeyError, match="unambiguous"):
        load_kpts_mat(ambiguous)


def test_graph_contract_checks_shape_finiteness_and_roundtrips(tmp_path):
    points = np.arange(33, dtype=np.float32).reshape(11, 3)
    adjacency, distances = build_speedplus_graph(points)
    graph_path = tmp_path / "graph.json"
    save_graph(graph_path, points, adjacency, distances)
    loaded = load_graph(graph_path)
    np.testing.assert_array_equal(loaded[0], points)
    np.testing.assert_array_equal(loaded[1], adjacency)
    np.testing.assert_allclose(loaded[2], distances)
    np.testing.assert_array_equal(load_graph(graph_path, expected_points=points)[0], points)
    wrong_points = points.copy()
    wrong_points[0, 0] += 1.0
    with pytest.raises(ValueError, match="do not match the active kpts.mat"):
        load_graph(graph_path, expected_points=wrong_points)

    with pytest.raises(ValueError, match="finite"):
        build_speedplus_graph(np.full((11, 3), np.nan, dtype=np.float32))

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["points_3d"] = np.zeros((3, 11)).tolist()
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"shape \[11,3\]"):
        load_graph(graph_path)

    payload["points_3d"] = points.tolist()
    payload["distances"][0][0] = float("nan")
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_graph(graph_path)

    payload["distances"][0][0] = 0.0
    payload["adjacency"][0][1] = 0.0
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="topology"):
        load_graph(graph_path)


@pytest.mark.parametrize("orientation", [(3, 11), (11, 3)])
def test_map_worker_reads_only_training_manifest(tmp_path, orientation):
    root = tmp_path / "data"
    dataset = root / "synthetic"
    (dataset / "images").mkdir(parents=True)
    (dataset / "kptsmap").mkdir()
    train = [{"filename": "train.jpg", "q_vbs2tango_true": [1, 0, 0, 0], "r_Vo2To_vbs_true": [0, 0, 10]}]
    (dataset / "train.json").write_text(json.dumps(train), encoding="utf-8")
    # It is intentionally not valid JSON: heatmap precomputation must not read it.
    (dataset / "validation.json").write_text("not json", encoding="utf-8")
    _camera_file(root)
    cad = np.arange(33, dtype=np.float32).reshape(11, 3)
    savemat(root / "kpts.mat", {"corners": cad.T if orientation == (3, 11) else cad})

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable
    maps = _load_with_stubs(
        "_cad_contract_create_maps",
        PROJECT_ROOT / "create_maps.py",
        {"tqdm": tqdm_module},
    )

    sample_ids = maps.initialize_worker({"root_dir": str(root), "split_submission": "synthetic"})
    assert sample_ids == ["train.jpg"]
    np.testing.assert_array_equal(maps.kpts, cad.T)


def test_create_maps_main_resolves_config_from_project_and_does_not_rewrite_it(tmp_path, monkeypatch):
    project = tmp_path / "uploaded_project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "maps.json"
    missing_data = project / "missing_data"
    config_text = json.dumps({"root_dir": "missing_data", "split_submission": "synthetic"}, indent=2)
    config_path.write_text(config_text, encoding="utf-8")

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable
    maps = _load_with_stubs(
        "_cad_contract_create_maps_main",
        PROJECT_ROOT / "create_maps.py",
        {"tqdm": tqdm_module},
    )
    monkeypatch.setattr(maps, "PROJECT_ROOT", project)
    monkeypatch.setattr(sys, "argv", ["create_maps.py", "--cfg", "configs/maps.json"])

    with pytest.raises(FileNotFoundError) as error:
        maps.main()
    assert str(missing_data / "synthetic" / "train.json") in str(error.value)
    assert config_path.read_text(encoding="utf-8") == config_text
    assert not missing_data.exists()


def test_create_maps_rejects_invalid_pose_and_skips_nonfinite_projection():
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable
    maps = _load_with_stubs(
        "_cad_contract_create_maps_invalid_inputs",
        PROJECT_ROOT / "create_maps.py",
        {"tqdm": tqdm_module},
    )

    with pytest.raises(ValueError, match="non-zero norm"):
        maps.quat2dcm([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="four finite"):
        maps.quat2dcm([1.0, 0.0, np.nan, 0.0])

    heatmap = np.zeros((16, 16), dtype=np.float32)
    result, visible = maps.draw_labelmap(heatmap, [np.nan, 4.0], 2.0)
    assert result is heatmap
    assert not visible


@pytest.mark.parametrize("orientation", [(3, 11), (11, 3)])
def test_validation_loader_does_not_require_or_read_heatmaps(tmp_path, orientation):
    root = tmp_path / "data"
    dataset_root = root / "synthetic"
    images = dataset_root / "images"
    images.mkdir(parents=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images / "validation.jpg"), image)
    _camera_file(root)
    cad = np.arange(33, dtype=np.float32).reshape(11, 3)
    cad[:, 2] = 0.0
    savemat(root / "kpts.mat", {"corners": cad.T if orientation == (3, 11) else cad})
    label = {"filename": "validation.jpg", "q_vbs2tango_true": [1, 0, 0, 0], "r_Vo2To_vbs_true": [0, 0, 10]}
    (dataset_root / "validation.json").write_text(json.dumps([label]), encoding="utf-8")
    (dataset_root / "train.json").write_text("malformed; train manifest is independent", encoding="utf-8")
    assert not (dataset_root / "kptsmap").exists()

    loader_module = _load_dataset_module()
    dataset = loader_module.PyTorchSatellitePoseEstimationDataset(
        split="validation",
        speed_root=str(root),
        transform_input=lambda value: value,
        config={"cols": 8, "rows": 8},
        dataset_id="synthetic",
    )
    sample = dataset[0]
    np.testing.assert_array_equal(dataset.kpts, cad.T)
    assert sample["image"].shape == (8, 8, 3)
    assert sample["y"] == "validation.jpg"
    assert sample["q0"].shape == (4,)
    assert not (dataset_root / "kptsmap").exists()


def test_train_loader_still_requires_training_heatmaps(tmp_path):
    root = tmp_path / "data"
    dataset_root = root / "synthetic"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "train.json").write_text("[]", encoding="utf-8")
    _camera_file(root)
    savemat(root / "kpts.mat", {"corners": np.zeros((11, 3), dtype=np.float32)})

    loader_module = _load_dataset_module()
    with pytest.raises(FileNotFoundError, match="kptsmap"):
        loader_module.PyTorchSatellitePoseEstimationDataset(
            split="train",
            speed_root=str(root),
            transform_input=lambda value: value,
            config={"cols": 8, "rows": 8},
            dataset_id="synthetic",
        )


def test_uda_build_loaders_skips_validation_dataset_and_honors_pin_memory():
    calls = []

    class FakeDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __len__(self):
            return 2

    torch = types.ModuleType("torch")
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(DataLoader=FakeLoader))
    fake_speedplus = types.ModuleType("loaders.speedplus_segmentation_precomputed")
    fake_speedplus.PyTorchSatellitePoseEstimationDataset = FakeDataset
    with patch.dict(sys.modules, {"torch": torch, "loaders.speedplus_segmentation_precomputed": fake_speedplus}):
        training_common = _load_with_stubs(
            "_cad_contract_training_common",
            PROJECT_ROOT / "src/training_common.py",
            {"torch": torch, "loaders.speedplus_segmentation_precomputed": fake_speedplus},
        )

    train_loader, validation_loader = training_common.build_loaders(
        {
            "isloop": True,
            "split_submission": "sunlamp",
            "root_dir": "/datasets",
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        object(),
    )
    assert validation_loader is None
    assert len(calls) == 1
    assert calls[0]["split"] == "sunlamp_train"
    assert train_loader.kwargs["pin_memory"] is False


def test_supervised_build_loaders_applies_pin_memory_to_both_loaders():
    class FakeDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __len__(self):
            return 2

    torch = types.ModuleType("torch")
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(DataLoader=FakeLoader))
    fake_speedplus = types.ModuleType("loaders.speedplus_segmentation_precomputed")
    fake_speedplus.PyTorchSatellitePoseEstimationDataset = FakeDataset
    training_common = _load_with_stubs(
        "_cad_contract_training_common_supervised",
        PROJECT_ROOT / "src/training_common.py",
        {"torch": torch, "loaders.speedplus_segmentation_precomputed": fake_speedplus},
    )
    train_loader, val_loader = training_common.build_loaders(
        {
            "isloop": False,
            "root_dir": "/datasets",
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        object(),
    )
    assert train_loader.kwargs["pin_memory"] is False
    assert val_loader.kwargs["pin_memory"] is False


def test_ast_confirms_shared_cad_reader_and_no_validation_map_scan():
    parsed = {}
    for relative in (
        "src/cad_graph.py",
        "create_maps.py",
        "loaders/speedplus_segmentation_precomputed.py",
        "src/training_common.py",
        "server/preflight.py",
    ):
        path = PROJECT_ROOT / relative
        parsed[relative] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for relative in ("create_maps.py", "loaders/speedplus_segmentation_precomputed.py", "server/preflight.py"):
        calls = [node for node in ast.walk(parsed[relative]) if isinstance(node, ast.Call)]
        assert any(isinstance(call.func, ast.Name) and call.func.id == "load_kpts_mat" for call in calls)

    map_source = (PROJECT_ROOT / "create_maps.py").read_text(encoding="utf-8")
    worker = next(node for node in parsed["create_maps.py"].body if isinstance(node, ast.FunctionDef) and node.name == "initialize_worker")
    worker_source = ast.get_source_segment(map_source, worker)
    assert "validation.json" not in worker_source
    common = next(node for node in parsed["src/training_common.py"].body if isinstance(node, ast.FunctionDef) and node.name == "build_loaders")
    assert any(isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2 for node in ast.walk(common))

