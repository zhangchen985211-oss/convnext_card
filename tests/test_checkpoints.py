from __future__ import annotations

import sys
import types

import pytest

from src.checkpoints import load_model_checkpoint


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self


class _FakeModel:
    def __init__(self, state):
        self._state = state
        self.loaded = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, strict):
        self.loaded = (state, strict)
        return "loaded"


def _fake_torch(checkpoint):
    torch = types.ModuleType("torch")
    torch.load = lambda path, map_location: checkpoint
    torch.equal = lambda left, right: left.values == right.values
    return torch


@pytest.mark.parametrize("mismatch_key", [None, "relation_decoder.adjacency", "relation_decoder.distances"])
def test_checkpoint_load_requires_current_cad_graph_buffers(monkeypatch, mismatch_key):
    state = {
        "relation_decoder.adjacency": _FakeTensor([[1, 0], [0, 1]]),
        "relation_decoder.distances": _FakeTensor([[0.0, 1.0], [1.0, 0.0]]),
        "head.weight": _FakeTensor([1.0]),
    }
    checkpoint = dict(state)
    if mismatch_key is not None:
        checkpoint[mismatch_key] = _FakeTensor([[0]])
    model = _FakeModel(state)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(checkpoint))

    if mismatch_key is None:
        assert load_model_checkpoint(model, "model.pth", "cpu") == "loaded"
        assert model.loaded == (checkpoint, True)
    else:
        with pytest.raises(ValueError, match="does not match the active CAD graph"):
            load_model_checkpoint(model, "model.pth", "cpu")
        assert model.loaded is None
