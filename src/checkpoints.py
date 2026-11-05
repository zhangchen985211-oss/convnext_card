"""Consistent full-model checkpoint loading for ConvNeXt-CARD commands."""

from collections.abc import Mapping


def extract_state_dict(checkpoint):
    state = checkpoint
    if isinstance(state, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(state.get(key), Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Expected a non-empty model state dictionary")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError("Model state dictionary keys must be strings")
        while key.startswith("module."):
            key = key[len("module."):]
        if key in normalized:
            raise ValueError(f"Duplicate checkpoint parameter after normalization: {key}")
        normalized[key] = value
    return normalized


def load_model_checkpoint(model, path, device):
    import torch

    state = extract_state_dict(torch.load(path, map_location=device))
    model_state = model.state_dict()
    if not set(state).intersection(model_state):
        raise ValueError(f"Checkpoint has no parameters matching this model: {path}")
    for key in ("relation_decoder.adjacency", "relation_decoder.distances"):
        if key in state and key in model_state:
            if not torch.equal(state[key].detach().cpu(), model_state[key].detach().cpu()):
                raise ValueError(
                    f"Checkpoint {key} does not match the active CAD graph: {path}"
                )
    # Architecture changes must be explicit. A silently partial load can leave
    # an apparently trained model with randomly initialized pose heads.
    return model.load_state_dict(state, strict=True)
