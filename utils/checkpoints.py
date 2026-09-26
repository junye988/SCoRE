"""PyTorch checkpoint state loading."""

import torch


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"No model state dictionary in {path}")
    return state
