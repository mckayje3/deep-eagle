"""Safe checkpoint loading helpers."""

from __future__ import annotations

import pickle

import torch


def safe_torch_load(path: str, map_location=None) -> dict:
    """
    Load a checkpoint with ``weights_only=True`` to prevent arbitrary code
    execution from a malicious pickle.

    Checkpoints saved by older versions of this framework (or any checkpoint
    containing non-tensor Python objects) cannot be loaded in this mode. Rather
    than silently falling back to the insecure ``weights_only=False`` path, we
    raise a clear, actionable error.

    Args:
        path: Path to the ``.pt`` checkpoint.
        map_location: Optional device mapping passed through to ``torch.load``.

    Returns:
        The loaded checkpoint dict.

    Raises:
        RuntimeError: If the checkpoint cannot be loaded under ``weights_only=True``.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError) as e:
        raise RuntimeError(
            f"Could not securely load checkpoint '{path}' with weights_only=True. "
            "It likely predates the security upgrade or contains non-tensor objects. "
            "Re-save it with the current framework version (model.save() / "
            "trainer.save_checkpoint()) from a trusted environment, or, only if you "
            "fully trust the file's origin, load it manually with "
            "torch.load(path, weights_only=False)."
        ) from e
