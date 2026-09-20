"""Accelerator selection and cache release, written once so the same code runs on CUDA boxes
and Apple silicon."""

import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache() -> None:
    """Both CUDA and MPS hold freed blocks in a caching allocator rather than returning them to
    the OS; the activation tensors here are large enough that releasing between batches
    matters. No-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
