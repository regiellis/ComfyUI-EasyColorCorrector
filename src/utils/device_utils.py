"""Device and GPU management utilities."""

import torch


def get_preferred_device(use_gpu: bool = True):
    """Get the preferred device considering user settings."""
    if not use_gpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")