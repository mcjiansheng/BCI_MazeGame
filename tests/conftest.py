"""Shared pytest configuration for predictable CPU resource use."""

import importlib.util


def pytest_sessionstart(session):
    """Avoid severe OpenMP oversubscription when multiple test runs overlap."""

    if importlib.util.find_spec("torch") is not None:
        import torch
        torch.set_num_threads(1)
