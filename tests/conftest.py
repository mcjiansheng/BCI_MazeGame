"""Shared pytest configuration for predictable CPU resource use."""

import torch


def pytest_sessionstart(session):
    """Avoid severe OpenMP oversubscription when multiple test runs overlap."""

    torch.set_num_threads(1)
