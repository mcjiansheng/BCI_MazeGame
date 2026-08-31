"""LK-Mini-EEG16 acquisition and subject-specific motor-imagery tools."""

from .source import EEGChunk, EEGSource, create_source

__all__ = ["EEGChunk", "EEGSource", "create_source"]
