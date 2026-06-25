"""Shared pytest fixtures for the ECG-LLaVA test suite."""

import pytest
from PIL import Image


@pytest.fixture
def dummy_ecg_image() -> Image.Image:
    """Create a synthetic 1344x672 RGB image for tiling tests.

    Uses a gradient pattern so tiles are visually distinguishable.
    """
    img = Image.new("RGB", (1344, 672))
    pixels = img.load()
    for y in range(672):
        for x in range(1344):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


@pytest.fixture
def tile_size() -> int:
    """Standard CLIP tile size."""
    return 336
