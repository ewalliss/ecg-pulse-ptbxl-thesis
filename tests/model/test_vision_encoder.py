"""Tests for src.model.vision_encoder — frozen CLIP ViT-L/14@336px wrapper.

Tests marked with @pytest.mark.slow require downloading CLIP weights (~1.7 GB).
Skip with: python -m pytest -m "not slow"
"""

import pytest
import torch
from PIL import Image

from preprocessing_pipeline.model.vision_encoder import (
    CLIP_MODEL_NAME,
    PATCHES_PER_TILE,
    TOTAL_VISION_TOKENS,
    VISION_HIDDEN_SIZE,
    VISION_SELECT_LAYER,
    VisionEncoder,
)
from preprocessing_pipeline.model.anyres_tiler import tile_ecg_image


def test_constants():
    """Module constants match the CLIP ViT-L/14@336px architecture."""
    assert VISION_HIDDEN_SIZE == 1024
    assert PATCHES_PER_TILE == 576
    assert TOTAL_VISION_TOKENS == 5184
    assert VISION_SELECT_LAYER == -2


@pytest.mark.slow
def test_encoder_loads_and_is_frozen():
    """VisionEncoder loads in eval mode with all parameters frozen."""
    encoder = VisionEncoder(device="cpu", dtype=torch.float32)
    assert encoder.model.training is False
    assert all(not p.requires_grad for p in encoder.model.parameters())


@pytest.mark.slow
def test_encode_tiles_output_shape(dummy_ecg_image):
    """encode_tiles produces (5184, 1024) tensor from 9 tiles."""
    encoder = VisionEncoder(device="cpu", dtype=torch.float32)
    tiles = tile_ecg_image(dummy_ecg_image)
    z_v = encoder.encode_tiles(tiles)
    assert z_v.shape == (TOTAL_VISION_TOKENS, VISION_HIDDEN_SIZE)
    assert z_v.dtype == torch.float32


@pytest.mark.slow
def test_encode_ecg_image_end_to_end(dummy_ecg_image):
    """encode_ecg_image produces (5184, 1024) from a 1344x672 image."""
    encoder = VisionEncoder(device="cpu", dtype=torch.float32)
    z_v = encoder.encode_ecg_image(dummy_ecg_image)
    assert z_v.shape == (TOTAL_VISION_TOKENS, VISION_HIDDEN_SIZE)


@pytest.mark.slow
def test_no_gradients_flow():
    """Frozen encoder output has no gradient tracking."""
    encoder = VisionEncoder(device="cpu", dtype=torch.float32)
    dummy_tiles = [Image.new("RGB", (336, 336), color=(i * 28, 0, 0)) for i in range(9)]
    z_v = encoder.encode_tiles(dummy_tiles)
    assert z_v.requires_grad is False


@pytest.mark.slow
def test_deterministic_output(dummy_ecg_image):
    """Frozen model produces identical output for the same input."""
    encoder = VisionEncoder(device="cpu", dtype=torch.float32)
    z_v_1 = encoder.encode_ecg_image(dummy_ecg_image)
    z_v_2 = encoder.encode_ecg_image(dummy_ecg_image)
    assert torch.allclose(z_v_1, z_v_2)
