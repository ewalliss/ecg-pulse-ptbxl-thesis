"""Tests for src.model.anyres_tiler — AnyRes fixed-grid tiling."""

import pytest
from PIL import Image

from preprocessing_pipeline.model.anyres_tiler import (
    tile_ecg_image,
    TILE_SIZE,
    GRID_COLS,
    GRID_ROWS,
    NUM_TOTAL_TILES,
)


def test_returns_nine_tiles(dummy_ecg_image):
    """tile_ecg_image returns exactly 9 tiles."""
    tiles = tile_ecg_image(dummy_ecg_image)
    assert len(tiles) == NUM_TOTAL_TILES


def test_all_tiles_are_336x336(dummy_ecg_image):
    """Every tile is 336x336 pixels."""
    tiles = tile_ecg_image(dummy_ecg_image)
    for i, tile in enumerate(tiles):
        assert tile.size == (TILE_SIZE, TILE_SIZE), f"Tile {i} has wrong size: {tile.size}"


def test_thumbnail_is_first(dummy_ecg_image):
    """Index 0 is the global thumbnail (bilinear downsample), not a crop."""
    tiles = tile_ecg_image(dummy_ecg_image)
    # The thumbnail is a full-image downsample, so its top-left pixel reflects
    # the entire image blended.  The first local tile (index 1) is an exact crop
    # of the top-left 336x336 region — its (0,0) pixel matches the original.
    original_top_left = dummy_ecg_image.getpixel((0, 0))
    local_top_left = tiles[1].getpixel((0, 0))
    thumbnail_top_left = tiles[0].getpixel((0, 0))

    # Local tile top-left must match the original exactly (it is a crop)
    assert local_top_left == original_top_left
    # Thumbnail is a downsample of the whole image — pixel values differ
    # because it squashes 1344x672 → 336x336 via bilinear interpolation
    assert thumbnail_top_left != local_top_left or tiles[0].tobytes() != tiles[1].tobytes()


def test_local_tiles_cover_full_image(dummy_ecg_image):
    """Reconstructing the original from local tiles matches at sample points."""
    tiles = tile_ecg_image(dummy_ecg_image)
    reconstructed = Image.new("RGB", (1344, 672))

    for idx in range(1, 9):  # local tiles only
        row = (idx - 1) // GRID_COLS
        col = (idx - 1) % GRID_COLS
        x = col * TILE_SIZE
        y = row * TILE_SIZE
        reconstructed.paste(tiles[idx], (x, y))

    # Compare at several sample points
    sample_points = [(0, 0), (335, 0), (672, 335), (1343, 671), (500, 300)]
    for px, py in sample_points:
        assert reconstructed.getpixel((px, py)) == dummy_ecg_image.getpixel((px, py)), (
            f"Mismatch at ({px}, {py})"
        )


def test_rejects_wrong_dimensions():
    """tile_ecg_image raises ValueError for non-1344x672 images."""
    with pytest.raises(ValueError, match="Expected 1344x672"):
        tile_ecg_image(Image.new("RGB", (500, 500)))

    with pytest.raises(ValueError, match="Expected 1344x672"):
        tile_ecg_image(Image.new("RGB", (672, 1344)))  # swapped

    with pytest.raises(ValueError, match="Expected 1344x672"):
        tile_ecg_image(Image.new("RGB", (1344, 673)))  # off by one


def test_tile_ordering_is_row_major(dummy_ecg_image):
    """Tiles 1-8 are in row-major order: top-left first, bottom-right last."""
    tiles = tile_ecg_image(dummy_ecg_image)

    # tiles[1] is top-left (row=0, col=0): its (0,0) matches original (0,0)
    assert tiles[1].getpixel((0, 0)) == dummy_ecg_image.getpixel((0, 0))

    # tiles[5] is second-row-left (row=1, col=0): its (0,0) matches original (0, 336)
    assert tiles[5].getpixel((0, 0)) == dummy_ecg_image.getpixel((0, 336))

    # tiles[4] is top-right (row=0, col=3): its (0,0) matches original (1008, 0)
    assert tiles[4].getpixel((0, 0)) == dummy_ecg_image.getpixel((1008, 0))

    # tiles[8] is bottom-right (row=1, col=3): its (0,0) matches original (1008, 336)
    assert tiles[8].getpixel((0, 0)) == dummy_ecg_image.getpixel((1008, 336))
