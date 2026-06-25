"""
AnyRes fixed-grid tiling for 1344x672 ECG images.

WARNING - UNUSED / DEAD CODE (kept for reference only).
This module is NOT on the real training or evaluation path. Training
(llava/train/train.py) and eval (khoaluan/code/eval_v2.py) both call LLaVA's
own `process_anyres_image` / `process_images`, which uses the model's
`image_grid_pinpoints` (max 1008). For a 1344x672 input that selects
best-fit 672x336, so the image is DOWNSCALED to 672x336 and split into a
2x1 grid (2 local tiles) + 1 global thumbnail = 3 tiles, 1728 visual
tokens (mm_patch_merge_type="flat"). The 4x2 / 9-tile / 5184-token scheme
below is therefore NOT what the trained model saw; it is only exercised by
tests. Do not cite its numbers in the thesis.

Implements the AnyRes tiling strategy from LLaVA-1.6, simplified for the
fixed 1344x672 input resolution used in this project.  The full image is
split into a 4x2 grid of 336x336 local tiles plus one bilinear-downscaled
336x336 global thumbnail, producing 9 tiles total.

Tile ordering follows LLaVA convention: thumbnail at index 0, followed by
local tiles in row-major order (top-left to bottom-right).

Reference
---------
- LLaVA ``process_anyres_image`` + ``divide_to_patches`` in ``mm_utils.py``
- Thesis §3.1.2: AnyRes tiling for ECG paper images
"""

from PIL import Image

from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tiling constants
# ---------------------------------------------------------------------------

TILE_SIZE: int = 336
GRID_COLS: int = 4                                # 1344 / 336
GRID_ROWS: int = 2                                # 672 / 336
NUM_LOCAL_TILES: int = GRID_COLS * GRID_ROWS      # 8
NUM_TOTAL_TILES: int = NUM_LOCAL_TILES + 1        # 9 (8 local + 1 thumbnail)
EXPECTED_WIDTH: int = 1344
EXPECTED_HEIGHT: int = 672


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tile_ecg_image(image: Image.Image) -> list[Image.Image]:
    """Split a 1344x672 ECG image into 9 tiles for CLIP encoding.

    Implements the AnyRes tiling strategy from LLaVA-1.6, simplified for the
    fixed 1344x672 input resolution used in this project (thesis §3.1.2).

    The thumbnail is produced by bilinear downsampling the full image to
    336x336 (intentionally squashing the 2:1 aspect ratio to provide a
    global context view, matching LLaVA's ``process_anyres_image`` behavior).

    Parameters
    ----------
    image:
        PIL Image of size exactly (1344, 672), mode RGB.

    Returns
    -------
    list of 9 PIL Images, each 336x336 px:
        Index 0: global thumbnail (bilinear downsample from full image)
        Index 1-8: local tiles in row-major order (top-left to bottom-right)

    Raises
    ------
    ValueError
        If image dimensions are not exactly 1344x672.
    """
    w, h = image.size
    if w != EXPECTED_WIDTH or h != EXPECTED_HEIGHT:
        raise ValueError(
            f"Expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT} image, got {w}x{h}"
        )

    # Global thumbnail: bilinear downsample to 336x336 (intentional squash)
    thumbnail = image.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)

    # Local tiles: 4x2 grid, row-major order
    local_tiles: list[Image.Image] = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            left = col * TILE_SIZE
            upper = row * TILE_SIZE
            box = (left, upper, left + TILE_SIZE, upper + TILE_SIZE)
            local_tiles.append(image.crop(box))

    log.debug("Tiled %dx%d image into %d tiles", w, h, NUM_TOTAL_TILES)
    return [thumbnail] + local_tiles
