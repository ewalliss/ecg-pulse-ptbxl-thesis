"""
Frozen CLIP ViT-L/14@336px vision encoder for AnyRes tiles.

Wraps HuggingFace ``CLIPVisionModel`` to extract patch-level features from
the penultimate transformer layer, following the LLaVA CLIPVisionTower
pattern.  All weights are frozen — no gradient computation during encoding.

Pipeline: 9 tiles (336x336 each) → CLIP → drop CLS → concatenate
→ Z_v in R^(5184 x 1024).

Reference
---------
- LLaVA ``CLIPVisionTower.feature_select`` in ``clip_encoder.py``
- Thesis §3.1.2: CLIP ViT-L/14@336px patch encoding
"""

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Encoder constants
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME: str = "openai/clip-vit-large-patch14-336"
VISION_HIDDEN_SIZE: int = 1024
VISION_SELECT_LAYER: int = -2        # penultimate layer (LLaVA default)
PATCHES_PER_TILE: int = 576          # (336 / 14)^2 = 24 * 24
TOTAL_VISION_TOKENS: int = 9 * PATCHES_PER_TILE   # 5184


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------

class VisionEncoder:
    """Frozen CLIP ViT-L/14@336px encoder for AnyRes tiles.

    Loads the vision-only model from HuggingFace, freezes all parameters,
    and extracts patch tokens from the penultimate hidden layer.  Follows
    the LLaVA CLIPVisionTower pattern (penultimate layer, CLS dropped).

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Default: openai/clip-vit-large-patch14-336.
    select_layer:
        Which hidden layer to extract features from.  Default: -2
        (penultimate layer, matching LLaVA convention).
    device:
        Torch device string.  Default: "cpu".
    dtype:
        Torch dtype for model weights.  Default: torch.float32.
        Use torch.float16 for GPU inference.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        select_layer: int = VISION_SELECT_LAYER,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.select_layer = select_layer
        self.device = device
        self.dtype = dtype

        self.model = CLIPVisionModel.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
        self.model.requires_grad_(False)
        self.model.eval()

        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        log.info(
            "CLIP vision encoder loaded: %s (device=%s, dtype=%s)",
            model_name, device, dtype,
        )

    # -----------------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def encode_tiles(self, tiles: list[Image.Image]) -> torch.Tensor:
        """Encode a list of PIL Image tiles into concatenated patch tokens.

        Parameters
        ----------
        tiles:
            List of 9 PIL Images, each 336x336 (thumbnail first, then
            8 local tiles in row-major order from ``tile_ecg_image``).

        Returns
        -------
        torch.Tensor of shape (5184, 1024) — all 9 tiles' patch tokens
        concatenated along the token dimension.
        """
        inputs = self.processor(images=tiles, return_tensors="pt")
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs, output_hidden_states=True)

        # Select penultimate hidden layer: shape (N, 577, 1024)
        hidden = outputs.hidden_states[self.select_layer]

        # Drop CLS token: shape (N, 576, 1024)
        patch_tokens = hidden[:, 1:, :]

        # Concatenate all tiles: shape (5184, 1024)
        z_v = patch_tokens.reshape(-1, patch_tokens.shape[-1])

        log.debug(
            "Encoded %d tiles → Z_v shape %s", len(tiles), tuple(z_v.shape),
        )
        return z_v

    def encode_ecg_image(self, image: Image.Image) -> torch.Tensor:
        """End-to-end: tile a 1344x672 image and encode all tiles.

        Parameters
        ----------
        image:
            PIL Image of size exactly (1344, 672).

        Returns
        -------
        torch.Tensor of shape (5184, 1024).
        """
        from preprocessing_pipeline.model.anyres_tiler import tile_ecg_image

        tiles = tile_ecg_image(image)
        return self.encode_tiles(tiles)
