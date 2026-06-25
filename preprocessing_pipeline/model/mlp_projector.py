"""
Vision-language MLP projector for ECG-LLaVA.

Maps CLIP ViT-L/14 vision tokens from R^{d_v=1024} to the Mistral-7B
embedding space R^{d_h=4096} using a 2-layer MLP with GELU activation.

References: thesis ch3 SS3.1.3, Eq. eq:mlp_proj.

    H_v = W2 * GELU(W1 * Z_v + b1) + b2

Parameter count: ~21.0M (W1: 4,194,304 + b1: 4,096 + W2: 16,777,216 + b2: 4,096).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (thesis ch3 SS3.1.3)
# ---------------------------------------------------------------------------

VISION_DIM: int = 1024         # CLIP ViT-L/14 hidden size (d_v)
LLM_HIDDEN_DIM: int = 4096    # Mistral-7B hidden size (d_h)
EXPECTED_PARAM_COUNT: int = 20_979_712  # ~21.0M


# ---------------------------------------------------------------------------
# MLP Projector
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """2-layer MLP projector with GELU activation (LLaVA-1.5 design).

    Maps CLIP vision tokens from R^{d_v} to the LLM embedding space R^{d_h}:

        H_v = GELU(Z_v @ W1 + b1) @ W2 + b2

    where W1 in R^{1024 x 4096}, W2 in R^{4096 x 4096}.

    Parameters
    ----------
    vision_dim:
        Input dimension from CLIP encoder.  Default: 1024.
    llm_hidden_dim:
        Output dimension matching LLM hidden size.  Default: 4096.
    """

    def __init__(
        self,
        vision_dim: int = VISION_DIM,
        llm_hidden_dim: int = LLM_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(vision_dim, llm_hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(llm_hidden_dim, llm_hidden_dim)
        log.info(
            "MLPProjector created: %d -> %d -> %d (%d params)",
            vision_dim, llm_hidden_dim, llm_hidden_dim, self.param_count,
        )

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        """Project vision tokens to LLM embedding space.

        Parameters
        ----------
        z_v:
            Vision tokens from CLIP encoder, shape (N, vision_dim) where
            N = 5184 for a single image (9 tiles x 576 patches).

        Returns
        -------
        torch.Tensor of shape (N, llm_hidden_dim) -- projected vision tokens.
        """
        return self.linear2(self.gelu(self.linear1(z_v)))

    @property
    def param_count(self) -> int:
        """Total number of parameters in the projector."""
        return sum(p.numel() for p in self.parameters())
