"""
Full ECG-LLaVA model: CLIP -> MLP projector -> QLoRA Mistral-7B.

Assembles the complete pipeline for multi-label ECG classification with
Stage 1 / Stage 2 training mode switching. References thesis section 3.1
architecture, Eq. eq:input_concat.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from preprocessing_pipeline.model.mlp_projector import MLPProjector
from preprocessing_pipeline.model.qlora_config import (
    load_quantized_mistral,
    attach_lora_adapters,
    EXPECTED_LORA_PARAMS,
    MISTRAL_MODEL_NAME,
)
from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected trainable parameter counts
# ---------------------------------------------------------------------------

EXPECTED_STAGE1_TRAINABLE: int = 20_979_712       # ~21.0M (MLP only)
EXPECTED_STAGE2_TRAINABLE: int = 34_611_200        # ~34.6M (MLP + LoRA)


# ---------------------------------------------------------------------------
# ECG-LLaVA model
# ---------------------------------------------------------------------------

class ECGLLaVA(nn.Module):
    """Full ECG-LLaVA model: CLIP -> MLP projector -> QLoRA Mistral-7B.

    Assembles the complete pipeline for multi-label ECG classification:
    1. Frozen CLIP ViT-L/14@336px encodes ECG tiles to Z_v in R^{5184 x 1024}
    2. MLP projector maps Z_v to H_v in R^{5184 x 4096}
    3. H_v is prepended to text embeddings and fed to Mistral-7B

    Two training stages (thesis SS3.3):
    - Stage 1: Only MLP projector is trainable (~21.0M params)
    - Stage 2: MLP + LoRA adapters are trainable (~34.6M params)

    Parameters
    ----------
    mistral_model_name:
        HuggingFace model ID for Mistral.
    device:
        Torch device string.  Default: "cuda".
    """

    def __init__(
        self,
        mistral_model_name: str = MISTRAL_MODEL_NAME,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device

        # MLP projector (BFloat16)
        self.projector = MLPProjector().to(torch.bfloat16)

        # Load quantized Mistral-7B + LoRA adapters
        base_model = load_quantized_mistral(mistral_model_name, device_map="auto")
        self.llm = attach_lora_adapters(base_model)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move projector to same device as LLM
        llm_device = next(self.llm.parameters()).device
        self.projector = self.projector.to(llm_device)

        # Default to Stage 1
        self.set_stage(1)
        log.info("ECGLLaVA created on %s", device)

    def set_stage(self, stage: int) -> None:
        """Configure trainable parameters for the given training stage.

        Parameters
        ----------
        stage:
            1 = train MLP only (freeze LoRA adapters)
            2 = train MLP + LoRA adapters jointly

        Raises
        ------
        ValueError
            If stage is not 1 or 2.
        """
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")

        self.projector.requires_grad_(True)

        if stage == 1:
            self.llm.disable_adapter_layers()
        else:
            self.llm.enable_adapter_layers()

        self._current_stage = stage
        trainable = self.get_trainable_param_count()
        log.info("Stage %d: %d trainable params", stage, trainable)

    def get_trainable_param_count(self) -> int:
        """Count parameters with requires_grad=True across all components."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass: project vision tokens and feed to LLM.

        Parameters
        ----------
        vision_tokens:
            Pre-encoded CLIP features, shape (batch_size, 5184, 1024).
        input_ids:
            Tokenized text input, shape (batch_size, seq_len).
        attention_mask:
            Attention mask for text, shape (batch_size, seq_len).
        labels:
            Target token IDs for loss, shape (batch_size, seq_len).
            If None, no loss is computed.

        Returns
        -------
        dict with "logits" and "loss" keys.
        """
        batch_size = vision_tokens.shape[0]
        n_vision = vision_tokens.shape[1]

        # 1. Project vision tokens: (batch, 5184, 1024) -> (batch, 5184, 4096)
        h_v = self.projector(vision_tokens)

        # 2. Get text embeddings: (batch, seq_len, 4096)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 3. Concatenate: [vision | text] -> (batch, 5184 + seq_len, 4096)
        inputs_embeds = torch.cat([h_v, text_embeds], dim=1)

        # 4. Build extended attention mask
        full_mask = None
        if attention_mask is not None:
            vision_mask = torch.ones(
                batch_size, n_vision,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # 5. Build extended labels (-100 for vision positions)
        full_labels = None
        if labels is not None:
            vision_labels = torch.full(
                (batch_size, n_vision), -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([vision_labels, labels], dim=1)

        # 6. Forward through LLM (inputs_embeds only, NOT input_ids)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )

        return {"logits": outputs.logits, "loss": outputs.loss}
