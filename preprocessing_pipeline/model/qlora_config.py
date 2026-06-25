"""
QLoRA configuration for Mistral-7B-Instruct-v0.2.

Provides NF4 4-bit quantization config, LoRA adapter config, and model
loading utilities. References thesis section 3.1.4 (QLoRA) and
section 3.1.4.1 (LoRA math, Eq. eq:lora_forward, Eq. eq:adapter_count).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

MISTRAL_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
MISTRAL_HIDDEN_DIM: int = 4096

# ---------------------------------------------------------------------------
# LoRA hyperparameters (thesis SS3.1.4.1, Eq. eq:lora_forward)
# ---------------------------------------------------------------------------

LORA_RANK: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---------------------------------------------------------------------------
# Expected parameter counts (thesis Eq. eq:adapter_count)
# Per layer: (W_q + W_o) + (W_k + W_v) = 262,144 + 163,840 = 425,984
# Total: 32 layers x 425,984 = 13,631,488
# ---------------------------------------------------------------------------

LORA_PARAMS_PER_LAYER: int = 425_984
NUM_TRANSFORMER_LAYERS: int = 32
EXPECTED_LORA_PARAMS: int = LORA_PARAMS_PER_LAYER * NUM_TRANSFORMER_LAYERS  # 13,631,488

# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

QUANT_TYPE: str = "nf4"
USE_DOUBLE_QUANT: bool = True
COMPUTE_DTYPE: str = "bfloat16"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_bnb_config() -> BitsAndBytesConfig:
    """Create BitsAndBytes NF4 quantization config for Mistral-7B.

    Configures:
    - NF4 4-bit quantization (thesis SS3.1.4)
    - Double quantization (~0.37 bits/param savings)
    - BFloat16 compute dtype for dequantized operations

    Returns
    -------
    BitsAndBytesConfig ready for AutoModelForCausalLM.from_pretrained().
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def create_lora_config() -> LoraConfig:
    """Create LoRA adapter config for Mistral-7B attention layers.

    Targets W_q, W_k, W_v, W_o across all 32 transformer layers
    (thesis SS3.1.4.1, Eq. eq:adapter_count).

    LoRA forward: h = W_0 @ x + (alpha/r) * B @ A @ x
    Scaling factor: alpha/r = 32/16 = 2

    Returns
    -------
    LoraConfig ready for get_peft_model().
    """
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_quantized_mistral(
    model_name: str = MISTRAL_MODEL_NAME,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """Load Mistral-7B-Instruct-v0.2 with NF4 quantization.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Default: mistralai/Mistral-7B-Instruct-v0.2.
    device_map:
        Device mapping strategy.  Default: "auto".

    Returns
    -------
    AutoModelForCausalLM with NF4 quantized weights.
    """
    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    log.info(
        "Loaded quantized Mistral: %s (device_map=%s)",
        model_name, device_map,
    )
    return model


def attach_lora_adapters(model: AutoModelForCausalLM) -> PeftModel:
    """Attach LoRA adapters to a quantized Mistral-7B model.

    Inserts low-rank adapters on W_q, W_k, W_v, W_o across all 32
    transformer layers.  B initialized to zero, A from N(0, sigma^2).

    Parameters
    ----------
    model:
        Quantized Mistral-7B from load_quantized_mistral().

    Returns
    -------
    PeftModel with LoRA adapters attached.
    """
    lora_config = create_lora_config()
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    log.info(
        "LoRA adapters attached: %d trainable params (expected ~%d)",
        trainable, EXPECTED_LORA_PARAMS,
    )
    return peft_model
