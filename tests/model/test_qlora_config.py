"""Tests for src.model.qlora_config -- QLoRA configuration and Mistral-7B loading.

Tests marked @pytest.mark.slow require downloading Mistral-7B weights (~4 GB quantized).
Tests marked @pytest.mark.gpu require a CUDA GPU.
Skip with: python -m pytest -m "not slow and not gpu"
"""

import pytest
import torch

from preprocessing_pipeline.model.qlora_config import (
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LORA_PARAMS_PER_LAYER,
    NUM_TRANSFORMER_LAYERS,
    EXPECTED_LORA_PARAMS,
    QUANT_TYPE,
    USE_DOUBLE_QUANT,
    create_bnb_config,
    create_lora_config,
)


def test_constants():
    assert LORA_RANK == 16
    assert LORA_ALPHA == 32
    assert LORA_DROPOUT == 0.05
    assert LORA_TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert NUM_TRANSFORMER_LAYERS == 32
    assert LORA_PARAMS_PER_LAYER == 425_984
    assert EXPECTED_LORA_PARAMS == 13_631_488
    assert QUANT_TYPE == "nf4"
    assert USE_DOUBLE_QUANT is True


def test_bnb_config_fields():
    cfg = create_bnb_config()
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == torch.bfloat16


def test_lora_config_fields():
    cfg = create_lora_config()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert set(cfg.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert cfg.bias == "none"
    assert cfg.task_type == "CAUSAL_LM"


def test_lora_scaling_factor():
    cfg = create_lora_config()
    scaling = cfg.lora_alpha / cfg.r
    assert scaling == 2, f"Expected scaling factor 2, got {scaling}"


@pytest.mark.slow
@pytest.mark.gpu
def test_load_quantized_mistral():
    from preprocessing_pipeline.model.qlora_config import load_quantized_mistral
    model = load_quantized_mistral()
    assert model.config.hidden_size == 4096
    assert model.config.num_hidden_layers == 32
    assert model.config.num_attention_heads == 32


@pytest.mark.slow
@pytest.mark.gpu
def test_adapter_param_count():
    from preprocessing_pipeline.model.qlora_config import load_quantized_mistral, attach_lora_adapters
    base = load_quantized_mistral()
    model = attach_lora_adapters(base)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == EXPECTED_LORA_PARAMS, (
        f"Expected {EXPECTED_LORA_PARAMS} trainable LoRA params, got {trainable}"
    )
