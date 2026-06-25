"""Tests for src.model.ecg_llava -- full ECG-LLaVA model assembly.

Tests marked @pytest.mark.slow require downloading Mistral-7B weights (~4 GB quantized).
Tests marked @pytest.mark.gpu require a CUDA GPU with >= 16 GB VRAM.
Skip with: python -m pytest -m "not slow and not gpu"
"""

import pytest
import torch

from preprocessing_pipeline.model.mlp_projector import MLPProjector
from preprocessing_pipeline.model.ecg_llava import EXPECTED_STAGE1_TRAINABLE, EXPECTED_STAGE2_TRAINABLE


def test_constants():
    assert EXPECTED_STAGE1_TRAINABLE == 20_979_712
    assert EXPECTED_STAGE2_TRAINABLE == 34_611_200


def test_mlp_projector_standalone():
    proj = MLPProjector()
    assert proj.param_count == 20_979_712
    z_v = torch.randn(100, 1024)
    h_v = proj(z_v)
    assert h_v.shape == (100, 4096)


@pytest.mark.slow
@pytest.mark.gpu
def test_stage1_trainable_params():
    from preprocessing_pipeline.model.ecg_llava import ECGLLaVA
    model = ECGLLaVA()
    model.set_stage(1)
    assert model.get_trainable_param_count() == EXPECTED_STAGE1_TRAINABLE


@pytest.mark.slow
@pytest.mark.gpu
def test_stage2_trainable_params():
    from preprocessing_pipeline.model.ecg_llava import ECGLLaVA
    model = ECGLLaVA()
    model.set_stage(2)
    assert model.get_trainable_param_count() == EXPECTED_STAGE2_TRAINABLE


@pytest.mark.slow
@pytest.mark.gpu
def test_forward_pass_shape():
    from preprocessing_pipeline.model.ecg_llava import ECGLLaVA
    model = ECGLLaVA()
    model.set_stage(2)
    vision_tokens = torch.randn(1, 5184, 1024, dtype=torch.bfloat16, device="cuda")
    input_ids = torch.randint(0, 32000, (1, 20), device="cuda")
    attention_mask = torch.ones(1, 20, dtype=torch.long, device="cuda")
    outputs = model(vision_tokens, input_ids, attention_mask)
    vocab_size = model.llm.config.vocab_size
    assert outputs["logits"].shape == (1, 5184 + 20, vocab_size)
    assert outputs["loss"] is None


@pytest.mark.slow
@pytest.mark.gpu
def test_forward_with_labels():
    from preprocessing_pipeline.model.ecg_llava import ECGLLaVA
    model = ECGLLaVA()
    model.set_stage(2)
    vision_tokens = torch.randn(1, 5184, 1024, dtype=torch.bfloat16, device="cuda")
    input_ids = torch.randint(0, 32000, (1, 20), device="cuda")
    attention_mask = torch.ones(1, 20, dtype=torch.long, device="cuda")
    labels = torch.randint(0, 32000, (1, 20), device="cuda")
    outputs = model(vision_tokens, input_ids, attention_mask, labels=labels)
    assert outputs["loss"] is not None
    assert outputs["loss"].requires_grad is True
