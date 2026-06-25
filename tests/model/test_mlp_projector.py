"""Tests for src.model.mlp_projector -- 2-layer GELU MLP projector."""

import pytest
import torch

from preprocessing_pipeline.model.mlp_projector import (
    MLPProjector,
    VISION_DIM,
    LLM_HIDDEN_DIM,
    EXPECTED_PARAM_COUNT,
)


def test_constants():
    assert VISION_DIM == 1024
    assert LLM_HIDDEN_DIM == 4096
    assert EXPECTED_PARAM_COUNT == 20_979_712


def test_param_count():
    proj = MLPProjector()
    assert proj.param_count == EXPECTED_PARAM_COUNT, (
        f"Expected {EXPECTED_PARAM_COUNT}, got {proj.param_count}"
    )


def test_output_shape():
    proj = MLPProjector()
    z_v = torch.randn(5184, 1024)
    h_v = proj(z_v)
    assert h_v.shape == (5184, 4096)


def test_output_shape_single_token():
    proj = MLPProjector()
    z_v = torch.randn(1, 1024)
    h_v = proj(z_v)
    assert h_v.shape == (1, 4096)


def test_bfloat16_dtype():
    proj = MLPProjector().to(torch.bfloat16)
    z_v = torch.randn(10, 1024, dtype=torch.bfloat16)
    h_v = proj(z_v)
    assert h_v.dtype == torch.bfloat16


def test_gradient_flow():
    proj = MLPProjector()
    z_v = torch.randn(10, 1024)
    out = proj(z_v)
    out.sum().backward()
    assert proj.linear1.weight.grad is not None
    assert proj.linear2.weight.grad is not None
    assert proj.linear1.weight.grad.abs().sum() > 0
    assert proj.linear2.weight.grad.abs().sum() > 0
