import importlib.util
from pathlib import Path

import pytest
import torch


ASL_PATH = Path(__file__).resolve().parents[1] / "PULSE/LLaVA/llava/train/asl_gen_loss.py"
spec = importlib.util.spec_from_file_location("asl_gen_loss", ASL_PATH)
asl_gen_loss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asl_gen_loss)

compute_asl_gen_loss = asl_gen_loss.compute_asl_gen_loss


def _logits_for_binary_probs(positive_probs: list[float], positive_token_id: int, negative_token_id: int):
    logits = torch.full((1, 6, 8), -20.0)
    for idx, prob in enumerate(positive_probs):
        position = idx + 1
        prediction_position = position - 1
        logits[0, prediction_position, positive_token_id] = torch.logit(torch.tensor(prob))
        logits[0, prediction_position, negative_token_id] = 0.0
    return logits


def test_asl_gen_loss_is_finite_for_binary_slots():
    positive_token_id = 1
    negative_token_id = 2
    logits = _logits_for_binary_probs([0.8, 0.2], positive_token_id, negative_token_id)
    labels = torch.tensor([[-100, positive_token_id, negative_token_id, -100, -100, -100]])
    asl_targets = torch.tensor([[1.0, 0.0]])
    asl_label_positions = torch.tensor([[0, 1]])

    loss, metrics = compute_asl_gen_loss(
        logits=logits,
        labels=labels,
        asl_targets=asl_targets,
        asl_label_positions=asl_label_positions,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        probability_mode="binary",
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["asl_label_loss"])
    assert loss.item() > 0


@pytest.mark.parametrize(
    ("negative_prob", "expected_weight"),
    [
        (0.03, 0.0),
        (0.20, (0.20 - 0.05) ** 4),
    ],
)
def test_negative_margin_uses_shifted_positive_probability(negative_prob, expected_weight):
    positive_token_id = 1
    negative_token_id = 2
    logits = _logits_for_binary_probs([negative_prob], positive_token_id, negative_token_id)
    labels = torch.tensor([[-100, negative_token_id, -100, -100, -100, -100]])
    asl_targets = torch.tensor([[0.0]])
    asl_label_positions = torch.tensor([[0]])

    loss, metrics = compute_asl_gen_loss(
        logits=logits,
        labels=labels,
        asl_targets=asl_targets,
        asl_label_positions=asl_label_positions,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        lambda_0=0.0,
        prob_shift_m=0.05,
        probability_mode="binary",
    )

    assert metrics["asl_negative_weight"].item() == pytest.approx(expected_weight, abs=1e-6)
    if expected_weight == 0.0:
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_positive_branch_is_not_margin_shifted():
    positive_token_id = 1
    negative_token_id = 2
    logits = _logits_for_binary_probs([0.8], positive_token_id, negative_token_id)
    labels = torch.tensor([[-100, positive_token_id, -100, -100, -100, -100]])
    asl_targets = torch.tensor([[1.0]])
    asl_label_positions = torch.tensor([[0]])

    loss, metrics = compute_asl_gen_loss(
        logits=logits,
        labels=labels,
        asl_targets=asl_targets,
        asl_label_positions=asl_label_positions,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        lambda_0=0.0,
        prob_shift_m=0.05,
        probability_mode="binary",
    )

    expected_weight = (1.0 - 0.8) ** 1
    expected_ce = -torch.log(torch.tensor(0.8)).item()
    assert metrics["asl_positive_weight"].item() == pytest.approx(expected_weight, abs=1e-6)
    assert loss.item() == pytest.approx(expected_weight * expected_ce, abs=1e-6)


def test_binary_mode_renormalizes_positive_negative_tokens():
    positive_token_id = 1
    negative_token_id = 2
    logits = torch.full((1, 3, 8), -20.0)
    logits[0, 0, positive_token_id] = 5.0
    logits[0, 0, negative_token_id] = 5.0
    logits[0, 0, 7] = 10.0
    labels = torch.tensor([[-100, positive_token_id, -100]])
    asl_targets = torch.tensor([[1.0]])
    asl_label_positions = torch.tensor([[0]])

    _, metrics = compute_asl_gen_loss(
        logits=logits,
        labels=labels,
        asl_targets=asl_targets,
        asl_label_positions=asl_label_positions,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        lambda_0=0.0,
        probability_mode="binary",
    )

    assert metrics["asl_mean_p_pos"].item() == pytest.approx(0.5, abs=1e-6)


def test_positions_remap_over_inserted_image_tokens():
    positive_token_id = 1
    negative_token_id = 2
    logits = torch.full((1, 8, 8), -20.0)
    logits[0, 3, positive_token_id] = torch.logit(torch.tensor(0.8))
    logits[0, 3, negative_token_id] = 0.0
    labels = torch.tensor([[-100, -100, -100, -100, positive_token_id, -100, -100, -100]])
    asl_targets = torch.tensor([[1.0]])
    asl_label_positions = torch.tensor([[0]])

    _, metrics = compute_asl_gen_loss(
        logits=logits,
        labels=labels,
        asl_targets=asl_targets,
        asl_label_positions=asl_label_positions,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        lambda_0=0.0,
        probability_mode="binary",
    )

    assert metrics["asl_mean_p_pos"].item() == pytest.approx(0.8, abs=1e-6)


def test_thesis_mode_uses_full_vocab_token_probability():
    positive_token_id = 1
    negative_token_id = 2
    logits = torch.full((1, 3, 8), -20.0)
    logits[0, 0, positive_token_id] = 5.0
    logits[0, 0, negative_token_id] = 5.0
    logits[0, 0, 7] = 10.0
    labels = torch.tensor([[-100, positive_token_id, -100]])
    asl_targets = torch.tensor([[1.0]])
    asl_label_positions = torch.tensor([[0]])

    _, metrics = compute_asl_gen_loss(
        logits=logits,
        labels=labels,
        asl_targets=asl_targets,
        asl_label_positions=asl_label_positions,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        lambda_0=0.0,
        probability_mode="thesis",
    )

    full_vocab_positive = torch.softmax(logits[0, 0], dim=-1)[positive_token_id].item()
    assert metrics["asl_mean_p_pos"].item() == pytest.approx(full_vocab_positive, abs=1e-6)


def test_default_probability_mode_is_thesis():
    positive_token_id = 1
    negative_token_id = 2
    logits = torch.full((1, 3, 8), -20.0)
    logits[0, 0, positive_token_id] = 5.0
    logits[0, 0, negative_token_id] = 5.0
    logits[0, 0, 7] = 10.0
    labels = torch.tensor([[-100, positive_token_id, -100]])
    asl_targets = torch.tensor([[1.0]])
    asl_label_positions = torch.tensor([[0]])

    _, metrics = compute_asl_gen_loss(
        logits=logits,
        labels=labels,
        asl_targets=asl_targets,
        asl_label_positions=asl_label_positions,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        lambda_0=0.0,
    )

    full_vocab_positive = torch.softmax(logits[0, 0], dim=-1)[positive_token_id].item()
    assert metrics["asl_mean_p_pos"].item() == pytest.approx(full_vocab_positive, abs=1e-6)
