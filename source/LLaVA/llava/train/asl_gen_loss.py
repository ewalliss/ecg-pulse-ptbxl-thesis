from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


ProbabilityMode = Literal["thesis", "binary"]


def _remap_label_positions(labels: torch.Tensor, positions: torch.Tensor, ignore_index: int) -> torch.Tensor:
    remapped = torch.full_like(positions, -1)
    for batch_idx in range(labels.shape[0]):
        supervised_positions = torch.where(labels[batch_idx] != ignore_index)[0]
        valid = positions[batch_idx] >= 0
        source_positions = positions[batch_idx, valid]
        in_range = source_positions < supervised_positions.numel()
        if in_range.any():
            valid_indices = torch.where(valid)[0][in_range]
            remapped[batch_idx, valid_indices] = supervised_positions[source_positions[in_range]]
    return remapped


def compute_asl_gen_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    asl_targets: torch.Tensor,
    asl_label_positions: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
    lambda_0: float = 0.3,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    prob_shift_m: float = 0.05,
    ignore_index: int = -100,
    probability_mode: ProbabilityMode = "thesis",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if probability_mode not in ("thesis", "binary"):
        raise ValueError(f"Unsupported ASL-Gen probability mode: {probability_mode}")

    targets = asl_targets.to(device=logits.device, dtype=logits.dtype)
    positions = asl_label_positions.to(device=logits.device, dtype=torch.long)
    labels = labels.to(device=logits.device)
    positions = _remap_label_positions(labels, positions, ignore_index)

    batch_size, seq_len, _ = logits.shape
    safe_positions = positions.clamp(min=0, max=seq_len - 1)
    batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1)
    batch_indices = batch_indices.expand_as(safe_positions)

    target_token_ids = labels.gather(1, safe_positions)
    valid_label_mask = (positions > 0) & (positions < seq_len) & (target_token_ids != ignore_index)

    prediction_positions = (positions - 1).clamp(min=0, max=seq_len - 1)
    label_logits = logits[batch_indices, prediction_positions]
    log_probs = F.log_softmax(label_logits, dim=-1)

    safe_target_ids = target_token_ids.clamp(min=0)
    label_ce = -log_probs.gather(-1, safe_target_ids.unsqueeze(-1)).squeeze(-1)

    if probability_mode == "binary":
        binary_logits = label_logits[..., [positive_token_id, negative_token_id]]
        p_pos = F.softmax(binary_logits, dim=-1)[..., 0]
    else:
        p_pos = log_probs[..., positive_token_id].exp()

    p_pos = p_pos.clamp(min=eps, max=1.0 - eps)
    shifted_negative_prob = (p_pos - prob_shift_m).clamp(min=0.0)
    positive_weight = (1.0 - p_pos).pow(gamma_pos)
    negative_weight = shifted_negative_prob.pow(gamma_neg)
    weights = (targets * positive_weight + (1.0 - targets) * negative_weight).detach()

    valid_label_weights = valid_label_mask.to(dtype=logits.dtype)
    per_sample_label_loss = (weights * label_ce * valid_label_weights).sum(dim=1)
    label_loss = per_sample_label_loss.mean()

    label_position_mask = torch.zeros_like(labels, dtype=torch.bool)
    label_position_mask.scatter_(1, safe_positions, valid_label_mask)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    non_label_mask = (shift_labels != ignore_index) & (~label_position_mask[:, 1:])
    if non_label_mask.any():
        token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view_as(shift_labels)
        prefix_ce = token_ce[non_label_mask].mean()
    else:
        prefix_ce = logits.sum() * 0.0

    loss = lambda_0 * prefix_ce + label_loss
    metrics = {
        "asl_label_loss": label_loss.detach(),
        "asl_prefix_ce": prefix_ce.detach(),
        "asl_mean_p_pos": p_pos[valid_label_mask].mean().detach() if valid_label_mask.any() else p_pos.mean().detach(),
        "asl_positive_weight": positive_weight[valid_label_mask].mean().detach() if valid_label_mask.any() else positive_weight.mean().detach(),
        "asl_negative_weight": negative_weight[valid_label_mask].mean().detach() if valid_label_mask.any() else negative_weight.mean().detach(),
    }
    return loss, metrics
