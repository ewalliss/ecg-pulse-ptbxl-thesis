"""Weighted cross-entropy cho v2 (method-spec §5).

L_total = lambda * CE_prefix + CE_label
  CE_prefix : CE next-token CHUẨN trên token reasoning/khung (proper scoring rule)
  CE_label  : CE next-token trên token nhãn ("1"/"0"), nhân trọng số lớp class-balanced

KHÔNG dùng ASL (ASL định nghĩa cho head sigmoid K-logit độc lập [Ridnik 2021], lệch
kiến trúc decoder sinh). Trọng số lớp theo "effective number of samples" [Cui 2019].
Focal là ABLATION tuỳ chọn; cảnh báo focal non-proper [Mukhoti 2020] nên chỉ điều biến.

Module độc lập, thuần PyTorch — KHÔNG import/sửa source/LLaVA. Dùng làm tham chiếu
hoặc hook tuỳ chọn; đường train chính của v2 là CE GỐC của PULSE trên dữ liệu mới.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def class_balanced_weights(class_counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """w_k = (1-beta)/(1-beta^{n_k}), chuẩn hoá để sum(w)=num_classes [Cui 2019]."""
    counts = class_counts.clamp(min=1).to(torch.float64)
    eff = 1.0 - torch.pow(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.sum() * len(w)
    return w.to(torch.float32)


def weighted_ce_loss(
    logits: torch.Tensor,        # [B, S, V]
    labels: torch.Tensor,        # [B, S] target token id, IGNORE_INDEX = bỏ
    label_token_mask: torch.Tensor,  # [B, S] bool: target token là token NHÃN ("1"/"0")
    label_weights: torch.Tensor,     # [B, S] float: trọng số lớp tại vị trí nhãn (1.0 ở chỗ khác)
    lambda_prefix: float = 1.0,
    focal_gamma: float = 0.0,    # 0 = tắt focal (mặc định); >0 = ablation điều biến token nhãn
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trả (loss, metrics). Shift next-token bên trong."""
    # shift: dự đoán token i+1 từ vị trí i
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_lmask = label_token_mask[:, 1:].contiguous()
    shift_w = label_weights[:, 1:].contiguous()

    B, Sm1, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    per_tok = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX, reduction="none")
    per_tok = per_tok.view(B, Sm1)

    supervised = shift_labels != IGNORE_INDEX
    is_label = supervised & shift_lmask
    is_prefix = supervised & (~shift_lmask)

    # CE_prefix: token reasoning/khung
    if is_prefix.any():
        ce_prefix = per_tok[is_prefix].mean()
    else:
        ce_prefix = logits.sum() * 0.0

    # CE_label: token nhãn, nhân trọng số lớp (+ focal tuỳ chọn)
    if is_label.any():
        ce_lab_tok = per_tok[is_label]
        w_lab = shift_w[is_label]
        if focal_gamma > 0.0:
            with torch.no_grad():
                p = torch.exp(-ce_lab_tok)  # p của token đúng
            ce_lab_tok = ce_lab_tok * torch.pow(1.0 - p, focal_gamma)
        ce_label = (w_lab * ce_lab_tok).sum() / w_lab.sum().clamp(min=1e-8)
    else:
        ce_label = logits.sum() * 0.0

    loss = lambda_prefix * ce_prefix + ce_label
    metrics = {
        "ce_prefix": ce_prefix.detach(),
        "ce_label": ce_label.detach(),
        "n_label_tok": is_label.sum().detach(),
        "n_prefix_tok": is_prefix.sum().detach(),
    }
    return loss, metrics
