"""Prompt theo task + định dạng answer (reasoning + khối nhãn) — method-spec §4.

Format nhãn mặc định: "code: 0|1" mỗi dòng (tham số hoá; biến thể 'positive_list'
chỉ liệt kê code dương). Reasoning đặt TRƯỚC khối nhãn để điều kiện hoá dự đoán.
"""
from __future__ import annotations

# Prompt tiếng ANH để khớp phân phối huấn luyện của PULSE/ECGInstruct (prompt tiếng
# Việt là off-distribution, hại hiểu task). Mã nhãn (NORM, IMI…) trung tính ngôn ngữ;
# luận văn/paper vẫn tiếng Việt — tách biệt với prompt train.
TASK_QUESTION = {
    "diag": "Based on the ECG image, identify which diagnostic statements are present. "
            "Briefly explain the key waveform findings, then list each label.",
    "rhythm": "Based on the ECG image, identify which cardiac rhythms are present. "
              "Briefly explain the rhythm and RR-interval findings, then list each label.",
    "form": "Based on the ECG image, identify which morphology (form) features are present. "
            "Briefly explain the QRS/ST/T morphology, then list each label.",
}

REASONING_HEADER = "[REASONING]"
LABELS_HEADER = "[LABELS]"


def build_prompt(task: str) -> str:
    """Phần text hỏi của task (đi kèm token ảnh ở pipeline PULSE)."""
    return TASK_QUESTION[task]


def format_answer(
    reasoning: str,
    vocab: list[str],
    label_vector: list[int],
    label_format: str = "binary",
) -> str:
    """Dựng chuỗi answer target để supervise (method-spec §4).

    label_format='binary'      -> mỗi dòng 'code: 0|1' (mặc định, trích macro-AUC)
    label_format='positive_list' -> chỉ liệt kê code dương
    """
    lines = [REASONING_HEADER, reasoning.strip(), LABELS_HEADER]
    if label_format == "binary":
        for code, y in zip(vocab, label_vector):
            lines.append(f"{code}: {int(y)}")
    elif label_format == "positive_list":
        pos = [c for c, y in zip(vocab, label_vector) if y]
        lines.append(", ".join(pos) if pos else "(none)")
    else:
        raise ValueError(f"label_format không hợp lệ: {label_format}")
    return "\n".join(lines)
