"""Dựng mẫu huấn luyện 3-task từ một bản ghi PTB-XL (method-spec §3–4).

Một ECG -> 3 mẫu hội thoại (diag/rhythm/form), mỗi mẫu định dạng LLaVA chuẩn
(human hỏi kèm <image>, gpt trả lời = reasoning + khối nhãn). KHÔNG đụng source/LLaVA:
chỉ phát ra JSON-able dict mà train entry của PULSE đọc được như dữ liệu thường.
"""
from __future__ import annotations

from .labels import assign_labels
from .prompts import build_prompt, format_answer
from .scp_tasks import TASKS

IMAGE_PLACEHOLDER = "<image>"


def build_sample(
    record_id: str,
    image_path: str,
    scp_codes: dict[str, float],
    vocabs: dict[str, list[str]],
    reasoning_by_task: dict[str, str] | None = None,
    label_format: str = "binary",
    diag_threshold: float = 50.0,
) -> list[dict]:
    """Trả list 3 mẫu hội thoại (một mỗi task) cho record này."""
    reasoning_by_task = reasoning_by_task or {}
    samples = []
    for task in TASKS:
        vocab = vocabs[task]
        y = assign_labels(scp_codes, task, vocab, diag_threshold=diag_threshold)
        reasoning = reasoning_by_task.get(task, "")  # rỗng nếu chưa có supervision reasoning
        answer = format_answer(reasoning, vocab, y, label_format=label_format)
        samples.append({
            "id": f"{record_id}__{task}",
            "image": image_path,
            "task": task,
            "label_vector": y,
            "conversations": [
                {"from": "human", "value": f"{IMAGE_PLACEHOLDER}\n{build_prompt(task)}"},
                {"from": "gpt", "value": answer},
            ],
        })
    return samples
