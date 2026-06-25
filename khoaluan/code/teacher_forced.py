"""Teacher-forced scoring cho v2 (method-spec §6) — KHÔNG generate, KHÔNG parse.

Ta tự dựng chuỗi answer mẫu "code: 0\\n" cho từng code, ghi vị trí token chữ số,
rồi MỘT forward pass đọc P("1")=softmax([z[id0],z[id1]]) tại logit của vị trí p-1.
Vì vị trí do ta đặt -> luôn đủ n_codes slot, không bao giờ malformed (vá lỗi 73%
malformed của free-generation).

Hàm build_answer_positions thuần tokenizer (test được không cần model). score_task
cần model + ảnh (chạy trên GPU).
"""
from __future__ import annotations

LABELS_HEADER = "[LABELS]\n"


def label_token_ids(tokenizer) -> tuple[int, int]:
    """id của token nhãn dương/âm, khớp _single_token_id lúc train (subtoken cuối của ' 1'/' 0')."""
    pos = tokenizer(" 1", add_special_tokens=False).input_ids[-1]
    neg = tokenizer(" 0", add_special_tokens=False).input_ids[-1]
    return pos, neg


def build_answer_positions(tokenizer, vocab: list[str], neg_id: int,
                           prefix: str = "[REASONING]\n\n",
                           placeholder_id: int | None = None) -> tuple[list[int], list[int]]:
    """Dựng token-ids của khối answer mẫu + vị trí token chữ số cho từng code.

    Dựng THEO TỪNG MẢNH (không tokenize cả chuỗi) để biết chính xác chỉ số slot,
    tránh SentencePiece merge làm lệch vị trí. `prefix` khớp template training
    (format_answer xuất "[REASONING]\\n\\n[LABELS]\\n..."), giữ điều kiện hoá tương tự.

    placeholder_id: token điền vào MỌI slot (mặc định neg_id="0"). KHÔNG phụ thuộc
    nhãn thật của mẫu -> nhãn thật không bao giờ vào context (chống leakage). Tham số
    này CHỈ để kiểm soát: đặt = pos_id ("1") và kiểm tra điểm số gần như không đổi
    chứng minh điểm đọc là logit DỰ ĐOÁN tại slot, không phải token bị teacher-force.

    Trả (answer_ids, digit_positions) — digit_positions[k] là chỉ số (trong answer_ids)
    của token chữ số ứng với vocab[k].
    """
    fill = neg_id if placeholder_id is None else placeholder_id
    answer_ids: list[int] = list(tokenizer(prefix + LABELS_HEADER, add_special_tokens=False).input_ids)
    digit_positions: list[int] = []
    newline_ids = tokenizer("\n", add_special_tokens=False).input_ids
    for code in vocab:
        prefix = tokenizer(f"{code}: ", add_special_tokens=False).input_ids
        answer_ids += prefix
        digit_positions.append(len(answer_ids))  # vị trí token chữ số sắp thêm
        answer_ids += [fill]                      # placeholder cố định (độc lập nhãn thật)
        answer_ids += newline_ids
    return answer_ids, digit_positions


def score_task(model, tokenizer, prompt_ids, images, image_sizes,
               answer_ids: list[int], digit_positions: list[int],
               pos_id: int, neg_id: int, device: str) -> list[float]:
    """Một forward pass teacher-forced -> P('1') cho từng code.

    prompt_ids: [1, Lp] token phần prompt (đã gồm <image> placeholder của LLaVA).
    answer_ids/digit_positions: từ build_answer_positions (chỉ số trong KHÔNG GIAN answer).
    Import torch lazy để module nhập được nơi không có torch (chỉ score_task cần).
    """
    import torch
    with torch.inference_mode():
        ans = torch.tensor(answer_ids, device=device, dtype=prompt_ids.dtype).unsqueeze(0)
        input_ids = torch.cat([prompt_ids, ans], dim=1)               # [1, Lp + La]
        out = model(input_ids=input_ids, images=images, image_sizes=image_sizes, use_cache=False)
        logits = out.logits[0]                                        # [seq, vocab]

        # LLaVA mở rộng <image> thành nhiều token thị giác BÊN TRONG forward -> seq dài hơn
        # input_ids. Phần answer luôn nằm ở ĐUÔI, nên neo theo cuối: chỉ số answer j trong
        # input gốc tương ứng vị trí (seq_len - La + j) ở logits.
        seq_len = logits.shape[0]
        la = len(answer_ids)
        p1 = []
        for j in digit_positions:
            pos_in_seq = seq_len - la + j        # vị trí token chữ số trong logits
            z = logits[pos_in_seq - 1]           # logit DỰ ĐOÁN token đó (teacher forcing)
            two = torch.stack([z[neg_id], z[pos_id]]).float()
            p1.append(torch.softmax(two, dim=0)[1].item())
        return p1
