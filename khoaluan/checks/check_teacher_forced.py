#!/usr/bin/env python3
"""Smoke-test logic build_answer_positions (KHÔNG cần model/tokenizer thật).

Fake tokenizer char-level: input_ids = [ord(c) for c in text]. ' 0'[-1]=48, ' 1'[-1]=49.
Kiểm: mỗi digit_position trỏ đúng vào placeholder neg_id, số slot == len(vocab),
và token ngay trước slot là khoảng trắng của 'code: '.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from khoaluan.code.teacher_forced import build_answer_positions, label_token_ids  # noqa: E402


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def main() -> int:
    tok = FakeTok()
    pos_id, neg_id = label_token_ids(tok)          # 49, 48
    vocab = ["NORM", "IMI", "1AVB", "AFIB"]
    answer_ids, digit_pos = build_answer_positions(tok, vocab, neg_id)

    ok = []
    ok.append(("pos_neg_ids", (pos_id, neg_id) == (ord("1"), ord("0"))))
    ok.append(("n_slots", len(digit_pos) == len(vocab)))
    ok.append(("each_slot_is_placeholder", all(answer_ids[p] == neg_id for p in digit_pos)))
    # token ngay trước mỗi slot phải là ' ' (của "code: ")
    ok.append(("prefix_is_space", all(answer_ids[p - 1] == ord(" ") for p in digit_pos)))
    # token ngay sau mỗi slot phải là '\n'
    ok.append(("suffix_is_newline", all(answer_ids[p + 1] == ord("\n") for p in digit_pos)))
    # vị trí tăng dần, nằm trong khoảng
    ok.append(("monotonic", digit_pos == sorted(digit_pos) and max(digit_pos) < len(answer_ids)))

    for name, c in ok:
        print(f"  {'PASS' if c else 'FAIL'}  {name}")
    allok = all(c for _, c in ok)
    print(f"\n{'ALL GREEN' if allok else 'HAS RED'} ({sum(c for _,c in ok)}/{len(ok)})")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
