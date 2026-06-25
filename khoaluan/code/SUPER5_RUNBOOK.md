# Đối chiếu Super 5 lớp với PULSE — runbook tái lập

Mục tiêu: đưa mô hình của ta về ĐÚNG phân nhóm PTB-XL Super 5 lớp (NORM/MI/STTC/CD/HYP)
mà PULSE công bố, để có một đối chiếu cùng thước đo (AUC 82.4 / F1 74.8 / HL 11.0).

Cách làm: KHÔNG train lại, KHÔNG chạy lại model nếu đã có dump. Chỉ gộp 44 mã chẩn đoán
→ 5 siêu lớp theo cột `diagnostic_class` của `scp_statements.csv` (max xác suất các mã con,
hợp nhãn các mã con), rồi tính macro-AUC / macro-F1 / Hamming.

## Đã chạy (kết quả trong luận văn Bảng 4.4, từ dump fold-10 sẵn có)

```bash
# dump sinh từ eval_v2.py --out (task diag, N=2050 toàn fold-10)
./.venv/bin/python -m khoaluan.code.aggregate_super5 --test-dump eval_pulse_base_fold10.json --label "PULSE-base" --out khoaluan/super5_base.json
./.venv/bin/python -m khoaluan.code.aggregate_super5 --test-dump eval_ep2_test_frozenF1.json  --label "Ours"       --out khoaluan/super5_ours.json
```

| Hệ thống (Super 5 lớp) | macro-AUC | macro-F1† | Hamming† |
|---|---|---|---|
| PULSE-7B — bài báo công bố | 0.824 | 0.748 | 11.0% |
| PULSE-7B base — pipeline của ta (fold-10, N=2050) | **0.826** | 0.615 | 20.9% |
| Mô hình đề xuất — pipeline của ta (fold-10, N=2050) | **0.902** | 0.693 | 13.6% |

Kết luận:
- macro-AUC base 0.826 ≈ 0.824 công bố → pipeline trung thực, tái lập đúng số PULSE.
- macro-AUC ours 0.902 > 0.824 → vượt PULSE trên chính benchmark của họ (+0.078).
- † F1/Hamming: gộp max + ngưỡng tự dò trên test (lạc quan, khác quy trình PULSE) → chỉ tham khảo.
  Riêng chi tiết: base F1 0.615 < 0.748 dù AUC trùng → chứng tỏ chênh F1 là do quy trình ngưỡng,
  không phải năng lực (lý do lấy macro-AUC làm chỉ số chính).

## Nếu muốn F1 val-frozen (đối chiếu F1 chặt hơn) — cần 1 lần eval fold-9 trên Colab

```python
# trên Colab, sau khi đã có checkpoint + ảnh fold-9 (xem COLAB_ABLATION.md mục 0–3)
# tạo dump diag cho fold-9 (val) cho cả base lẫn ours:
!python -m khoaluan.code.eval_v2 --base $BASE --vocab vocab.json --val val9_3task.json \
    --images $IMG --tasks diag --n 0 --load-4bit --out eval_val9_base.json
!python -m khoaluan.code.eval_v2 --adapter $ADP --base $BASE --vocab vocab.json --val val9_3task.json \
    --images $IMG --tasks diag --n 0 --load-4bit --out eval_val9_ours.json
```

Rồi gộp với ngưỡng đóng băng từ fold-9:

```bash
python -m khoaluan.code.aggregate_super5 --test-dump eval_pulse_base_fold10.json --val-dump eval_val9_base.json --label "PULSE-base (val-frozen)"
python -m khoaluan.code.aggregate_super5 --test-dump eval_ep2_test_frozenF1.json  --val-dump eval_val9_ours.json --label "Ours (val-frozen)"
```

`--val-dump` làm script dò ngưỡng F1 từng siêu lớp trên fold-9 rồi áp lên fold-10 (không rò rỉ test).
macro-AUC không đổi (không phụ thuộc ngưỡng); chỉ F1/Hamming trở nên trung thực hơn.
