# Bằng chứng bảo vệ — tổng hợp toàn bộ

Cập nhật: 2026-06-20. Branch `research/pulse-v2`. Mô hình cuối: **checkpoint-312** (epoch-2, PULSE-7B + QLoRA). Tất cả số đo theo giao thức teacher-forced, split chuẩn Strodthoff (fold 1–8 train, fold 9 val, fold 10 test).

## 1. Kết quả headline — fold-10 (test độc lập), epoch-2
| Tác vụ | Số lớp có AUC | macro-AUC | macro-F1 (val-frozen) | mean P(1) +/− |
|---|---|---|---|---|
| Chẩn đoán (44) | 44/44 | 0.893 | 0.24 | 0.44 / 0.03 |
| Nhịp (12) | 11/12 | 0.887 | 0.33 | 0.64 / 0.06 |
| Hình thái (19) | 19/19 | 0.849 | 0.22 | 0.35 / 0.05 |
| **Trung bình** | – | **0.876** | **0.26** | – |

- macro-AUC = chỉ số chính (không phụ thuộc ngưỡng), lấy trung bình trên các lớp có ≥1 mẫu dương trên fold-10 (nhịp 11/12: 1 lớp không có mẫu dương → AUC không xác định, bị loại).
- macro-F1: ngưỡng per-class hiệu chuẩn trên fold-9 (val) rồi **đóng băng** khi chấm fold-10 → không rò rỉ test; F1 thấp do đuôi lớp hiếm, KHÔNG so với F1 paper ngoài.
- File: `eval_ep2_test_frozenF1.json`.

## 2. So sánh có kiểm soát với PULSE-base (cùng fold-10, cùng giao thức)
| Tác vụ | PULSE-base zero-shot | v2 (epoch-2) | Δ |
|---|---|---|---|
| Chẩn đoán | 0.835 | 0.893 | +0.058 |
| Nhịp | 0.774 | 0.887 | +0.113 |
| Hình thái | 0.728 | 0.849 | +0.121 |
| **TB (macro-AUC)** | **0.779** | **0.876** | **+0.097** |
| macro-F1 (val-frozen) | ~0.19† | 0.26 | — |

- +0.097 macro-AUC = đóng góp thuần của fine-tune (so sánh tương đương DUY NHẤT). File base: `eval_pulse_base_fold10.json`.
- † base macro-F1 val-frozen: diag 0.168 / nhịp 0.265 / hình thái 0.148, TB **0.194** — đo trên N=150 mẫu/tác vụ (Colab T4; file Drive `eval_base_test_frozenF1.json`). v2 0.26 đo trên FULL fold-10. Cùng hướng (fine-tune nâng F1 cả 3 task) nhưng KHÁC cỡ mẫu. Để có cột F1 base full-fold đúng chuẩn cho Bảng 4.3, chạy trên V100 (Mục 6).

## 3. Nghiên cứu cắt bỏ — mô hình có dùng ảnh ECG & có rò rỉ nhãn không?
checkpoint-312, fold-10, N=80 mẫu/tác vụ, Colab T4, 4-bit teacher-forced. File: `eval_abl_results.json`.

| Điều kiện | Chẩn đoán | Nhịp | Hình thái | TB |
|---|---|---|---|---|
| **real** (ảnh đúng) | 0.936 | 0.968 | 0.931 | **0.945** |
| **blank** (ảnh xám) | 0.500 | 0.500 | 0.500 | **0.500** |
| **shuffle** (ảnh sai) | 0.454 | 0.366 | 0.461 | **0.427** |
| **placeholder = "1"** | 0.833 | 0.722 | 0.839 | **0.798** |

**Diễn giải (chốt 2 phản biện):**
- **Model thực sự dùng ảnh ECG:** xoá ảnh (blank) hoặc gán sai ảnh (shuffle) → macro-AUC sụp từ 0.945 về mức ngẫu nhiên (0.50 / 0.43) ở cả ba tác vụ. Không phải đoán theo prior nhãn.
- **Không rò rỉ nhãn (label leakage):** khối nhãn answer GIỮ NGUYÊN hệt nhau giữa real/blank/shuffle, chỉ ảnh đổi. Nếu điểm số rút từ nhãn nằm trong khối answer thì blank/shuffle vẫn phải cao — đằng này sụp về ngẫu nhiên ⇒ toàn bộ tín hiệu phân biệt đến từ ẢNH, không từ khối nhãn.
- **placeholder = "1":** đổi token điền ở mọi ô từ "0"→"1" vẫn cho AUC 0.798 (>> 0.50) ⇒ điểm là phân phối DỰ ĐOÁN của mô hình, không phải sao chép token điền. Mức giảm 0.945→0.798 là do tiền tố toàn "1" là ngữ cảnh tự hồi quy bất thường (lệch hiệu chuẩn), KHÔNG phải rò nhãn thật (nhãn thật của lớp đang chấm không bao giờ vào chuỗi đầu vào dưới bất kỳ token điền nào).
- Lưu ý: real ở đây 0.945 cao hơn headline 0.876 chỉ vì N=80/task (tập con, lớp hiếm bị loại); ablation chỉ cần KHOẢNG CÁCH real-vs-blank nên không ảnh hưởng kết luận.

## 4. Cơ chế chống-leakage (giải thích logic, không cần số)
Ô nhãn của mọi lớp luôn điền một token cố định ("0"), độc lập nhãn thật; xác suất đọc tại logit DỰ ĐOÁN ngay TRƯỚC ô đó. Khối nhãn là như nhau cho mọi mẫu → không mang một bit thông tin nào về nhãn thật của mẫu cụ thể. Eval một forward pass, không sinh tự do → 0% mẫu hỏng.

## 5. Mốc tham chiếu (KHÔNG tương đương — chỉ tham khảo)
- Strodthoff CNN 1-D (tín hiệu, trần bài toán): diag 0.937 / nhịp 0.957 / hình thái 0.896.
- PTB-XL Super 5 lớp (DỄ hơn 75 mã): PULSE 0.824, GEM 0.834, GPT-4o 0.556, LLaVA-1.6 0.500.
- Chỉ so macro-AUC ở mức ý nghĩa; v2 đánh giá full 75 mã (khó hơn Super 5 lớp nhiều).

## 6. Còn lại (tùy chọn)
- [ ] **F1 PULSE-base** (val-frozen, full fold-10) để điền cột F1 Bảng 4.3. Chạy trên Windows V100 (đã hồi):
```powershell
$env:PYTHONIOENCODING="utf-8"; chcp 65001
$B="C:\pedestrian_detection\graduation_thesis\model"
$V=".\v2\runs\v2-run1\data\scp_3task_vocab.json"
$Va=".\v2\runs\v2-run1\data\val_3task.json"; $T=".\v2\runs\v2-run1\data\test_3task.json"; $I=".\data\pulse_ptbxl_stage2\images"
# base trên val -> ngưỡng
python -m khoaluan.code.eval_v2 --base $B --vocab $V --val $Va --images $I --n 0 --load-4bit --dtype float16 --out eval_base_val.json
# base trên test, đóng băng ngưỡng val
python -m khoaluan.code.eval_v2 --base $B --vocab $V --val $T --images $I --thresholds eval_base_val.json --n 0 --load-4bit --dtype float16 --out eval_base_test_frozenF1.json
```
(Chạy full fold-10 chỉ khả thi trên V100; Colab T4 quá chậm cho full ~2050 mẫu/task.)
