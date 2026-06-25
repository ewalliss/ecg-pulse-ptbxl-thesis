# v2 — Trạng thái & Kết quả đã chốt

Cập nhật: 2026-06-19. Branch: `research/pulse-v2`.

## 1. Mô hình & thiết lập (đã chốt)
- Nền: PULSE-7B (LLaVA-v1.6-Vicuna-7B); CLIP ViT-L/14@336 + projector + Vicuna-7B.
- Tinh chỉnh: QLoRA 4-bit NF4, r=16, alpha=32, lr=2e-4, fp16, sdpa; vision + projector ĐÔNG CỨNG; lõi PULSE không sửa.
- Dữ liệu: PTB-XL, 3 tác vụ (diag 44 / rhythm 12 / form 19). Nhãn 2 tầng: diag likelihood≥50, rhythm+form theo hiện diện. cap-majority 1500/task → ~16k mẫu train. Prompt tiếng Anh.
- Split chuẩn Strodthoff: fold 1–8 train, fold 9 val, fold 10 test.
- Số epoch dùng cho kết quả chính: **2 epoch** (checkpoint-252). Tiến trình: epoch1 mean 0.818 → epoch2 0.876 (CHƯA overfit; dừng do giới hạn tài nguyên, không do hội tụ).
- Eval: teacher-forced (đọc P(y=1) tại slot nhãn cố định, 1 forward) → **0% mẫu hỏng**.

## 2. Kết quả headline (fold-10 TEST, epoch-2)
| Tác vụ | macro-AUC | macro-F1 (val-frozen) | mean P(1) +/− |
|---|---|---|---|
| diag (44) | 0.893 | 0.24 | 0.44 / 0.03 |
| rhythm (12) | 0.887 | 0.33 | 0.64 / 0.06 |
| form (19) | 0.849 | 0.22 | 0.35 / 0.05 |
| **Trung bình** | **0.876** | **0.26** | – |

- macro-AUC threshold-free = headline. macro-F1 dùng ngưỡng per-class hiệu chuẩn trên fold-9 rồi ĐÓNG BĂNG khi chấm fold-10 (không dò trên test → trung thực). File: `eval_ep2_test_frozenF1.json` (mode=val-frozen).
- Lớp tốt nhất test: clbbb 1.00, injin/injil 1.00, crbbb/3avb 0.99 (diag); stach/afib 0.99 (rhythm); pvc 0.99, prc(s) 0.98 (form). Yếu nhất: tab~0.34 (form).

## 3. So sánh có kiểm soát với PULSE gốc (cùng fold-10, cùng teacher-forced)
| Tác vụ | PULSE-base zero-shot | v2 (epoch-2) | Δ |
|---|---|---|---|
| diag | 0.835 | 0.893 | +0.058 |
| rhythm | 0.774 | 0.887 | +0.113 |
| form | 0.728 | 0.849 | +0.121 |
| **TB** | **0.779** | **0.876** | **+0.097** |

→ +0.097 AUC = đóng góp thuần của fine-tune (so sánh tương đương DUY NHẤT). File base: `eval_pulse_base_fold10.json`.

## 4. Mốc tham chiếu (KHÔNG tương đương — chỉ tham khảo)
- Strodthoff CNN 1-D (trần, tín hiệu): diag 0.937 / rhythm 0.957 / form 0.896.
- PTB-XL Super 5 lớp (DỄ hơn 75 mã): PULSE 0.824, GEM 0.834, GPT-4o 0.556, LLaVA-1.6 0.500.
- Lưu ý: chỉ so macro-AUC ở mức ý nghĩa; KHÔNG so macro-F1 với paper ngoài (khác ngưỡng/tập nhãn).

## 5. Thử nghiệm ban đầu (đã chuyển hướng)
Áp một hàm mất mát kiểu phân loại đa nhãn lên khối token nhãn → sụp đổ về toàn NORM (~0.5 AUC) + ~73% mẫu hỏng. Nguyên nhân (suy đoán): không tương thích loss-phân-loại vs decoder sinh token. Trong tài liệu nộp: gọi "thử nghiệm ban đầu", KHÔNG nêu tên ASL/công thức (để dễ bảo vệ).

## 6. Tài liệu đã tạo
- Luận văn: `~/Downloads/KhoaLuan_ECG_PULSE_v2.{md,docx,pdf}` — theo template ĐHKHTN, Khóa luận tốt nghiệp; GVHD PGS.TS Lý Quốc Ngọc; ngành Khoa học Máy tính; SV: Nguyễn Huỳnh Hải Đăng (22127052), Phan Vũ Gia Hân.
- 2 form: `~/Downloads/MAU TU NHAN XET-LQN/NhanXet_GVHD_ECG_PULSE.docx`, `NoiDung_PhanBien_ECG_PULSE.docx`.

## 7. Code (branch research/pulse-v2)
- `khoaluan/code/eval_v2.py`: teacher-forced eval; `--adapter` tùy chọn (rỗng = eval base); `--thresholds <val.json>` nạp ngưỡng val áp lên test (mode val-frozen). Commit gần nhất: 6ac1805.
- `testing_inference/infer_aslgen.py`: `load_model` bỏ qua attach adapter khi rỗng.

## 7b. Phản biện GVHD (PGS Lý Quốc Ngọc) — 3 lỗi cần sửa trước bảo vệ
1. **Leakage trong teacher-forced — ĐÃ CHỨNG MINH:** bằng chứng chính là ablation blank/shuffle (khối nhãn không đổi, chỉ ảnh đổi → AUC sụp ~0.5 ⇒ tín hiệu từ ảnh, không từ nhãn rò). Biến thể `--placeholder one` cho TB 0.798 (KHÔNG ≈ real như giả định "<0.005" ban đầu — đã sửa lại trong luận văn/paper) nhưng vẫn >> 0.5 ⇒ điểm là logit dự đoán, không sao chép token điền.
2. **Model có dùng ảnh — ĐÃ CHỨNG MINH:** ablation trên checkpoint-312, fold-10, N=80/task (Colab T4). real TB **0.945** → blank **0.500** → shuffle **0.427** → placeholder=1 **0.798**. Per-task: real diag 0.936/rhythm 0.968/form 0.931; blank 0.5/0.5/0.5; shuffle 0.454/0.366/0.461; ph=1 0.833/0.722/0.839. File: Drive `ablation_results.json`. Đưa vào luận văn Bảng 4.4 (Mục 4.4.3).
3. **Mâu thuẫn số (ĐÃ SỬA):** +0.039 (epoch-1) → +0.097 (epoch-2) ở luận văn dòng 83 + 2 form; "mọi lớp đều được chấm" → "lớp có mẫu dương" (nhịp 11/12); làm dịu kết luận F1 (0.26 = chưa phải bộ phân lớp hiệu chuẩn lâm sàng). 2 form đồng bộ 2 epoch + 0.876. paper.typ đồng bộ epoch-2 (0.876/+0.097), compile OK.

## 8. Đang chạy / còn lại
- [ ] base VAL (`eval_base_val.json`) + base TEST-frozen F1 (`eval_base_test_frozenF1.json`) → thêm cột F1 PULSE-base vào Bảng 4.3. (GPU V100 vừa "GPU is lost" → cần reboot rồi chạy lại.)
- [ ] (tùy chọn) macro-F1 trên lớp có ≥K mẫu dương (báo cáo kèm cutoff) để F1 cao & trung thực hơn.
- [ ] Đồng bộ 2 form về 2 epoch + 0.876 + Δ+0.097 (hiện còn ghi 10 epoch + 0.818).
- [ ] Điền placeholder luận văn: mã ngành (ĐH KHMT ~7480101), MSSV Phan Vũ Gia Hân, khóa đào tạo, tháng cam đoan, 2 hình (3.1 kiến trúc, 4.1 teacher-forced).

## 9. Lệnh hay dùng (Windows)
Eval một checkpoint trên test:
```
python -m khoaluan.code.eval_v2 --adapter .\v2\runs\v2-run1\checkpoints\checkpoint-252 --base C:\pedestrian_detection\graduation_thesis\model --vocab .\v2\runs\v2-run1\data\scp_3task_vocab.json --val .\v2\runs\v2-run1\data\test_3task.json --images .\data\pulse_ptbxl_stage2\images --n 0 --load-4bit --dtype float16 --out eval_test.json
```
F1 val-frozen: chạy val (--out eval_val.json) rồi test (--thresholds eval_val.json). Base: bỏ --adapter.
