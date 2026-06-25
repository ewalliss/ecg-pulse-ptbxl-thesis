# Tái lập (Reproducibility) — bản đồ code và quy trình end-to-end

Toàn bộ code cần để tái lập **nằm trong CHÍNH repo này** (một repo duy nhất), và được tổ chức
thành đúng HAI khối:

- **Khối dùng lại từ PULSE (phần lớn):** `source/LLaVA/` — fork LLaVA của PULSE, 151 file. Đây là
  toàn bộ stack huấn luyện + suy luận. Huấn luyện gọi THẲNG `llava.train.train` (xem §3), suy luận
  nạp model qua `llava.*` (anyres, builder, LlavaLlama). **Khóa luận KHÔNG sửa lõi này.**
- **Khối code của khóa luận (lớp mỏng):** `khoaluan/code/` — TẤT CẢ code tự viết nằm gọn trong một thư mục:
  sinh dữ liệu, quy tắc nhãn, prompt, launcher train, eval teacher-forced, gộp Super, loader model.

Nói cách khác: ta **dùng lại gần như toàn bộ PULSE** và chỉ bổ sung một lớp mỏng dữ liệu + đánh giá.

## 1. Code nằm ở đâu (đã kiểm: không import nào trỏ ra ngoài repo)

| Vai trò | Đường dẫn | Ghi chú |
|---|---|---|
| Nền dùng lại (LLaVA/PULSE fork) | `source/LLaVA/` | 151 file, `pip install -e`. Cung cấp `llava.*`; KHÔNG sửa. |
| Launcher train | `khoaluan/code/train_v2.py` | Gọi `from llava.train.train import train` với cờ native + CE gốc PULSE (không ASL). |
| Sinh dữ liệu 3-task | `khoaluan/code/build_3task_json.py`, `scp_tasks.py`, `labels.py`, `build_dataset.py`, `prompts.py` | Đọc `scp_statements.csv`, quy tắc nhãn 2 tầng. |
| Eval teacher-forced (3 task) | `khoaluan/code/eval_v2.py`, `teacher_forced.py` | macro-AUC/F1 theo từng task. |
| Loader model (4-bit + LoRA) | `khoaluan/code/infer_aslgen.py` | `load_model` mà `eval_v2.py` dùng (đã gom về `khoaluan/code/`). |
| Gộp Super 5 lớp | `khoaluan/code/aggregate_super5.py` | Hậu xử lý dump → so với PULSE (Bảng 4.4–4.6). |
| Ablation weighted-CE (tùy chọn) | `khoaluan/code/weighted_ce.py` | Không nằm trên đường train chính. |

Phụ thuộc Python: `pyproject.toml` (gốc, gói `ecg-llava`, Python ≥ 3.12) + danh sách pip trong
`khoaluan/code/COLAB_ABLATION.md` mục 2 (`transformers==4.37.2 peft bitsandbytes accelerate
sentencepiece scikit-learn pillow` và `pip install -e source/LLaVA`).

**Code KHÔNG thuộc đường tái lập v2 (legacy v1, để tham khảo):**
- `src/` — pipeline tiền xử lý + model của phiên bản v1 (renderer, scp_parser, `qlora_config`,
  `ecg_llava`, `anyres_tiler`…). **v2 KHÔNG import `src/`** (đã kiểm: `khoaluan/code` và loader không
  dùng `src`); chỉ `tests/` và vài script v1 cũ còn gọi nó. `src/model/anyres_tiler.py` là DEAD CODE.
- `testing_inference/eval_scored.py` — eval ASL-gen của v1; đường eval chính của v2 là `eval_v2.py`.
- `PULSE/` (clone repo gốc PULSE ở thư mục gốc) — KHÔNG file nào tham chiếu; đã gitignore.

## 2. Cái KHÔNG nằm trong git (lấy từ nguồn ngoài)

| Hạng mục | Lấy ở đâu |
|---|---|
| Trọng số nền PULSE-7B | Hugging Face (`PULSE-ECG/PULSE-7B`) |
| Dataset PTB-XL 1.0.3 | PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/ |
| Checkpoint LoRA đã train | Google Drive (xem `khoaluan/code/COLAB_ABLATION.md`) |
| Papers tham khảo | `papers_backup.zip` trên Google Drive (ngoài git vì bản quyền) |

## 3. Quy trình end-to-end

```bash
# 0) Môi trường
pip install -e source/LLaVA
pip install transformers==4.37.2 peft bitsandbytes accelerate sentencepiece scikit-learn pillow pandas

# 1) Sinh dữ liệu 3-task từ PTB-XL (split chuẩn Strodthoff: fold 1-8 train / 9 val / 10 test)
export ECGLLAVA_DATASET_ROOT=/path/to/ptb-xl-1.0.3
python -m khoaluan.code.build_3task_json \
    --db   /path/to/ptb-xl-1.0.3/ptbxl_database.csv \
    --images <img_dir> \
    --out  khoaluan/runs/data \
    --cap-majority 1500           # giới hạn lớp đa số 1500/task (xem §4.1); tạo *_3task.json + scp_3task_vocab.json

# 2) Train (CE gốc PULSE, QLoRA r=16/alpha=32, 2 epoch, đông cứng nhánh thị giác)
python -m khoaluan.code.train_v2 --data khoaluan/runs/data --model PULSE-ECG/PULSE-7B \
    --output khoaluan/runs/ckpt --dry-run        # in lệnh train chính xác; bỏ --dry-run để chạy thật

# 3) Eval teacher-forced trên fold-10 (cả base lẫn fine-tuned), dump scores+gt
python -m khoaluan.code.eval_v2 --base PULSE-ECG/PULSE-7B --vocab khoaluan/runs/data/scp_3task_vocab.json \
    --val khoaluan/runs/data/test_3task.json --images <img_dir> --n 0 --load-4bit \
    --out eval_pulse_base_fold10.json                      # base
python -m khoaluan.code.eval_v2 --adapter khoaluan/runs/ckpt/checkpoint-XXX --base PULSE-ECG/PULSE-7B \
    --vocab khoaluan/runs/data/scp_3task_vocab.json --val khoaluan/runs/data/test_3task.json \
    --images <img_dir> --n 0 --load-4bit --out eval_ep2_test_frozenF1.json   # ours

# 4) Gộp 44 mã chẩn đoán -> 5 siêu lớp Super, so với PULSE (Bảng 4.4-4.6)
python -m khoaluan.code.aggregate_super5 --test-dump eval_pulse_base_fold10.json --label "PULSE-base"
python -m khoaluan.code.aggregate_super5 --test-dump eval_ep2_test_frozenF1.json  --label "Ours"
```

Các dump mẫu (`eval_pulse_base_fold10.json`, `eval_ep2_test_frozenF1.json`, `eval_fold10_test.json`)
đã commit sẵn nên có thể chạy thẳng bước 4 mà không cần GPU.

## 4. Tính tất định

Đánh giá teacher-forced chỉ một forward pass, KHÔNG lấy mẫu → cùng đầu vào cho cùng kết quả,
không phụ thuộc seed sinh. macro-AUC không phụ thuộc ngưỡng. Hạn chế đã biết: ngưỡng F1/lâm sàng
ở phần Super (Bảng 4.5/4.6) hiện self-tuned trên test (lạc quan); để hết rò rỉ cần một lần
eval fold-9 rồi đóng băng ngưỡng (`aggregate_super5.py --val-dump`, xem `khoaluan/code/SUPER5_RUNBOOK.md`).
