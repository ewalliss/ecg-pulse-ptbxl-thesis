# Phân lớp bệnh tim đa phương thức trên ảnh điện tâm đồ (ECG)

**Từ hỏi–đáp thị giác của PULSE sang bộ phân lớp theo tác vụ trên PTB-XL**

*Multimodal cardiac disease classification on ECG images: from PULSE's visual question answering to a task-decomposed classifier on PTB-XL*

Khóa luận tốt nghiệp — Ngành **Thị giác máy tính** — Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM.

| | |
|---|---|
| **Sinh viên** | Nguyễn Huỳnh Hải Đăng (22127052 — nhhdang22@clc.fitus.edu.vn) |
| | Phan Vũ Gia Hân (22127102 — pvghan22@clc.fitus.edu.vn) |
| **Giảng viên hướng dẫn** | PGS.TS Lý Quốc Ngọc |
| **Khóa đào tạo** | 2022–2026 |
| **Mô hình nền** | PULSE-7B (họ LLaVA-v1.6-Vicuna-7B) |
| **Bộ dữ liệu** | PTB-XL (12 chuyển đạo, ảnh ECG) |

## Bản PDF khóa luận

→ **[`khoaluan/thesis/main.pdf`](khoaluan/thesis/main.pdf)** (biên dịch từ nguồn Typst trong `khoaluan/thesis/`).

## Tóm tắt

Khóa luận chuyển mô hình ngôn ngữ–thị giác PULSE-7B — vốn là một mô hình hỏi–đáp thị giác (VQA) sinh văn bản — thành một bộ phân lớp bệnh tim đa nhãn trên PTB-XL, mà **không phá vỡ lõi mô hình**. Bài toán PTB-XL được phân rã thành ba tác vụ độc lập (chẩn đoán, nhịp, hình thái) với quy tắc gán nhãn hai tầng; giữ nguyên hàm mất mát entropy chéo next-token gốc của PULSE và chỉ tinh chỉnh nhẹ mô hình ngôn ngữ bằng QLoRA 4-bit trên nhánh thị giác đông cứng. Đóng góp về phương pháp đánh giá là **giao thức teacher-forced**: tự dựng khối nhãn và đọc xác suất tại các vị trí token đã biết trong một lượt suy luận, cho phép đo macro-AUC không phụ thuộc ngưỡng một cách nhất quán trên mọi lớp.

## Kết quả chính

Trên tập kiểm thử độc lập **fold-10** của PTB-XL (mô hình tinh chỉnh 2 epoch):

| Tác vụ | macro-AUC |
|---|---|
| Chẩn đoán (44 mã) | 0.893 |
| Nhịp (12 mã) | 0.887 |
| Hình thái (19 mã) | 0.849 |
| **Trung bình** | **0.876** |

- Vượt chính mô hình nền PULSE-7B chưa tinh chỉnh thêm **+0.097 macro-AUC** trong so sánh có kiểm soát cùng giao thức.
- Trên đúng phân nhóm **PTB-XL Super (5 siêu lớp)** mà PULSE công bố: macro-AUC **0.902**, macro-F1 **0.693** (fold-10).

> Lưu ý: macro-AUC là chỉ số chính (không phụ thuộc ngưỡng). Đây là khẳng định về khả năng *phân biệt*, chưa phải một bộ phân lớp đã hiệu chuẩn sẵn sàng lâm sàng.

## Cấu trúc repo

```
.
├── khoaluan/thesis/           # Khóa luận (nguồn Typst + main.pdf)
│   ├── main.typ               #   điểm vào: typst compile main.typ main.pdf
│   ├── chapters/              #   các chương
│   └── cosolythuyet/, *.png   #   hình minh hoạ
├── khoaluan/code/             # Mã của khóa luận (lớp dữ liệu + đánh giá)
│   ├── build_dataset.py       #   sinh dữ liệu 3-task định dạng LLaVA
│   ├── labels.py, scp_tasks.py#   quy tắc gán nhãn hai tầng
│   ├── train_v2.py            #   launcher huấn luyện (CE gốc PULSE, KHÔNG ASL)
│   ├── teacher_forced.py      #   giao thức đánh giá teacher-forced
│   ├── eval_v2.py             #   chấm macro-AUC / macro-F1 theo tác vụ
│   ├── aggregate_super5.py    #   gộp 44 mã → 5 siêu lớp (đối chiếu PULSE)
│   └── rerender_images_noid_colab.ipynb  # render lại ảnh ECG (Colab)
├── khoaluan/paper/, khoaluan/checks/   # bản thảo bài báo + script kiểm tra
├── khoaluan/*.json            # số liệu kết quả (super5_base/ours, ablation)
├── preprocessing_pipeline/    # Tiền xử lý: WFDB → lọc → render ảnh ECG → nhãn
├── source/LLaVA/              # Stack LLaVA MƯỢN từ PULSE (engine train/suy luận)
├── tests/                     # Kiểm thử
├── REPRODUCE.md               # Hướng dẫn tái lập kết quả
├── WINDOWS_SETUP.md           # Thiết lập môi trường (Windows)
└── pyproject.toml, uv.lock    # Phụ thuộc
```

## Biên dịch khóa luận

Yêu cầu [Typst](https://github.com/typst/typst).

```bash
cd khoaluan/thesis
typst compile main.typ main.pdf
```

## Tái lập kết quả

Xem **[`REPRODUCE.md`](REPRODUCE.md)**. Tóm tắt đường chạy: sinh dữ liệu 3-task (`khoaluan/code/build_dataset.py`) → huấn luyện QLoRA bằng engine của PULSE (`khoaluan/code/train_v2.py`, gọi `llava.train.train`) → đánh giá teacher-forced (`khoaluan/code/eval_v2.py`) → gộp Super (`khoaluan/code/aggregate_super5.py`).

## Nguồn gốc & tái sử dụng (provenance)

- **`source/LLaVA/`** là bản fork LLaVA của **PULSE** (`https://github.com/AIMedLab/PULSE`, vốn dựa trên LLaVA-NeXT của haotian-liu). Đây là **engine huấn luyện và suy luận** mà khóa luận chạy trực tiếp lên; phần lớn giữ nguyên byte.
- Khóa luận **giữ nguyên objective huấn luyện** (entropy chéo next-token gốc của PULSE). Các bổ sung ASL-Gen (`asl_gen_loss.py` + hook trong `train.py`/`llava_trainer.py`) là di sản của một thử nghiệm đã bỏ; chúng **bị khóa mặc định** (`aslgen_enable=False`) và **không dùng ở mô hình cuối**. Các thay đổi còn hiệu lực ở đường v2 chỉ gồm vài sửa tương thích tối thiểu (dtype projector 4-bit, shim `num_items_in_batch`).
- Mô hình nền PULSE-7B và bộ dữ liệu PTB-XL là tài nguyên công khai của các tác giả gốc.

## Dữ liệu

Bộ **PTB-XL** không được đính kèm trong repo (kích thước lớn, bản quyền theo PhysioNet). Tải tại: <https://physionet.org/content/ptb-xl/>. Đặt đường dẫn qua biến môi trường `ECGLLAVA_DATASET_ROOT` (xem `preprocessing_pipeline/config/preprocessing_config.py`).

## Lời cảm ơn

Cảm ơn PGS.TS Lý Quốc Ngọc đã hướng dẫn; các tác giả của **PTB-XL**, **PULSE** và **LLaVA** đã công khai tài nguyên phục vụ nghiên cứu.
