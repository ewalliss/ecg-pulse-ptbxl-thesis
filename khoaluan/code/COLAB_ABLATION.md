# Colab runbook — ablation chứng minh model DÙNG ảnh + kiểm soát leakage

Mục tiêu: trả lời 2 phản biện của thầy Lý Quốc Ngọc.
- **Lỗi 2 (model có dùng ảnh không):** chạy v2 với `--image-mode blank` và `--image-mode shuffle`.
  Nếu macro-AUC tụt về ~0.5 trong khi `real` ~0.876 => mô hình thật sự đọc ảnh ECG.
- **Lỗi 1 (leakage):** chạy `--placeholder one`. Nếu AUC gần như không đổi so với `zero`
  (chênh < 0.005) => điểm số là logit DỰ ĐOÁN tại slot, nhãn thật không vào context.

Chạy trên **v2 (adapter checkpoint-252)**, KHÔNG phải base — vì cần chứng minh chính mô hình cuối.

## 0. Chuẩn bị (upload lên Google Drive trước)
- `checkpoint-252/` (adapter LoRA, ~vài trăm MB) — nén `checkpoint-252.zip`.
- `scp_3task_vocab.json`, `test_3task.json` (chỉ cần test fold-10 cho ablation).
- `eval_images_subset.zip` — chỉ ảnh test (chạy `collect_eval_images.py` với 1 json test là đủ;
  ablation chỉ cần fold-10).
- (Không cần base model: tải từ HF.)

## 1. GPU + clone code
```python
!nvidia-smi
!git clone -b research/pulse-v2 https://<TOKEN>@github.com/ewalliss/graduation_thesis.git
%cd graduation_thesis
```

## 2. Deps
```python
!pip install -q transformers==4.37.2 peft bitsandbytes accelerate sentencepiece scikit-learn pillow
!pip install -q -e source/LLaVA   # llava fork của PULSE; nếu kẹt version -> xem source/LLaVA/pyproject.toml
```

## 3. Lấy data từ Drive
```python
from google.colab import drive; drive.mount('/content/drive')
import zipfile, os
D = '/content/drive/MyDrive/ecg_ablation'   # thư mục bạn upload
for z in ['checkpoint-252.zip', 'eval_images_subset.zip']:
    zipfile.ZipFile(f'{D}/{z}').extractall('/content/')
!cp {D}/scp_3task_vocab.json {D}/test_3task.json /content/
# -> /content/checkpoint-252, /content/eval_images_subset, /content/*.json
```

## 4. Bốn lần eval (dùng --n 500/task cho nhanh & SO SÁNH ĐƯỢC với nhau)
Cùng `--n 500` cho cả 4 để 4 con số AUC cùng cỡ mẫu, đối chiếu trực tiếp.
```python
BASE = 'PULSE-ECG/PULSE-7B'   # KIỂM TRA đúng base mà adapter được train trên đó
ADP  = '/content/checkpoint-252'
COMMON = (f'--adapter {ADP} --base {BASE} --vocab /content/scp_3task_vocab.json '
          f'--val /content/test_3task.json --images /content/eval_images_subset '
          f'--n 500 --load-4bit --dtype float16')

# (a) real — sanity, kỳ vọng ~0.876
!python -m khoaluan.code.eval_v2 {COMMON} --image-mode real    --out eval_abl_real.json
# (b) blank — xoá tín hiệu, kỳ vọng ~0.5
!python -m khoaluan.code.eval_v2 {COMMON} --image-mode blank   --out eval_abl_blank.json
# (c) shuffle — đảo ảnh-nhãn, kỳ vọng ~0.5
!python -m khoaluan.code.eval_v2 {COMMON} --image-mode shuffle --out eval_abl_shuffle.json
# (d) placeholder=one — control leakage, kỳ vọng ≈ (a)
!python -m khoaluan.code.eval_v2 {COMMON} --image-mode real --placeholder one --out eval_abl_phone.json
```

## 5. Tổng hợp bảng
```python
import json, numpy as np
def macro(fp):
    d = json.load(open(fp))
    return {t: round(d[t]['macro_auc'], 3) for t in d}
for name, fp in [('real',   'eval_abl_real.json'),
                 ('blank',  'eval_abl_blank.json'),
                 ('shuffle','eval_abl_shuffle.json'),
                 ('placeholder=1', 'eval_abl_phone.json')]:
    m = macro(fp); mean = round(np.mean(list(m.values())), 3)
    print(f'{name:14s} {m}  mean={mean}')
```

## 6. Đọc kết quả (đưa vào luận văn Bảng 4.x "Nghiên cứu cắt bỏ")
| Điều kiện | macro-AUC TB kỳ vọng | Kết luận |
|---|---|---|
| real | ~0.876 | mốc tham chiếu |
| blank (ảnh xám) | ~0.50 | model SỤP khi mất ảnh => nó dùng ảnh |
| shuffle (sai ảnh) | ~0.50 | dự đoán không còn khớp nhãn => tín hiệu đến từ ảnh đúng |
| placeholder=1 | ~0.876 (chênh <0.005) | điểm độc lập token điền => KHÔNG leakage |

Tải 4 file `eval_abl_*.json` về, commit. Tôi sẽ thêm bảng ablation + 1 đoạn phân tích
vào Chương 4 (Mục 4.4 Nghiên cứu cắt bỏ) của luận văn.

Lưu ý: nếu `blank`/`shuffle` KHÔNG tụt về ~0.5 (vd vẫn 0.7+), đó là tín hiệu model dựa
nhiều vào prior nhãn hơn ảnh — phải báo cáo trung thực và thảo luận, KHÔNG giấu.
