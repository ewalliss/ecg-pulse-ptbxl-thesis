# Giải thích các công thức trong luận văn (kèm ánh xạ vào code thật)

Tài liệu hỗ trợ bảo vệ. Mỗi công thức gồm: (a) dạng công thức, (b) ý nghĩa bằng lời,
(c) định nghĩa biến, (d) chỗ tương ứng trong code, (e) trực giác + câu hỏi hội đồng dễ hỏi.

Đã đối chiếu: tất cả công thức KHỚP với mô hình train/eval thực tế trong code.

---

## 1. Ký hiệu đầu vào / đầu ra (Mục 1.3)

**Công thức:**
- Ảnh ECG: `I ∈ ℝ^(H×W×3)`
- Câu hỏi tác vụ: `q_t`, với `t ∈ {chẩn đoán, nhịp, hình thái}`
- Từ vựng mã của tác vụ: `V_t = {c_1, …, c_{K_t}}`, với `K_chẩn-đoán = 44`, `K_nhịp = 12`, `K_hình-thái = 19`

**Ý nghĩa:** Mỗi mẫu đầu vào là một cặp (ảnh ECG, câu hỏi văn bản). Bài toán được phân rã
thành 3 tác vụ độc lập; mỗi tác vụ có một danh sách mã riêng (44/12/19).

**Lưu ý chồng lấn (để trả lời hội đồng):** tổng 71 mã SCP của PTB-XL = 44 chẩn đoán + 12
nhịp + 19 hình thái, nhưng ba nhóm KHÔNG rời nhau hoàn toàn: có 4 mã vừa thuộc nhóm chẩn
đoán vừa thuộc nhóm hình thái (nên 44+12+19 = 75, trừ 4 chồng lấn = 71 mã duy nhất)
[Wagner 2020]. Ta xử lý mỗi tác vụ như một từ vựng riêng (mã chồng lấn xuất hiện ở cả hai
tác vụ liên quan) — đây là lựa chọn mô hình hóa, không phải phân hoạch tách rời.

**Biến:**
- `I`: ảnh ECG 12 chuyển đạo đã kết xuất thành RGB (H×W×3 kênh màu).
- `q_t`: chuỗi token văn bản (tiếng Anh) mô tả tác vụ `t`.
- `V_t`: tập mã SCP của tác vụ; `c_k` là mã thứ k; `K_t` là số mã.

**Code:** `khoaluan/code/build_3task_json.py` (sinh mẫu 3 tác vụ), `khoaluan/code/scp_tasks.py`
(`load_task_vocabs` đọc 44/12/19 từ `scp_statements.csv`), `khoaluan/code/prompts.py`
(`TASK_QUESTION` = `q_t`).

**Trực giác:** không gắn một "đầu phân loại" ngoài kiến trúc; thay vào đó hỏi mô hình bằng
ngôn ngữ tự nhiên, đúng cách PULSE đã được huấn luyện.

---

## 2. Xác suất mỗi mã — công thức (1.1)

**Công thức:** `p_k = P(y_k = 1 | I, q_t) ∈ [0, 1]`

**Ý nghĩa:** Với mỗi mã `c_k`, mô hình ước lượng xác suất mã đó HIỆN DIỆN trên ảnh ECG,
có điều kiện vào ảnh `I` và câu hỏi `q_t`. Đây là đầu ra mềm (soft), nằm trong [0,1].

**Biến:** `y_k ∈ {0,1}` là nhãn thật (1 = có, 0 = không) của mã `c_k`; `p_k` là xác suất
dự đoán.

**Code:** giá trị `p_k` được tính bởi giao thức teacher-forced — xem công thức 4.
Trong `khoaluan/code/eval_v2.py`, biến `score_rows` chính là các `p_k`.

**Trực giác:** vì là xác suất liên tục, ta đo được macro-AUC (không phụ thuộc ngưỡng) —
chỉ số chính của luận văn.

---

## 3. Quyết định nhị phân — công thức (1.2)

**Công thức:** `ŷ_k = 𝟙[p_k ≥ τ_k]`

**Ý nghĩa:** Để ra quyết định "có/không", so xác suất `p_k` với một ngưỡng riêng cho từng
lớp `τ_k`: nếu `p_k ≥ τ_k` thì dự đoán dương (`ŷ_k = 1`), ngược lại âm.

**Biến:** `𝟙[·]` là hàm chỉ thị (1 nếu điều kiện đúng, 0 nếu sai); `τ_k` là ngưỡng của
lớp `c_k`, hiệu chuẩn trên tập kiểm định fold-9 rồi đóng băng khi đánh giá fold-10.

**Code:** `khoaluan/code/eval_v2.py` — `(scores[:, c] >= bt_c)`, với `bt_c` = `τ_k` nạp từ
`per_class_threshold` (val-frozen). Khớp dấu `≥`.

**Trực giác:** macro-F1 phụ thuộc lựa chọn `τ_k` này; còn macro-AUC thì không → đó là lý do
lấy AUC làm chỉ số chính. Ngưỡng per-class vì các lớp hiếm cần điểm cắt khác nhau.

---

## 4. Đọc xác suất kiểu teacher-forced (Mục 3) — công thức cốt lõi

**Công thức:** `P(y_k = 1) = exp(z_1) / (exp(z_0) + exp(z_1))`

**Ý nghĩa:** Thay vì để mô hình SINH tự do rồi phân tích văn bản, ta tự dựng sẵn khối nhãn
`code: 0` cho mọi mã, rồi tại đúng vị trí ô chữ số ta đọc logit dự đoán của mô hình cho
token "1" và token "0", và lấy softmax HAI CHIỀU giữa đúng hai logit đó.

**Biến:**
- `z_1`: logit mô hình gán cho token "1" tại vị trí ô nhãn của mã `c_k`.
- `z_0`: logit cho token "0" tại cùng vị trí.
- Chỉ dùng đúng 2 logit này (không phải softmax trên toàn bộ 32000 từ vựng).

**Code:** `khoaluan/code/teacher_forced.py`, hàm `score_task`:
```python
two = torch.stack([z[neg_id], z[pos_id]]).float()   # [z_0, z_1]
p1.append(torch.softmax(two, dim=0)[1].item())       # exp(z_1)/(exp(z_0)+exp(z_1))
```
`pos_id` = id token " 1", `neg_id` = id token " 0" (`label_token_ids` — lấy subtoken cuối
của " 1"/" 0"). Logit `z` đọc tại vị trí NGAY TRƯỚC ô chữ số (vị trí dự đoán token kế tiếp).

**Lưu ý về tên gọi (quan trọng cho hội đồng):** "teacher-forced" ở đây là tên gọi vay mượn.
Đúng theo học thuật, teacher forcing nguyên gốc [Williams & Zipser 1989] là kỹ thuật khi
HUẤN LUYỆN: nạp token nhãn thật làm đầu vào bước kế thay cho token mô hình tự sinh. Cái ta
làm là ở thì SUY LUẬN: đặt sẵn khung `code: 0` cố định rồi đọc xác suất tại ô trả lời —
đúng bản chất là *forced-choice scoring* / *restricted-vocabulary scoring* (chấm điểm trên
từ vựng giới hạn, cùng họ với constrained decoding mà các benchmark trắc nghiệm LM dùng).
Lý do gọi "teacher-forced" là vì chuỗi đầu vào tới ngay trước ô chấm điểm bị ÉP cố định
(không cho mô hình tự sinh). Khi bảo vệ nên nói rõ: đây là đọc xác suất ở thì suy luận trên
từ vựng giới hạn, KHÁC với teacher forcing thì huấn luyện; khung điền là hằng số "0", độc lập
nhãn thật, nên không có rò rỉ.

**Lưu ý về softmax 2 chiều:** `exp(z_1)/(exp(z_0)+exp(z_1))` là xác suất CÓ ĐIỀU KIỆN
`P(token="1" | token ∈ {"0","1"})` — tức chuẩn hóa lại chỉ trên hai lựa chọn, BỎ toàn bộ
khối lượng xác suất của các token khác (yes/Yes/khoảng trắng/chữ số khác). Đây không phải
xác suất biên thật mà mô hình gán cho "1"; nó chỉ hiệu chuẩn cục bộ cho quyết định nhị phân.
Giả định ngầm: tại ô nhãn, mô hình gần như chỉ phân vân giữa "1" và "0" (đúng vì khung được
huấn luyện theo định dạng đó), nên phần khối lượng bỏ đi là nhỏ.

**Trực giác / câu hỏi hội đồng:**
- "Sao luôn đủ điểm cho mọi mã?" → vì vị trí ô nhãn do ta đặt trước, không phụ thuộc việc
  mô hình sinh ra gì; nên không bao giờ thiếu/hỏng (đặc tính thiết kế đo, không phải năng
  lực sinh).
- "Có rò rỉ nhãn không?" → token điền vào mọi ô luôn cố định ("0"), độc lập nhãn thật; ablation
  blank/shuffle/placeholder=1 (Bảng 4.7) xác nhận điểm đến từ ẢNH, không từ token điền.
- "Sao softmax 2 chiều?" → bài toán mỗi ô là nhị phân "1" vs "0"; chuẩn hóa giữa đúng hai
  logit cho xác suất hiệu chuẩn cục bộ cho quyết định nhị phân đó (xem lưu ý ở trên).

---

## 5. Quy tắc gán nhãn hai tầng (Mục 3.4)

**Công thức (luật):**
- Chẩn đoán: `y_k = 1` nếu mã `c_k` có mặt VÀ `likelihood(c_k) ≥ 50`
- Nhịp / Hình thái: `y_k = 1` nếu mã `c_k` CÓ MẶT (bất kể likelihood)

**Ý nghĩa:** PTB-XL gán mỗi mã một giá trị likelihood 0–100, nhưng chỉ NHÓM CHẨN ĐOÁN mới
có likelihood có nghĩa; nhóm nhịp và hình thái luôn bị gán likelihood = 0. Nếu áp ngưỡng
≥50 đồng loạt, toàn bộ nhịp/hình thái sẽ bị loại sạch (lỗi của phiên bản trước). Quy tắc
hai tầng sửa đúng lỗi này.

**Biến:** `likelihood(c_k)` = giá trị trong trường `scp_codes` của PTB-XL; ngưỡng 50 là
`DEFAULT_DIAG_THRESHOLD`.

**Code:** `khoaluan/code/labels.py`, hàm `assign_labels`:
```python
elif rule == "threshold":   # diag
    out.append(1 if float(scp_codes[code]) >= diag_threshold else 0)
else:                       # presence (rhythm/form)
    out.append(1)
```
Luật `LABEL_RULE = {"diag":"threshold","rhythm":"presence","form":"presence"}` ở `scp_tasks.py`.

**Trực giác:** đây là sửa lỗi dữ liệu cốt lõi của luận văn, dựa trên quy ước PTB-XL [Wagner 2020].

---

## 6. Bộ điều hợp QLoRA (Hình 3.1)

**Công thức:** `h = W_0 x + (α / r) · B A x`

**Ý nghĩa:** Trọng số gốc `W_0` của mô hình ngôn ngữ bị ĐÔNG CỨNG (và lượng tử hóa 4-bit);
việc tinh chỉnh chỉ học hai ma trận hạng thấp `A`, `B`. Đầu ra cộng thêm một số hạng hạng
thấp được nhân tỉ lệ `α/r`.

**Biến:**
- `x`: vector đầu vào của một lớp tuyến tính; `h`: đầu ra.
- `W_0 ∈ ℝ^(d_out×d_in)`: trọng số gốc (4-bit NF4, đông cứng).
- `A ∈ ℝ^(r×d_in)`, `B ∈ ℝ^(d_out×r)`: ma trận hạng thấp học được, ghép lại
  `B A ∈ ℝ^(d_out×d_in)` đúng kích thước `W_0`; `r = 16` (hạng), `α = 32` (hệ số tỉ lệ).
  Phải tách `d_in`, `d_out` (KHÔNG dùng chung một `d`) vì nhiều lớp của Vicuna không vuông —
  ví dụ gate/up chiếu `4096 → 11008`, down chiếu `11008 → 4096`. Khởi tạo `B = 0`, `A` ngẫu
  nhiên Gauss → đầu phiên tinh chỉnh `B A = 0`, đầu ra trùng hệt mô hình gốc. Số tham số học
  giảm mạnh so với tinh chỉnh toàn phần [Hu 2021; Dettmers 2023].

**Code:** `khoaluan/code/train_v2.py` đặt `--bits 4 --quant_type nf4 --lora_r 16 --lora_alpha 32
--lora_dropout 0.05`; gọi `llava.train.train`. Tập lớp áp LoRA = `find_all_linear_names`
(`source/LLaVA/llava/train/train.py:200,1022`) → mọi lớp tuyến tính của LLM
(q,k,v,o,gate,up,down), KHÔNG đụng nhánh thị giác / projector / lm_head.

**Trực giác:** tiết kiệm tài nguyên (hợp với ràng buộc máy yếu, 2 epoch) và bảo toàn năng
lực ECG mà PULSE học sẵn.

---

## 7. Token thị giác (Mục 3 / Hình 3.1)

**Công thức:** mỗi ô 336×336 → `(336/14)² = 24×24 = 576` token; projector `1024 → 4096`;
3 ô → `3 × 576 = 1728` token thị giác.

**Ý nghĩa:** CLIP ViT-L/14 chia ảnh 336px theo patch 14px thành lưới 24×24 = 576 mảnh
(bỏ token CLS), mỗi mảnh thành một token 1024 chiều; lớp chiếu mlp2x_gelu ánh xạ sang 4096
chiều của mô hình ngôn ngữ. Đặc trưng thị giác lấy từ tầng GẦN CUỐI của ViT
(`mm_vision_select_layer = -2`, mặc định của LLaVA), không phải tầng cuối. AnyRes cho 3 ô
(2 cục bộ + 1 toàn cục) → 1728 token.

**Code:** cấu hình từ `PULSE-7B/config.json` (`mm_vision_tower` clip-vit-large-patch14-336,
`mm_projector_type` mlp2x_gelu, `mm_hidden_size` 1024, `mm_vision_select_layer` -2,
`mm_patch_merge_type` flat). Xử lý ảnh: `source/LLaVA/llava/mm_utils.py`
(`process_anyres_image`); số ô 3 là kết quả của tỉ lệ ảnh ở thì chạy, không do config quy định.

**Trực giác:** 1728 token thị giác + token văn bản + khối nhãn phải lọt cửa sổ ngữ cảnh 4096
→ giới hạn độ phân giải tiling (Mục 5.2).

---

## 8. (Tùy chọn — KHÔNG nằm trên đường train chính) CE có trọng số lớp

**Công thức:**
- `L = λ · CE_prefix + CE_label`
- Trọng số lớp cân bằng: `w_k = (1 − β) / (1 − β^{n_k})`, chuẩn hóa để `Σ w = số lớp` [Cui 2019]

**Ý nghĩa:** Một biến thể ablation: tách entropy chéo trên token khung/lập luận (`CE_prefix`)
khỏi entropy chéo trên token nhãn (`CE_label`), và nhân token nhãn với trọng số cân bằng theo
"số mẫu hiệu dụng".

**Biến:** `n_k` = số mẫu lớp `c_k`; `β` (mặc định 0.999) điều khiển mức cân bằng; `λ` cân
giữa hai thành phần.

**Code:** `khoaluan/code/weighted_ce.py` (`class_balanced_weights`, `weighted_ce_loss`).

**CẢNH BÁO QUAN TRỌNG (phải nói đúng khi bảo vệ):** đây CHỈ là ablation tham chiếu. Mô hình
cuối (epoch-2, mọi số headline) được huấn luyện bằng **entropy chéo next-token GỐC của PULSE**,
KHÔNG dùng hàm có trọng số này và KHÔNG dùng ASL. Đừng trình bày công thức này như loss thật.

---

## Tóm tắt đối chiếu code

| Công thức | Code xác minh |
|---|---|
| (1.2) ngưỡng | `eval_v2.py`: `scores >= threshold` |
| Teacher-forced softmax 2 chiều | `teacher_forced.py`: `softmax([z_0, z_1])[1]` |
| Nhãn 2 tầng (≥50 / hiện diện) | `labels.py` + `scp_tasks.py` |
| QLoRA r=16, α=32, target | `train_v2.py` + `source/LLaVA/.../train.py` (find_all_linear) |
| 576 token / 1024→4096 / 1728 | `PULSE-7B/config.json` + `mm_utils.py` |
| CE có trọng số (ablation) | `weighted_ce.py` — KHÔNG trên đường train chính |
