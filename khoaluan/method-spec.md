# method-spec v2 — PULSE từ thuần-VQA sang bộ phân lớp ECG đa nhãn task-decomposed có suy luận

> Hợp đồng toán + kiến trúc mà `khoaluan/code/` (hiện thực) và `khoaluan/paper/` (trình bày) cùng tuân.
> Mọi công thức/claim có trích dẫn `[Key YYYY]` khớp một entry trong `khoaluan/papers/manifest.json`.
> Ràng buộc bất biến: KHÔNG sửa `source/LLaVA` (bản cũ ASL-Gen); vision/projector đông cứng.
> Phiên bản: v1 (FROZEN contract @ 2026-06-11).

---

## 1. Notation & ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| `x` | ảnh ECG 12-đạo-trình (plot từ tín hiệu PTB-XL), đầu vào thị giác |
| `q_t` | prompt câu hỏi của task `t` (text), `t ∈ T = {diag, rhythm, form}` |
| `C_t` | tập mã nhãn của task `t`; \|C_diag\|=44, \|C_rhythm\|=12, \|C_form\|=19 |
| `y_t ∈ {0,1}^{\|C_t\|}` | vector nhãn nhị phân ground-truth của mẫu cho task `t` |
| `a_t` | chuỗi answer sinh ra = [reasoning trace] + [khối nhãn] |
| `θ` | tham số LoRA (chỉ phần LLM được cập nhật) |
| `f_θ` | mô hình sinh tự hồi quy PULSE-7B + LoRA |
| `z_i ∈ R^{\|V\|}` | logit trên từ vựng `V` tại bước sinh `i` |
| `p_i = softmax(z_i)` | phân phối token tại bước `i` |
| `τ⁺, τ⁻` | id token nhãn dương `"1"` và âm `"0"` (sau prefix-space của LLaMA SP) |
| `n_k` | số mẫu dương của lớp `k` trong train (cho class-balanced) |
| `w_k` | trọng số lớp `k` |
| `β` | siêu tham số class-balanced (mặc định 0.999) |
| `λ` | hệ số trộn giữa CE prefix (reasoning) và CE nhãn (mặc định 1.0) |
| `ℓ_diag` | ngưỡng likelihood cho task diag (mặc định 50, theo PTB-XL) |

Quy ước: macro-AUC threshold-free là metric đầu bảng, báo RIÊNG theo từng task `t` [Strodthoff 2020].

---

## 2. Kiến trúc v2

Nền tảng giữ nguyên PULSE-7B = LLaVA-v1.6-Vicuna-7B: vision encoder CLIP ViT-L/14@336 → projector MLP → LLM Vicuna-7B, huấn luyện bằng next-token cross-entropy với token chỉ thị bị mask [PULSE 2024].

Quyết định kiến trúc cho v2:
- **Đông cứng vision encoder + projector** (stage-1). Lý do: PULSE đã instruction-tune vision trên >1M mẫu ECG; ta chỉ cần thích nghi phần ngôn ngữ. GEM cũng SFT trên nền PULSE-7B mà không phá vision, đạt PTB-XL Super AUC 83.4 [GEM 2025].
- **Chỉ cập nhật LoRA trên LLM** (QLoRA 4-bit NF4, r=16, alpha=32). Tham số `θ` = adapter LoRA.
- **KHÔNG sửa core** trong `source/LLaVA`: toàn bộ logic v2 (dựng dữ liệu, loss, eval) đặt trong `khoaluan/code/`, gọi PULSE ở mức runtime (read-only). Đây là ranh giới bất biến của dự án.

Khác biệt cốt lõi so với PULSE gốc: PULSE là hệ **VQA thuần** (sinh văn bản trả lời), không phải bộ phân lớp đa nhãn có điểm số hiệu chuẩn. v2 biến năng lực VQA này thành **bộ phân lớp đa nhãn task-decomposed** bằng cách (i) chuẩn hoá định dạng output để trích xác suất theo lớp, (ii) thêm reasoning bám đặc trưng, (iii) huấn luyện bằng CE có cân bằng lớp.

---

## 3. Task decomposition & quy tắc gán nhãn

PTB-XL gán mỗi ECG nhiều phát biểu SCP-ECG, chia 3 nhóm: 44 diagnostic, 19 form, 12 rhythm (4 mã chồng lấn diag/form) [Wagner 2020]. Benchmark chuẩn vận hành PTB-XL như **6 task riêng**, củng cố việc tách diagnostic/form/rhythm thành các bài multi-label độc lập [Strodthoff 2020]. PULSE cũng phân rã PTB-XL theo task (Feature/Rhythm/Morphology/Report), không học một bộ phân lớp 71-way phẳng [PULSE 2024].

Định nghĩa 3 task `T = {diag, rhythm, form}`, mỗi task một prompt `q_t` và tập nhãn `C_t` riêng.

**Quy tắc gán nhãn nhị phân từ `scp_codes` (dict mã→likelihood) của PTB-XL:**

- Task **diag** (có likelihood): với `k ∈ C_diag`,
  `y_diag[k] = 1` nếu `k ∈ scp_codes` và `scp_codes[k] ≥ ℓ_diag` (mặc định `ℓ_diag = 50`); ngược lại `0`.
- Task **rhythm** và **form** (likelihood = 0 theo thiết kế PTB-XL): với `k ∈ C_rhythm ∪ C_form`,
  `y_t[k] = 1` nếu `k ∈ scp_codes` (**presence-based**); ngược lại `0`.

Lý do tách quy tắc (chống lỗi của bản cũ): form và rhythm được PTB-XL gán likelihood = 0 [Wagner 2020]; nếu áp một ngưỡng `likelihood ≥ 50` đồng loạt lên cả 71 mã thì **mọi nhãn form/rhythm bị loại sạch** (reject `label_rule_drops_form_rhythm`). Vì thế diag dùng ngưỡng likelihood riêng, còn rhythm/form dùng presence.

---

## 4. Định dạng I/O sinh + reasoning

Với mỗi mẫu `(x, q_t)`, mô hình sinh answer `a_t` gồm hai phần, theo thứ tự:

```
[REASONING] <lập luận bám đặc trưng: nhịp, trục, sóng P/QRS/T, đoạn ST...>
[LABELS]
<code_1>: <0|1>
<code_2>: <0|1>
...
<code_{|C_t|}>: <0|1>
```

- **Khối nhãn** liệt kê đúng `|C_t|` dòng `code: 0|1` theo thứ tự cố định của `C_t`. Token nhãn dùng để trích xác suất là token `"1"`/`"0"` đứng ngay sau dấu `:` và trước newline (`τ⁺`, `τ⁻`). Định dạng `"code: 1/0"` được tham số hoá: biến thể thay thế là **"liệt kê chỉ các code dương"** (cho phép đổi ở `khoaluan/code` mà không phá hợp đồng) — ghi nhận đây là điểm ít chắc nhất (xem flag freeze).
- **Reasoning** đặt TRƯỚC khối nhãn để điều kiện hoá dự đoán. Cơ sở: tác giả PULSE khuyến nghị thêm dữ liệu suy luận từng bước [PULSE 2024]; RL thưởng CoT nâng accuracy ECG-QA 60.0→66.5 [COUNTS 2025]; GEM cho thấy reasoning bám đặc trưng đo được tương thích với hiệu năng cao [GEM 2025].
- **Nguồn supervision reasoning** (mặc định): tận dụng template ECG-QA để dựng câu hỏi/định dạng [ECG-QA 2023]; tuỳ chọn nâng cấp: sinh rationale theo công thức GPT-4 self-instruct + GPT-4-as-judge của LLaVA-Med [LLaVA-Med 2023]. Mặc định ECG-QA (rẻ, không cần API).

---

## 5. Hàm loss

Mục tiêu chính là **cross-entropy next-token tự hồi quy** (proper scoring rule), KHÔNG dùng ASL.

**Vì sao bỏ ASL (loss bản cũ):** ASL được định nghĩa cho head phân biệt phát ra `K` logit **sigmoid độc lập** (một logit/nhãn), `L_tot = Σ_k L(σ(z_k), y_k)` với `γ⁻>γ⁺` và margin `m` [Ridnik 2021]. Nó không dành cho decoder sinh token tự hồi quy; ép ASL lên xác suất token `"1"/"0"` là lệch kiến trúc (reject `loss_math_invalid`). Trên thực nghiệm, không ECG-MLLM công bố nào dùng ASL; công thức thắng cuộc là CE next-token trên prompt phân rã task [PULSE 2024][GEM 2025].

**Loss v2.** Tách answer thành tập vị trí token reasoning `R` và tập vị trí token nhãn `L` (các token `"1"/"0"`):

```
L_total = λ · CE_prefix + CE_label
CE_prefix = (1/|R|) · Σ_{i∈R} −log p_i[a_i]              # CE chuẩn trên reasoning + khung
CE_label  = (1/|L|) · Σ_{i∈L} w_{k(i)} · ( −log p_i[a_i] )  # CE có trọng số lớp trên token nhãn
```

với `a_i` là token đích tại bước `i`, `k(i)` là lớp ứng với vị trí nhãn `i`.

**Trọng số class-balanced** [Cui 2019]:
```
w_k = (1 − β) / (1 − β^{n_k}),   β ∈ [0,1)  (mặc định β = 0.999)
```
chuẩn hoá để `Σ_k w_k = |C_t|`. Đây là tái trọng số "effective number of samples", mượt hơn nghịch đảo tần suất và cắm thẳng vào CE.

**Ablation (không phải mặc định): focal-CE** [Lin 2017]: nhân thêm `(1 − p_i[a_i])^γ` vào số hạng `CE_label`. Cảnh báo: focal là non-proper scoring rule [Mukhoti 2020], nên chỉ dùng như điều biến trên token nhãn, KHÔNG thay toàn bộ mục tiêu sinh.

**Cơ chế collapse cần khắc phục:** huấn luyện MLE/CE của decoder gán quá nhiều xác suất cho token tần suất cao; với PTB-XL mất cân bằng (NORM ~44%) [PTBXL-Imbalance 2022] điều này dẫn tới mode-collapse "luôn NORM" [Welleck 2020]. `w_k` + cân bằng dữ liệu + hiệu chuẩn ngưỡng (mục 6) là ba đòn bẩy chống collapse.

---

## 6. Giao thức trích điểm & đánh giá

**Trích xác suất theo lớp (token-marginalization).** Tại vị trí nhãn của code `k`, lấy logit bước tương ứng và marginalize trên hai token nhãn:
```
P(y_k = 1 | x, q_t) = softmax( [ z_i[τ⁻], z_i[τ⁺] ] )[1]
```
Đây là điểm mềm threshold-free dùng cho macro-AUC (không greedy, tránh artifact all-negative).

**Macro-AUC per-task.** Với mỗi task `t`, tính AUC theo từng lớp trên các lớp có cả dương lẫn âm trong test, rồi lấy trung bình (macro). Báo RIÊNG `AUC_diag`, `AUC_rhythm`, `AUC_form` [Strodthoff 2020]. Phụ: macro/micro-F1.

**Hiệu chuẩn ngưỡng theo lớp.** Quét xác suất trên val (fold 9) để chọn ngưỡng tối đa F1 cho từng lớp; ngưỡng thường < 0.5. Tách bạch với AUC (AUC không cần ngưỡng).

**Split chuẩn PTB-XL.** 8 fold train / fold 9 val / fold 10 test (chất lượng nhãn cao nhất ở fold 9–10) [Wagner 2020][Strodthoff 2020].

**Baseline.** PULSE-7B zero-shot là mốc chính (PTB-XL Super macro-AUC 82.4) [PULSE 2024]; ghi rõ "zero-shot" nghĩa là áp PULSE-7B không fine-tune thêm — PULSE đã thấy PTB-XL qua ECGInstruct, nên baseline "chưa biết ECG" thực sự là backbone LLaVA zero-shot (AUC 50.0) [PULSE 2024]. Báo cả hai mốc.

**Đánh giá reasoning.** Rubric LLM-judge có grounding kiểu GEM: feature-grounding / evidence-based reasoning / diagnostic-fidelity (mỗi trục 0–100) [GEM 2025], kèm BLEU/ROUGE cho phần văn bản.

---

## 7. Tham chiếu (map mục → nguồn trong manifest)

- Kiến trúc PULSE & CE next-token: [PULSE 2024]; nền frozen + reasoning grounded: [GEM 2025].
- Label space 71 = 44/19/12, likelihood semantics: [Wagner 2020]; 6-task benchmark + macro-AUC + split: [Strodthoff 2020].
- Task-decomposed precedent & reasoning supervision: [PULSE 2024][ECG-QA 2023][LLaVA-Med 2023]; CoT↑accuracy: [COUNTS 2025]; MedTVT chain-of-evidence: [MedTVT 2025].
- Loss: bỏ ASL vì head sigmoid K-logit [Ridnik 2021]; class-balanced [Cui 2019]; focal [Lin 2017]; focal non-proper [Mukhoti 2020]; collapse cơ chế [Welleck 2020]; imbalance PTB-XL [PTBXL-Imbalance 2022].
- Các mô hình đối chiếu (related work): [MERL 2024][HeartLang 2025][Tang 2024][Nandakishor 2025][ECG-Chat 2024].

Mọi `[Key YYYY]` trên khớp một entry trong `khoaluan/papers/manifest.json` (kiểm bằng `khoaluan/checks/check_method_spec.py`).
