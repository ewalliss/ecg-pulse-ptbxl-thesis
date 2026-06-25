#set page(width: 24cm, height: auto, margin: 12mm)
#set text(font: "Helvetica", size: 11pt, lang: "en")
#set par(leading: 0.85em)

// ---------- palette ----------
#let c_img  = rgb("#e0883a")
#let c_vis  = rgb("#bcd6ef")
#let c_proj = rgb("#c2dbb2")
#let c_llm  = rgb("#e8dcc0")
#let c_txt  = rgb("#cfe6c4")
#let c_head = rgb("#a8d8d0")
#let c_cat  = rgb("#f0c267")
#let c_frz  = rgb("#d9d9d9")
#let c_tr   = rgb("#ffd8a8")

#let node(title, sub: none, fill: c_vis, w: 7cm, tag: none) = box(
  width: w, fill: fill, stroke: 0.7pt + luma(70), radius: 4pt, inset: (x: 11pt, y: 9pt),
)[
  #set align(center)
  #grid(columns: (1fr,), row-gutter: 2pt,
    [#text(weight: "bold", size: 10pt)[#title] #if tag != none [#h(3pt) #box(fill: white, stroke: 0.5pt, inset: 2pt, radius: 2pt)[#text(size: 7pt)[#tag]]]],
    if sub != none { text(size: 8pt)[#sub] },
  )
]
#let sh(label) = align(center)[#text(size: 8pt, fill: rgb("#3a6ea5"))[#label]]
#let ar = align(center)[#text(size: 13pt)[↓]]
#let arl(label) = stack(dir: ttb, spacing: 4pt, ar, sh(label))

// a single token square
#let tok(label, fill) = box(width: 1.5cm, height: 1.05cm, fill: fill, stroke: 0.5pt + luma(80), radius: 3pt, inset: 4pt)[
  #set align(center + horizon); #text(size: 8.5pt)[#label]
]

#align(center)[#text(size: 14pt, weight: "bold")[Cơ chế hợp nhất thị giác–ngôn ngữ (early fusion, joint self-attention)]]
#v(2pt)
#align(center)[#text(size: 9pt)[Token ảnh được CHIẾU về đúng chiều 4096 của mô hình ngôn ngữ rồi GHÉP thẳng vào chuỗi văn bản — không có nhánh cross-attention riêng.]]
#v(10pt)

// ============================================================
//  TWO INPUT STREAMS -> ALIGN TO 4096
// ============================================================
#grid(columns: (1fr, 1fr), column-gutter: 16pt, align: center + top,
  // ----- vision stream -----
  stack(dir: ttb, spacing: 8pt,
    node("Ảnh ECG", sub: [12 chuyển đạo render RGB, 1344×672], fill: c_img),
    arl[anyres → 3 ô 336×336],
    node("CLIP ViT-L/14 @336 (đông cứng)", sub: [patch 14 → 24×24 = 576 token/ô · tầng −2], fill: c_frz, tag: "frozen"),
    arl[3 × 576 = 1728 token × #text(fill: rgb("#b00"))[*1024*]],
    node("Projector mlp2x_gelu (đông cứng)", sub: [Linear 1024→4096 · GELU · Linear 4096→4096], fill: c_frz, tag: "căn chiều"),
    arl[#text(fill: rgb("#0a0"))[*1728 token × 4096*]],
  ),
  // ----- text stream -----
  stack(dir: ttb, spacing: 8pt,
    node([Văn bản: instruction $q_t$ + đáp án], sub: [`<image>\n` câu hỏi + khối `[LABELS]`], fill: c_img),
    arl[chuỗi token (có 1 token `<image>`)],
    node("Embedding table (Vicuna)", sub: [tra bảng 32000 × 4096], fill: c_frz),
    arl[],
    node([Token văn bản], sub: [mỗi token một vector], fill: c_txt),
    arl[#text(fill: rgb("#0a0"))[*T token × 4096*]],
  ),
)

#v(8pt)
#align(center)[#text(13pt)[↓ #h(4pt) #text(weight: "bold")[điểm FUSION — cùng chiều 4096, nối thành MỘT chuỗi]]]
#v(6pt)

// ============================================================
//  CONCAT SEQUENCE (token strip)
// ============================================================
#align(center)[
  #box(stroke: 0.8pt + luma(60), radius: 5pt, inset: 13pt, fill: luma(250))[
    #stack(dir: ttb, spacing: 9pt,
      align(left)[#text(9pt)[Thay token `<image>` bằng 1728 token ảnh tại đúng vị trí của nó:]],
      grid(columns: 9, column-gutter: 5pt, row-gutter: 5pt,
        tok("img 1", c_vis), tok("img 2", c_vis), tok("…", c_vis), tok("img 1728", c_vis),
        tok("Based", c_txt), tok("…", c_txt), tok([`IMI`], c_txt), tok([`:`], c_txt), tok([?], c_head),
      ),
      align(left)[#text(7.5pt)[#text(fill: rgb("#3a6ea5"))[xanh = token ảnh] · #text(fill: rgb("#3a7a2a"))[lục = token văn bản] · #text(fill: rgb("#2a8a80"))[ô đọc xác suất] — tổng (1728 + T) vector, đều 4096 chiều]],
    )
  ]
]
#v(4pt)
#align(center)[#text(8.5pt)[Mặt nạ NHÂN QUẢ (causal): ô `?` (sau `IMI:`) nhìn về TRÁI → attend được mọi token ảnh + câu hỏi phía trước. Token ảnh đứng trước, không thấy đáp án phía sau.]]

#v(8pt)
#align(center)[#ar]
#v(2pt)

// ============================================================
//  32 DECODER LAYERS (sequential depth, joint across tokens)
// ============================================================
#align(center)[
  #box(stroke: 0.9pt + luma(40), radius: 5pt, inset: 14pt, width: 94%, fill: c_llm.lighten(55%))[
    #align(center)[#text(11pt, weight: "bold")[Vicuna-7B — 32 lớp decoder (xếp chồng TUẦN TỰ theo độ sâu)]]
    #v(8pt)
    #grid(columns: (auto, auto, auto, auto, auto), column-gutter: 6pt, align: horizon + center,
      box(stroke: 0.6pt, radius: 3pt, inset: 5pt, fill: white)[#text(8.5pt)[Lớp 1]],
      text(12pt)[→],
      box(stroke: 0.6pt, radius: 3pt, inset: 5pt, fill: white)[#text(8.5pt)[Lớp 2]],
      text(12pt)[→ … →],
      box(stroke: 0.6pt, radius: 3pt, inset: 5pt, fill: white)[#text(8.5pt)[Lớp 32]],
    )
    #v(7pt)
    #box(stroke: 0.6pt + luma(90), radius: 4pt, inset: 12pt, width: 92%, fill: white)[
      #align(center)[#text(9pt, weight: "bold")[Bên trong MỖI lớp (ảnh + văn bản xử lý ĐỒNG THỜI — joint, không tuần tự theo modal)]]
      #v(8pt)
      #grid(columns: (1fr, auto, 1fr, auto, 1fr), align: horizon + center, column-gutter: 8pt,
        box(fill: c_llm, stroke: 0.5pt, radius: 3pt, inset: 8pt)[#text(8.5pt)[RMSNorm]],
        text(11pt)[→],
        box(fill: c_llm, stroke: 0.5pt, radius: 3pt, inset: 8pt)[#text(8.5pt)[Self-Attention (RoPE, causal) #box(fill: c_tr, inset: 2pt)[#text(7.5pt)[+LoRA]] \ #text(7.5pt)[mọi token ảnh+text CÙNG attend]]],
        text(11pt)[→],
        box(fill: c_llm, stroke: 0.5pt, radius: 3pt, inset: 8pt)[#text(8.5pt)[RMSNorm + SwiGLU MLP #box(fill: c_tr, inset: 2pt)[#text(7.5pt)[+LoRA]] \ #text(7.5pt)[4096 → 11008 → 4096]]],
      )
      #v(7pt)
      #align(center)[#text(8pt, style: "italic")[chiều ẩn giữ nguyên 4096 xuyên suốt · base 4-bit NF4 đông cứng, chỉ LoRA (r=16) học]]
    ]
  ]
]

#v(7pt)
#align(center)[#arl[(1728 + T) × 4096]]
#align(center)[#node("LM head → logits", sub: [Linear 4096 → 32000 (đông cứng)], fill: c_frz, w: 7cm)]
#v(3pt)
#align(center)[#arl[đọc cặp logit (id"0", id"1") tại mỗi ô nhãn]]
#align(center)[#node([Teacher-forced readout], sub: [$P(y_k=1) = e^(z_1)\/(e^(z_0)+e^(z_1))$ cho từng mã], fill: c_head, w: 7cm)]
