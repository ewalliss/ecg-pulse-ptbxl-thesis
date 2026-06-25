#set page(width: 16cm, height: auto, margin: 8mm)
#set text(font: "Times New Roman", size: 13pt, lang: "vi")

#let node(body, fill: luma(245)) = box(width: 100%, stroke: 0.7pt, radius: 3pt, inset: 7pt, fill: fill)[
  #set text(size: 9.5pt); #set align(center); #body
]
#let down = align(center)[#text(size: 13pt)[↓]]
#set align(center)
#grid(columns: (1fr, 1fr), column-gutter: 1.2cm, row-gutter: 5pt,
  node([*Ảnh ECG* \ 12 chuyển đạo → ảnh RGB]),
  node([*Câu hỏi tác vụ* $q_t$ \ chẩn đoán / nhịp / hình thái]),
  down, down,
  node([CLIP ViT-L/14\@336 \ _bộ mã hóa thị giác (đông cứng)_], fill: luma(232)),
  node([Token embedding \ (Vicuna tokenizer)]),
  down, [],
  node([Lớp chiếu MLP \ (mlp2x_gelu)]),
  [],
)
#v(3pt)
#align(center)[#text(size: 13pt)[↘ #h(3.2cm) ↙]]
#block(width: 78%)[
  #node([*Vicuna-7B + QLoRA 4-bit* \ NF4, $r = 16$, $alpha = 32$ — _tinh chỉnh, đông cứng phần còn lại_], fill: luma(232))
  #down
  #node([Khối nhãn dạng văn bản \ `[REASONING] … [LABELS]` CODE: 0/1 …])
  #down
  #node([*Đọc xác suất* $P(y_k = 1)$ tại vị trí dự đoán \ (giao thức teacher-forced, Mục 3.6)], fill: luma(240))
]
