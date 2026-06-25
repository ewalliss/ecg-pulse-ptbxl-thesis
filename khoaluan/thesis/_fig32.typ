#set page(width: 16cm, height: auto, margin: 8mm)
#set text(font: "Times New Roman", size: 13pt, lang: "vi")

#set align(center)
#let cell(body, fill: white) = box(stroke: 0.6pt, inset: (x: 6pt, y: 5pt), radius: 2pt, fill: fill)[#set text(size: 9.5pt); #body]
#stack(dir: ttb, spacing: 9pt,
  cell([⟨ảnh ECG⟩ + ⟨câu hỏi tác vụ $q_t$⟩ + `[REASONING] … [LABELS]`], fill: luma(238)),
  text(13pt)[↓#h(0.3em)_nối tiếp khối nhãn cố định (token điền "0", độc lập nhãn thật)_],
  grid(columns: 8, column-gutter: 4pt, row-gutter: 4pt, align: horizon,
    cell([`NORM`]), cell([`:`], fill: rgb("#ffe0e6")), cell([`0`], fill: luma(232)),
    cell([`MI`]), cell([`:`], fill: rgb("#ffe0e6")), cell([`0`], fill: luma(232)),
    cell([`…`]), cell([`(K mã)`], fill: luma(245)),
  ),
  text(13pt)[↑ ô tô hồng = vị trí đọc xác suất],
  block(width: 86%)[#set text(size: 9pt); Tại mỗi ô ":" (ngay TRƯỚC ô số), đọc phân phối token kế tiếp của mô hình và lấy $P(y_k = 1) = "softmax"(z_("id0"), z_("id1"))_1$. Token điền "0" cố định cho mọi mẫu và mọi lớp nên không mang thông tin nhãn thật; toàn bộ tín hiệu phân biệt đến từ ảnh ECG.],
)
