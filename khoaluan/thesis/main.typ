// ============================================================
//  main.typ — Khóa luận tốt nghiệp (Typst), bản dạng code, mỗi chương 1 file.
//  Engine: Typst 0.13.x.  Trích dẫn IEEE qua refs.bib.
//  Build:  typst compile main.typ KhoaLuan_ECG_PULSE.pdf
//  Văn phong: chuẩn GVHD PGS.TS Lý Quốc Ngọc (xem THESIS_STYLE_GUIDE.md).
// ============================================================

// ---------- Cấu hình trang & chữ ----------
#set page(
  paper: "a4",
  margin: (left: 3cm, right: 2cm, top: 2cm, bottom: 2cm),
)
#set text(font: "Times New Roman", size: 13pt, lang: "vi")
#set par(justify: true, leading: 1em, first-line-indent: 1.25em)

// ---------- Đánh số & kiểu tiêu đề ----------
// Cấp 1 = "CHƯƠNG N"; cấp con = N.k, N.k.j
#let chapter-numbering = (..nums) => {
  let p = nums.pos()
  if p.len() == 1 { "CHƯƠNG " + str(p.at(0)) }
  else { p.map(str).join(".") }
}

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  set align(center)
  set text(size: 14pt, weight: "bold")
  block(below: 1.2em)[
    #if it.numbering != none [#counter(heading).display(it.numbering) \ #v(0.3em)]
    #upper(it.body)
  ]
}
#show heading.where(level: 2): it => {
  set text(size: 13pt, weight: "bold")
  block(above: 1.1em, below: 0.6em)[#it]
}
#show heading.where(level: 3): it => {
  set text(size: 13pt, weight: "bold", style: "italic")
  block(above: 0.9em, below: 0.5em)[#it]
}

// ---------- Caption: Bảng TRÊN, Hình DƯỚI ----------
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: image): set figure.caption(position: bottom)
#set figure(supplement: none, numbering: none)   // nhãn "Bảng 4.1:"/"Hình 3.1:" ghi thủ công trong caption
#show figure.caption: set text(size: 12pt)

// ============================================================
//  TRANG BÌA
// ============================================================
#set page(numbering: none)
#align(center)[
  #text(13pt)[ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH]
  #v(0.2em)
  #text(14pt, weight: "bold")[TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN]
  #v(3cm)
  #text(13pt, weight: "bold")[NGUYỄN HUỲNH HẢI ĐĂNG — PHAN VŨ GIA HÂN]
  #v(2cm)
  #block(width: 90%)[#text(16pt, weight: "bold")[
    PHÂN LỚP BỆNH TIM ĐA PHƯƠNG THỨC TRÊN ẢNH ĐIỆN TÂM ĐỒ (ECG):
    TỪ HỎI–ĐÁP THỊ GIÁC CỦA PULSE SANG BỘ PHÂN LỚP THEO TÁC VỤ TRÊN PTB-XL
  ]]
  #v(2cm)
  #text(14pt, weight: "bold")[KHÓA LUẬN TỐT NGHIỆP]
  #v(4cm)
  #text(13pt)[TP. HỒ CHÍ MINH, NĂM 2026]
]
#pagebreak()

#align(center)[
  #text(13pt)[ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH]
  #v(0.2em)
  #text(14pt, weight: "bold")[TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN]
  #v(2cm)
  #text(13pt, weight: "bold")[NGUYỄN HUỲNH HẢI ĐĂNG — PHAN VŨ GIA HÂN]
  #v(1.5cm)
  #block(width: 90%)[#text(15pt, weight: "bold")[
    PHÂN LỚP BỆNH TIM ĐA PHƯƠNG THỨC TRÊN ẢNH ĐIỆN TÂM ĐỒ (ECG):
    TỪ HỎI–ĐÁP THỊ GIÁC CỦA PULSE SANG BỘ PHÂN LỚP THEO TÁC VỤ TRÊN PTB-XL
  ]]
  #v(1em)
  #text(13pt)[Ngành: Thị giác máy tính]
  #linebreak()
  #text(13pt)[Mã số ngành: #box(width: 3cm, repeat[.])]
  #v(1em)
  #text(14pt, weight: "bold")[KHÓA LUẬN TỐT NGHIỆP]
  #v(1.5em)
  #text(13pt, weight: "bold")[GIẢNG VIÊN HƯỚNG DẪN]
  #linebreak()
  #text(13pt)[PGS.TS Lý Quốc Ngọc]
  #v(3cm)
  #text(13pt)[TP. HỒ CHÍ MINH – NĂM 2026]
]

// ============================================================
//  PHẦN ĐẦU (không đánh số chương) + MỤC LỤC
// ============================================================
#set page(numbering: "i")
#counter(page).update(1)
#set heading(numbering: none)

#include "chapters/00-front-matter.typ"

#pagebreak()
#heading(level: 1, outlined: false)[Mục lục]
#outline(title: none, depth: 3, indent: auto)

#pagebreak()
#heading(level: 1, outlined: false)[Danh mục các hình vẽ, đồ thị]
#outline(title: none, target: figure.where(kind: image))

#pagebreak()
#heading(level: 1, outlined: false)[Danh mục các bảng số liệu]
#outline(title: none, target: figure.where(kind: table))

// ============================================================
//  THÂN BÀI (đánh số chương Ả Rập)
// ============================================================
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#counter(heading).update(0)
#set heading(numbering: chapter-numbering)

#include "chapters/01-mo-dau.typ"
#include "chapters/02-tong-quan.typ"
#include "chapters/03-phuong-phap.typ"
#include "chapters/04-ket-qua.typ"
#include "chapters/05-ket-luan.typ"

// ============================================================
//  TÀI LIỆU THAM KHẢO (IEEE)
// ============================================================
#set heading(numbering: none)
#bibliography("refs.bib", title: "TÀI LIỆU THAM KHẢO", style: "ieee")

// ============================================================
//  PHỤ LỤC
// ============================================================
#include "chapters/99-phu-luc.typ"
